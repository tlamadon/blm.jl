using Distributions
using DataFrames
using Printf
using Dates




"""
    Event study BLM model for movers
     - static version
     - dynamic version
"""
mutable struct DistributionModelES
    nl::Int  # number of worker types
    nk::Int  # number of firm types

    # type probabilities
    pk1 :: Array{Float64,3}

    # wages for first time we see worker
    A1::Array{Float64,2}  
    S1::Array{Float64,2}  

    # wages for coming out of unemployment
    A2::Array{Float64,2}  
    S2::Array{Float64,2}
    
    # distributions over pair of moves
    NNm::Array{Float64,2}

    # dirichlet prior
    dprior :: Float64
end

function DistributionModelES(nl::Int,nk::Int)
  
    pk1 = zeros(nk,nk,nl)
    ddir = Distributions.Dirichlet( repeat([2.0], nl))
    for k1 in 1:nk, k2 in 1:nk 
        pk1[k1,k2,:] = rand(ddir) 
    end

    # wages for day 1
    A1    = 0.9 .* (1 .+ 0.5 .* randn(nk,nl) )
    S1    = 0.3 .* (1 .+ 0.5 .* rand(nk,nl) )

    # wages for day 1
    A2    = 0.9 .* (1 .+ 0.5 .* randn(nk,nl) )
    S2    = 0.3 .* (1 .+ 0.5 .* rand(nk,nl) )

    NNm = ones(nk,nk)/(nk*nk)

    DistributionModelES(nl,nk, pk1, A1, S1, A2, S2, NNm, 1.0)
end

# simply copy method for DistributionalModel
Base.copy(x::DistributionModelES) = DistributionModelES([ copy(getfield(x, k)) for k ∈ fieldnames(DistributionModelES)]...)




# Using the model, simulates a dataset
# The timing is important, the event happens at the end of the period
# hence if a current period is u2e there will be no wage in the current period, as well as no firm.
function simulate(model::DistributionModelES, nn) 

    J1  = zeros(Int64,nn)     # firm id when entering the period
    J2  = zeros(Int64,nn)     # firm id when entering the period
    L   = zeros(Int64,nn)     # firm id when entering the period 
    W1  = zeros(Float64,nn)   # wage if employed
    W2  = zeros(Float64,nn)   # wage if employed

    nl  = model.nl
    nk  = model.nk

    for i in 1:nn 
        k1 = sample( 1:model.nk )
        k2 = sample( 1:model.nk )

        #  -----  draw worker and firm ------- #
        l    = wsample( 1:model.nl , model.pk1[k1,k2,:])

        #  -----  draw wages -------#
        W1[i] = model.A1[k1,l] + model.S1[k1,l] * randn()  # wage next period
        W2[i] = model.A2[k2,l] + model.S2[k2,l] * randn()  # wage next period
        J1[i] = k1
        J2[i] = k2
        L[i] = l

    end # end of loop on i

    dd = DataFrame(i=1:nn, j1=J1, j2=J2, w1=W1, w2=W2, l=L)
    return(dd)
end

# computes the likelihood for
# each observation i in the data, using the given model
# for a given value of the unobserved worker heterogeneity
# @fixme make sure input is sorted by worker and time
function likelihood!(lpt_vec::Array{Float64,2},
                        W1::Array{Float64,1}, 
                        J1::Array{Int64,1},
                        W2::Array{Float64,1}, 
                        J2::Array{Int64,1},
                        l::Int64,
                        model::DistributionModelES)
  
    for ii in 1:length(W1)
        j1 = J1[ii]
        j2 = J2[ii]
        lpt = log(model.pk1[j1,j2,l]) 
        lpt += lognormpdf( W1[ii], model.A1[j1,l], model.S1[j1,l])
        lpt += lognormpdf( W2[ii], model.A2[j2,l], model.S2[j2,l]) 
        lpt_vec[ii,l] = lpt 
    end
  
    return()
end

function distributional_posteriors(dm::DistributionModelES, data::DataFrame)
    nn = size(data,1)
    nl = dm.nl
    
    lik_all = zeros(nn,nl)
    for l in 1:nl
        likelihood!(lik_all, data.w1, data.j1, data.w2, data.j2, l, dm)
    end

    qs = zero(lik_all)
    for i in 1:nn
        qs[i,:] = exp_prob(lik_all[i,:], 1e-10)
    end

    return(qs,lik_all)
end


function distributional_posteriors!(lik::Array{Float64,2}, qs::Array{Float64,2}, dm::DistributionModelES, data::DataFrame)
    nn = size(data,1)
    nl = dm.nl
    
    for l in 1:nl
        likelihood!(lik, data.w1, data.j1, data.w2, data.j2, l, dm)
    end

    qs .= lik
    sigmoid_row!(qs, 1e-9)
end


function check_em_step(dm::DistributionModelES,key,val,data)
    qs,lk = distributional_posteriors(dm, data)
    dm2 = copy(dm)
    setfield!(dm2,key,copy(val))
    qs2,lk2 = distributional_posteriors(dm2, data)
    ΔQ = sum( qs .* lk2) - sum( qs .* lk)
    ΔH = - sum(qs .* log.(qs2)) +  sum(qs .* log.(qs))  
    ΔL = ΔQ+ΔH

    str = ""
    if (ΔQ < 0) | (ΔH <0 )
        str = " <<<<<"
    end

    @printf("[check][%s] ΔQ=%2.2e ΔH=%2.2e ΔL=%2.2e %s\n", string(key), ΔQ, ΔH, ΔL, str)
end

function aggXwX(
    X::Array{Float64,1}, 
    J::Array{Int64,1}, 
    weight::Array{Float64,2}, 
    S::Array{Float64,2},
    nl::Int, nk::Int)       

    XwXv = zeros(nk,nl)
    for ii in 1:length(X), l in 1:nl
        XwXv[J[ii],l] += weight[ii,l] / S[J[ii],l]^2 * X[ii] 
    end    
    return(XwXv)
end

function aggXwX_var(
    X::Array{Float64,1},
    J::Array{Int,1} ,
    M::Array{Float64,2},
    weight::Array{Float64,2}, 
    S::Array{Float64,2},
    nl::Int, nk::Int)     

    XwXv = zeros(nk,nl)
    for ii in 1:length(X), l in 1:nl
        XwXv[J[ii],l] += weight[ii,l] / S[J[ii],l]^2 * (X[ii] - M[J[ii],l])^2 
    end    
    return(XwXv)
end


"""
    Run the em algorithm on the passed DistributionalModel using the given data
"""
function distributional_em!(dm::DistributionModelES, data::DataFrame, maxiter::Int, tol::Float64 = 1e-8; update_level = true, constrained=nothing, iterprint=50, msg="")

    nl = dm.nl
    nk = dm.nk
    nn = size(data.i,1)
    lik_all = zeros(nn,nl)
    qs      = zero(lik_all)
    qs_last = zero(lik_all)
    qs_it   = zeros(size(data,1), nl)


    total_lik = 0

    Qc = -Inf
    Ql = -Inf
    Hc = -Inf
    Hl = -Inf

    D1 = ones(length(data.w1))
    last_lik = -Inf
    cs = nothing #ConstraintLinear(nl,nk)
    if constrained=="fixb"
        cs = ConstraintFixB(nk=nk, nl=nl, nt=2)
    elseif constrained=="para"
        cs = ConstraintPara(nk=nk, nl=nl, nt=2)
    elseif constrained=="linear"
        cs = ConstraintLinear(nk=nk, nl=nl, nt=2)
    end

    dm_tmp = copy(dm)

    # update NNm
    dm.NNm .= 0
    for k1 in 1:nk, k2 in 1:nk
        dm.NNm[k1,k2] = sum( (data.j1 .== k1) .* (data.j2 .== k2))
    end

    # Expectation Maximization loop
    for rep in 1:(maxiter+1)

        # ----------  E-step ---------------
        Ql = sum( qs .* lik_all) 
        qs_last .= qs
        distributional_posteriors!(lik_all, qs, dm, data)
        Qc = sum( qs_last .* lik_all) 

        ΔH = - sum(qs_last .* log.(qs ./ qs_last))
        #ΔH2 = - sum(qs_last .* log.(qs)) +  sum(qs_last .* log.(qs_last))
        
        # change in prior contribution
        ΔP = (1 - dm.dprior) * sum(log.(qs) - log.(qs_last))

        # if (ΔH<0) 
        #     return(qs,qs_last)
        # end

        total_lik = sum( qs .* lik_all) - sum(qs .* log.(qs)) + (1.0 - dm.dprior) * sum(log.(qs))
        if (rep>1) & (mod(rep,iterprint)==0)
            if (ΔH + (Qc-Ql))<0
                str = string(msg,"!!!")
            else
                str = msg
            end
            @printf("[%s][%04i] lik=%+2.4e Δlik=%+2.2e ΔH=%+2.2e ΔQ=%+2.2e ΔP=%+2.2e (%s)\n",
                        Dates.format(now(), "HH:MM:SS"), rep, 
                        round(total_lik/nn,digits=5),
                        ΔH + (Qc-Ql), ΔH, Qc-Ql, ΔP, str)
        end

        if (abs(ΔH + (Qc-Ql)) < tol) | (rep==maxiter+1)
            @printf("[%s][%04i][final] lik=%+2.4e Δlik=%+2.2e ΔH=%+2.2e ΔQ=%+2.2e ΔP=%+2.2e (%s)\n",
                Dates.format(now(), "HH:MM:SS"), rep, 
                round(total_lik/nn,digits=5),
                ΔH + (Qc-Ql), ΔH, Qc-Ql, ΔP, msg)
            break
        end
        
        if isnan(total_lik)
            return(qs, lik_all)
        end

        # ----------  M-step ---------------
        if update_level
            ## A1
            XwX1 = aggXwX(D1,      data.j1, qs, dm.S1, nl, nk)
            XwY1 = aggXwX(data.w1, data.j1, qs, dm.S1, nl, nk)

            ## A2
            XwX2 = aggXwX(D1,      data.j2, qs, dm.S2, nl, nk)
            XwY2 = aggXwX(data.w2, data.j2, qs, dm.S2, nl, nk)

            if !isnothing(cs)
                # --- QP problem ----
                P = sparse(1:(nl*nk*2), 1:(nl*nk*2), vcat( XwX1[:],XwX2[:]) ) 
                q = -vcat( XwY1[:],XwY2[:])   
                res = QPConstrainedSolve(P,q,cs)                
                #res = -q ./ vcat( XwX1[:],XwX2[:])

                Ahat = reshape(res,nk,nl,2)
                dm.A1 .= Ahat[:,:,1]
                dm.A2 .= Ahat[:,:,2]
            else
                dm.A1 .= XwY1 ./ XwX1
                dm.A2 .= XwY2 ./ XwX2
            end

            # ## S1
            XwYv = aggXwX_var(data.w1, data.j1, dm.A1, qs, dm.S1, nl, nk)
            dm.S1 .= sqrt.( XwYv ./ XwX1 )

            ## S2
            XwYv = aggXwX_var(data.w2, data.j2, dm.A2, qs, dm.S2, nl, nk)
            dm.S2 .= sqrt.( XwYv ./ XwX2 )
        end

        # # proportions
        pk1 = zeros(nk,nk,nl)
        for ii in 1:size(data,1), l in 1:nl
            pk1[data.j1[ii],data.j2[ii],l] += qs[ii,l]
        end
        for k1 in 1:nk, k2 in 1:nk
            pk1[k1,k2,:] .= ( pk1[k1,k2,:] .+ dm.dprior .-1 ) ./ sum(pk1[k1,k2,:] .+ dm.dprior .-1)
        end
        dm.pk1 .= pk1
    end

    con = model_connectedness(dm)
    return(Dict(:lik => total_lik, :con => con))
end

function model_connectedness(dm::DistributionModelES)

    nl = dm.nl
    nk = dm.nk
    EV = zeros(nl)
    pk1 = dm.pk1
    
    for l in 1:nl
        A = pk1[:, :, l] .* dm.NNm / sum(dm.NNm)
        A ./= sum(A)
        A = 0.5 * A + 0.5 * A'
        D = Diagonal(sum(A, dims=2)[:].^(-0.5))
        L = Diagonal(ones(nk)) - D * A * D
        evals, evects = eigen(L)
        #@show evals
        EV[l] = sort(evals)[2]
    end

    return(minimum(abs.(EV)))
end

function distributional_em_all(data::DataFrame, nl, nk; nstart=10, tol=1e-10, iterprint=Inf, maxiter=200)

    all_reps = []
    for rep in 1:nstart
        dm1 = llmr.DistributionModelES(nl,nk);
        dm1.dprior = 1.00001
        mm = mean(data.w1)
        ms = 2 * std(data.w1)
        A0par = repeat(sort(randn(nl)).*ms .+ mm,1,nk)'
        dm1.A1 .= A0par
        dm1.A2 .= A0par
    
        llmr.distributional_em!(dm1, data, maxiter,tol; iterprint=iterprint, update_level=false,  msg="probs");
        emres = llmr.distributional_em!(dm1, data, maxiter, tol; iterprint=iterprint, constrained = "para",  msg="para");
        emres = llmr.distributional_em!(dm1, data, maxiter, tol; iterprint=iterprint, constrained = "fixb", msg="fixb")

        @printf("[%s][%03i/%i] lik=%+2.4e con=%2.2f \n",
                        Dates.format(now(), "HH:MM:SS"), rep, nstart,
                        emres[:lik],emres[:con])
        
        push!(all_reps,Dict( :model => dm1, :emres => emres))
    end

    return(all_reps)
end