using Statistics
using Clustering
using Printf
using Dates

function kmeansw()
    C = zeros(ng,nq)
    G = rand(1:ng,nf)
    D = zeros(nf,ng)
    weight = fsize/sum(fsize)
    bestval = Inf
    
    for iter in 1:100

        # update centroid
        for g in axes(C,1)        
            C[g,:] = sum( weight[ G.==g] .* fcdf[ G.==g ,:] ,dims = 1)
        end

        D .= 0 
        # compute distances
        for g in 1:ng, q in 1:nq        
            D[:,g] += (fcdf[:,q] .- C[g,q] ).^2
        end

        mxval, mxindx = findmin(D, ; dims=2)
        G .= [v[2] for v in mxindx][:]    
        newval = sum(mxval)
        
        if newval < bestval
            bestval = newval
        else
            break
        end
                
    end
    return(G,bestval)
end

"""
`group_firms(y,fids; nq=20,ng=10,nstart=100)`

    groups firms based on empirical cdf
    pass a vector of firm ids and a vector of wages, at the individual level 
    nq is number of points for the empirical cdf
    ng is the number of groups
    nstart is the number of kmeans to run
    maxiter for each kmeans is set at 1000
"""
function group_firms(y,fids; nq=20,ng=10,nstart=100)

    # we first get the points of supports in the data
    qs = range(1/(nq+1),1-1/(nq+1),length=nq)
    ys = quantile(y,  qs)

    # index the firms
    firm_ids = unique(fids)
    firm_new_ids = Dict( firm_ids[i] => i for i in 1:length(firm_ids) );
    fids_int = [firm_new_ids[f] for f in fids]
    nf = length(firm_ids)
    
    # compute the cdf and sizes
    fcdf  = zeros(nf,nq)
    fsize = zeros(nf)
    
    for q in 1:nq
        for i in 1:length(y)    
            if q==1
                fsize[fids_int[i]] += 1
            end
            if (y[i] < ys[q])
                fcdf[fids_int[i], q] += 1
            end
        end
    end
    
    for f in 1:nf
        fcdf[f,:] = fcdf[f,:]/ fsize[f]
    end

    bestg = 0
    bestval = Inf
    for istart in 1:nstart
        res = kmeans(fcdf', ng, weights = fsize, maxiter = 1000);
        if res.totalcost < bestval
            @printf("[kmean][%s][%i/%i] found better value old=%2.2e new=%2.2e Δ=%2.2e\n",
                        Dates.format(now(), "HH:MM:SS"), istart,nstart, bestval, res.totalcost, bestval - res.totalcost)
            bestval = res.totalcost
            bestg = res.assignments
        end
    end

    firm_grp = Dict( firm_ids[i] => bestg[i] for i in 1:length(firm_ids) )
    return(firm_grp, bestval, fcdf)
end
