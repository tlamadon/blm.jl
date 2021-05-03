"""
Constraints for BLM
"""

using Parameters
using LinearAlgebra
using SparseArrays
using OSQP
using Suppressor

##################
##### Struct #####
##################

# Constraints structure with keyword arguments (that can have default values)
# Source: https://discourse.julialang.org/t/can-a-struct-be-created-with-field-keywords/27531
# Source: https://discourse.julialang.org/t/initialization-of-mutable-structs-with-many-default-values/17959
@with_kw mutable struct QPConstrained
    # Gx <= h
    # Ax = b
    nk::Int # Number of firm types
    nl::Int # Number of worker types

    G::Union{Matrix{Float64}, Nothing} = nothing
    h::Union{Vector{Float64}, Nothing} = nothing
    A::Union{Matrix{Float64}, Nothing} = nothing
    b::Union{Vector{Float64}, Nothing} = nothing
end

function AddConstraint(; qp::QPConstrained, constraint::Function, kwargs...)
    """
    Add constraint to a QPConstrained object.

    Arguments:
        qp (QPConstrained): instance of QPConstrained
        constraint (Function): which constraint to use
        kwargs (kwargs): keyword arguments for the particular constraint (skip nk and nl)
    """
    # Create constraint
    constraint_qp = constraint(; nk=qp.nk, nl=qp.nl, kwargs...)
    if !isnothing(constraint_qp.G) # If you have inequality constraints
        if !isnothing(qp.G)
            qp.G = vcat(qp.G, constraint_qp.G)
            qp.h = vcat(qp.h, constraint_qp.h)
        else
            qp.G = constraint_qp.G
            qp.h = constraint_qp.h
        end
    end
    if !isnothing(constraint_qp.A) # If you have equality constraints
        if !isnothing(qp.A)
            qp.A = vcat(qp.A, constraint_qp.A)
            qp.b = vcat(qp.b, constraint_qp.b)
        else
            qp.A = constraint_qp.A
            qp.b = constraint_qp.b
        end
    end
end

##################
##### Solver #####
##################

function QPConstrainedSolve(P::SparseMatrixCSC{Float64, Int64}, q::Vector{Float64}, qp::QPConstrained)
    """
    Solve constrained QP problem, min_x 0.5x'Px + q'x s.t. l <= Ax <= u.
    Arguments:
        P (Matrix): P matrix for QP problem
        q (Vector): q vector for QP problem
        constraint (Function): which constraint to use
        kwargs (kwargs): keyword arguments for the particular constraint
    """
    #####
    # Our constraints solve
    # Gx <= h
    # Ax = b
    # OSQP solves
    # l <= Ax <= u
    #####
    # Create OSQP object
    prob = OSQP.Model()

    # Setup workspace and change alpha parameter
    if !isnothing(qp.G) && !isnothing(qp.A) # Use both constraints
        # Set lower bound to NaN for G
        OSQP.setup!(prob; P=sparse(P), q=q, A=sparse(vcat(qp.G, qp.A)), l=vcat(fill(-Inf, size(qp.h, 1)), qp.b), u=vcat(qp.h, qp.b))

    elseif !isnothing(qp.G) # Use G constraint
        OSQP.setup!(prob; P=sparse(P), q=q, A=sparse(qp.G), u=qp.h)

    elseif !isnothing(qp.A) # Use A constraint
        OSQP.setup!(prob; P=sparse(P), q=q, A=sparse(qp.A), l=qp.b, u=qp.b)

    else # Use no constraints
        OSQP.setup!(prob; P=sparse(P), q=q)
    end
    # Solve problem
    @suppress return OSQP.solve!(prob).x; # .x gives result
end

#######################
##### Constraints #####
#######################

function ConstraintLinear(;nk::Int64, nl::Int64, nt::Int64=2)::QPConstrained
    """
    Linear constraint.
    for a set of coeficient nk x nl x nt this makes sure that 

        a_k1_l1_t - a_k2_l1_t = a_k1_l1_t - a_k2_l1_t        
        for all l1, t and 2 firms k1, k2

    Arguments:
        nk (int): number of firm types
        nl (int): number of worker types
        n_periods (int): number of periods in event study
    """
    A = []
    for t in 1:nt, l in 2:nl, k in 2:nk
        A0 = zeros(nk,nl,nt)
        A0[k-1, l,   t] = 1
        A0[k,   l,   t] = -1
        A0[k-1, l-1, t] = -1
        A0[k,   l-1, t] = 1
        push!(A,A0[:]')
    end
    A = vcat(A...)
    b = - zeros(Float64, size(A,1))
    return QPConstrained(nk=nk, nl=nl, G=nothing, h=nothing, A=A, b=b);
end

function ConstraintPara(;nk::Int64, nl::Int64, nt::Int64=2)::QPConstrained
    """
    Parallel constraint, wokrer get same wage everywhere.
    for a set of coeficient nk x nl x nt this makes sure that 

        a_k1_l_t = a_k2_l_t 
        for all l, t and 2 firms k1, k2

    Arguments:
        nk (int): number of firm types
        nl (int): number of worker types
        n_periods (int): number of periods in event study
    """
    A = []
    for t in 1:nt, l in 1:nl, k in 2:nk
        A0 = zeros(nk,nl,nt)
        A0[k-1, l,   t] = 1
        A0[k,   l,   t] = -1
        push!(A,A0[:]')
    end
    A = vcat(A...)
    b = - zeros(Float64, size(A,1))
    return QPConstrained(nk=nk, nl=nl, G=nothing, h=nothing, A=A, b=b);
end

function ConstraintAKMMono(; nk::Int64, nl::Int64, gap::Union{Float64, Int64}=0)::QPConstrained
    """
    AKM mono constraint.

    Arguments:
        nk (int): number of firm types
        nl (int): number of worker types
        gap (int): FIXME
    """
    LL = zeros(Int8, (nl - 1, nl))
    for l = 1:nl - 1
        LL[l, l] = 1
        LL[l, l + 1] = - 1
    end
    KK = zeros(Int8, (nk - 1, nk))
    for k = 1:nk - 1
        KK[k, k] = 1
        KK[k, k + 1] = - 1
    end
    G = kron(I(nl), KK)
    h = - gap * ones(Int8, nl * (nk - 1))

    A = - kron(LL, KK)
    b = - zeros(Int8, size(A)[1])

    return QPConstrained(nk=nk, nl=nl, G=G, h=h, A=A, b=b)
end

function ConstraintMonoK(; nk::Int64, nl::Int64, gap::Union{Float64, Int64}=0)::QPConstrained
    """
    Mono K constraint.

    Arguments:
        nk (int): number of firm types
        nl (int): number of worker types
        gap (int): FIXME
    """
    KK = zeros(Int8, (nk - 1, nk))
    for k = 1:nk - 1
        KK[k, k] = 1
        KK[k, k + 1] = - 1
    end
    G = kron(I(nl), KK)
    h = - gap * ones(Int8, nl * (nk - 1))

    return QPConstrained(nk=nk, nl=nl, G=G, h=h, A=nothing, b=nothing)
end

function ConstraintFixB(;nk::Int64, nl::Int64, nt::Int64=4)::QPConstrained
    """
    Fix B constraint.

    Arguments:
        nk (int): number of firm types
        nl (int): number of worker types
        nt (int): FIXME
    """
    KK = zeros(Int8, (nk - 1, nk))
    for k = 1:nk - 1
        KK[k, k] = 1
        KK[k, k + 1] = - 1
    end
    A = - kron(I(nl), KK)
    MM = zeros(Int8, (nt - 1, nt))
    for m = 1:nt - 1
        MM[m, m] = 1
        MM[m, m + 1] = - 1
    end

    A = - kron(MM, A)
    b = - zeros(Int8, nl * (nk - 1) * (nt - 1))

    return QPConstrained(nk=nk, nl=nl, G=nothing, h=nothing, A=A, b=b)
end

function ConstraintBiggerThan(; nk::Int64, nl::Int64, gap::Union{Float64, Int64}=0, n_periods::Int64=2)::QPConstrained
    """
    Bigger than constraint.

    Arguments:
        nk (int): number of firm types
        nl (int): number of worker types
        gap (int): lower bound
        n_periods (int): number of periods in event study
    """
    G = - I(n_periods * nk * nl)
    h = - gap * ones(Int8, n_periods * nk * nl)

    return QPConstrained(nk=nk, nl=nl, G=G, h=h, A=nothing, b=nothing)
end

function ConstraintSmallerThan(; nk::Int64, nl::Int64, gap::Union{Float64, Int64}=0, n_periods::Int64=2)::QPConstrained
    """
    Bigger than constraint.

    Arguments:
        nk (int): number of firm types
        nl (int): number of worker types
        gap (int): upper bound
        n_periods (int): number of periods in event study
    """
    G = I(n_periods * nk * nl)
    h = gap * ones(Int8, n_periods * nk * nl)

    return QPConstrained(nk=nk, nl=nl, G=G, h=h, A=nothing, b=nothing)
end

function ConstraintStationary(; nk::Int64, nl::Int64)::QPConstrained
    """
    Stationary constraint.

    Arguments:
        nk (int): number of firm types
        nl (int): number of worker types
    """
    LL = zeros(Int8, (nl - 1, nl))
    for l = 1:nl - 1
        LL[l, l] = 1
        LL[l, l + 1] = - 1
    end
    A = kron(LL, I(nk))
    b = - zeros(Int8, (nl - 1) * nk)

    return QPConstrained(nk=nk, nl=nl, G=nothing, h=nothing, A=A, b=b)
end

function ConstraintNone(; nk::Int64, nl::Int64)::QPConstrained
    """
    No constraint.

    Arguments:
        nk (int): number of firm types
        nl (int): number of worker types
    """
    G = - zeros(Int8, (1, nk * nl))
    h = - zeros(Int8, 1)

    return QPConstrained(nk=nk, nl=nl, G=G, h=h, A=nothing, b=nothing)
end

function ConstraintSum(; nk::Int64, nl::Int64)::QPConstrained
    """
    Sum constraint.

    Arguments:
        nk (int): number of firm types
        nl (int): number of worker types
    """
    A = - kron(I(nl), transpose(ones(Int8, nk)))
    b = - zeros(Int8, nl)

    return QPConstrained(nk=nk, nl=nl, G=nothing, h=nothing, A=A, b=b)
end
