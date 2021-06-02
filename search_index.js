var documenterSearchIndex = {"docs":
[{"location":"functions-clustering/","page":"Clustering","title":"Clustering","text":"# To build documentation:\n# julia --color=yes --project make.jl","category":"page"},{"location":"functions-clustering/#Clustering-functions","page":"Clustering","title":"Clustering functions","text":"","category":"section"},{"location":"functions-clustering/#Functions","page":"Clustering","title":"Functions","text":"","category":"section"},{"location":"functions-clustering/","page":"Clustering","title":"Clustering","text":"Modules = [blm]\nPages = [\"clustering.jl\"]","category":"page"},{"location":"functions-clustering/#blm.group_firms-Tuple{Any, Any}","page":"Clustering","title":"blm.group_firms","text":"group_firms(y,fids; nq=20,ng=10,nstart=100)\n\ngroups firms based on empirical cdf pass a vector of firm ids and a vector of wages, at the individual level  nq is number of points for the empirical cdf ng is the number of groups nstart is the number of kmeans to run maxiter for each kmeans is set at 1000\n\n\n\n\n\n","category":"method"},{"location":"functions-clustering/#Contents","page":"Clustering","title":"Contents","text":"","category":"section"},{"location":"functions-clustering/","page":"Clustering","title":"Clustering","text":"","category":"page"},{"location":"functions-clustering/#Index","page":"Clustering","title":"Index","text":"","category":"section"},{"location":"functions-clustering/","page":"Clustering","title":"Clustering","text":"","category":"page"},{"location":"functions-constraints/","page":"Constraints","title":"Constraints","text":"# To build documentation:\n# julia --color=yes --project make.jl","category":"page"},{"location":"functions-constraints/#Constraints-functions","page":"Constraints","title":"Constraints functions","text":"","category":"section"},{"location":"functions-constraints/#Functions","page":"Constraints","title":"Functions","text":"","category":"section"},{"location":"functions-constraints/","page":"Constraints","title":"Constraints","text":"Modules = [blm]\nPages = [\"constraints.jl\"]","category":"page"},{"location":"functions-constraints/#blm.AddConstraint-Tuple{}","page":"Constraints","title":"blm.AddConstraint","text":"AddConstraint(; qp::QPConstrained, constraint::Function, kwargs...)\n\nAdd constraint to a QPConstrained object.\n\nArguments:     qp (QPConstrained): instance of QPConstrained     constraint (Function): which constraint to use     kwargs (kwargs): keyword arguments for the particular constraint (skip nk and nl)\n\n\n\n\n\n","category":"method"},{"location":"functions-constraints/#blm.ConstraintAKMMono-Tuple{}","page":"Constraints","title":"blm.ConstraintAKMMono","text":"ConstraintAKMMono(; nk::Int64, nl::Int64, gap::Union{Float64, Int64}=0)::QPConstrained\n\nAKM mono constraint.\n\nArguments:     nk (int): number of firm types     nl (int): number of worker types     gap (int): FIXME\n\n\n\n\n\n","category":"method"},{"location":"functions-constraints/#blm.ConstraintBiggerThan-Tuple{}","page":"Constraints","title":"blm.ConstraintBiggerThan","text":"ConstraintBiggerThan(; nk::Int64, nl::Int64, gap::Union{Float64, Int64}=0, n_periods::Int64=2)::QPConstrained\n\nBigger than constraint.\n\nArguments:     nk (int): number of firm types     nl (int): number of worker types     gap (int): lower bound     n_periods (int): number of periods in event study\n\n\n\n\n\n","category":"method"},{"location":"functions-constraints/#blm.ConstraintFixB-Tuple{}","page":"Constraints","title":"blm.ConstraintFixB","text":"ConstraintFixB(; nk::Int64, nl::Int64, nt::Int64=4)::QPConstrained\n\nFix B constraint.\n\nArguments:     nk (int): number of firm types     nl (int): number of worker types     nt (int): FIXME\n\n\n\n\n\n","category":"method"},{"location":"functions-constraints/#blm.ConstraintLinear-Tuple{}","page":"Constraints","title":"blm.ConstraintLinear","text":"ConstraintLinear(; nk::Int64, nl::Int64, nt::Int64=2)::QPConstrained\n\nLinear constraint. for a set of coeficient nk x nl x nt this makes sure that\n\na_k1_l1_t - a_k2_l1_t = a_k1_l1_t - a_k2_l1_t\nfor all l1, t and 2 firms k1, k2\n\nArguments:     nk (int): number of firm types     nl (int): number of worker types     n_periods (int): number of periods in event study\n\n\n\n\n\n","category":"method"},{"location":"functions-constraints/#blm.ConstraintMonoK-Tuple{}","page":"Constraints","title":"blm.ConstraintMonoK","text":"ConstraintMonoK(; nk::Int64, nl::Int64, gap::Union{Float64, Int64}=0)::QPConstrained\n\nMono K constraint.\n\nArguments:     nk (int): number of firm types     nl (int): number of worker types     gap (int): FIXME\n\n\n\n\n\n","category":"method"},{"location":"functions-constraints/#blm.ConstraintNone-Tuple{}","page":"Constraints","title":"blm.ConstraintNone","text":"ConstraintNone(; nk::Int64, nl::Int64)::QPConstrained\n\nNo constraint.\n\nArguments:     nk (int): number of firm types     nl (int): number of worker types\n\n\n\n\n\n","category":"method"},{"location":"functions-constraints/#blm.ConstraintPara-Tuple{}","page":"Constraints","title":"blm.ConstraintPara","text":"ConstraintPara(; nk::Int64, nl::Int64, nt::Int64=2)::QPConstrained\n\nParallel constraint, worker get same wage everywhere. For a set of coeficient nk x nl x nt this makes sure that\n\na_k1_l_t = a_k2_l_t\nfor all l, t and 2 firms k1, k2\n\nArguments:     nk (int): number of firm types     nl (int): number of worker types     n_periods (int): number of periods in event study\n\n\n\n\n\n","category":"method"},{"location":"functions-constraints/#blm.ConstraintSmallerThan-Tuple{}","page":"Constraints","title":"blm.ConstraintSmallerThan","text":"ConstraintSmallerThan(; nk::Int64, nl::Int64, gap::Union{Float64, Int64}=0, n_periods::Int64=2)::QPConstrained\n\nBigger than constraint.\n\nArguments:     nk (int): number of firm types     nl (int): number of worker types     gap (int): upper bound     n_periods (int): number of periods in event study\n\n\n\n\n\n","category":"method"},{"location":"functions-constraints/#blm.ConstraintStationary-Tuple{}","page":"Constraints","title":"blm.ConstraintStationary","text":"ConstraintStationary(; nk::Int64, nl::Int64)::QPConstrained\n\nStationary constraint.\n\nArguments:     nk (int): number of firm types     nl (int): number of worker types\n\n\n\n\n\n","category":"method"},{"location":"functions-constraints/#blm.ConstraintSum-Tuple{}","page":"Constraints","title":"blm.ConstraintSum","text":"ConstraintSum(; nk::Int64, nl::Int64)::QPConstrained\n\nSum constraint.\n\nArguments:     nk (int): number of firm types     nl (int): number of worker types\n\n\n\n\n\n","category":"method"},{"location":"functions-constraints/#blm.QPConstrainedSolve-Tuple{SparseArrays.SparseMatrixCSC{Float64, Int64}, Vector{Float64}, blm.QPConstrained}","page":"Constraints","title":"blm.QPConstrainedSolve","text":"QPConstrainedSolve(P::SparseMatrixCSC{Float64, Int64}, q::Vector{Float64}, qp::QPConstrained)\n\nSolve constrained QP problem, min_x 0.5x'Px + q'x s.t. l <= Ax <= u. Arguments:     P (Matrix): P matrix for QP problem     q (Vector): q vector for QP problem     constraint (Function): which constraint to use     kwargs (kwargs): keyword arguments for the particular constraint\n\n\n\n\n\n","category":"method"},{"location":"functions-constraints/#Contents","page":"Constraints","title":"Contents","text":"","category":"section"},{"location":"functions-constraints/","page":"Constraints","title":"Constraints","text":"","category":"page"},{"location":"functions-constraints/#Index","page":"Constraints","title":"Index","text":"","category":"section"},{"location":"functions-constraints/","page":"Constraints","title":"Constraints","text":"","category":"page"},{"location":"functions-model/","page":"Model","title":"Model","text":"# To build documentation:\n# julia --color=yes --project make.jl","category":"page"},{"location":"functions-model/#Model-functions","page":"Model","title":"Model functions","text":"","category":"section"},{"location":"functions-model/#Functions","page":"Model","title":"Functions","text":"","category":"section"},{"location":"functions-model/","page":"Model","title":"Model","text":"Modules = [blm]\nPages = [\"model.jl\"]","category":"page"},{"location":"functions-model/#blm.DistributionModel","page":"Model","title":"blm.DistributionModel","text":"DistributionModel\n\nEvent study BLM model for movers     - static version     - dynamic version\n\n\n\n\n\n","category":"type"},{"location":"functions-model/#blm.distributional_em!","page":"Model","title":"blm.distributional_em!","text":"distributional_em!(dm::DistributionModel, data::DataFrame, maxiter::Int, tol::Float64 = 1e-8; update_level = true, constrained=nothing, iterprint=50, msg=\"\")\n\nRun the em algorithm on the passed DistributionalModel using the given data\n\n\n\n\n\n","category":"function"},{"location":"functions-model/#blm.simulate-Tuple{blm.DistributionModel, Any}","page":"Model","title":"blm.simulate","text":"simulate(model::DistributionModel, nn)\n\nUsing the model, simulates a dataset. The timing is important, the event happens at the end of the period, hence if a current period is u2e there will be no wage in the current period, as well as no firm.\n\n\n\n\n\n","category":"method"},{"location":"functions-model/#Contents","page":"Model","title":"Contents","text":"","category":"section"},{"location":"functions-model/","page":"Model","title":"Model","text":"","category":"page"},{"location":"functions-model/#Index","page":"Model","title":"Index","text":"","category":"section"},{"location":"functions-model/","page":"Model","title":"Model","text":"","category":"page"},{"location":"","page":"blm.jl","title":"blm.jl","text":"# To build documentation:\n# julia --color=yes --project make.jl","category":"page"},{"location":"#blm.jl","page":"blm.jl","title":"blm.jl","text":"","category":"section"},{"location":"","page":"blm.jl","title":"blm.jl","text":"","category":"page"},{"location":"","page":"blm.jl","title":"blm.jl","text":"blm.jl is a Julia package that implements the BLM estimator.","category":"page"},{"location":"#Index","page":"blm.jl","title":"Index","text":"","category":"section"},{"location":"","page":"blm.jl","title":"blm.jl","text":"","category":"page"},{"location":"#Functions","page":"blm.jl","title":"Functions","text":"","category":"section"},{"location":"","page":"blm.jl","title":"blm.jl","text":"","category":"page"}]
}