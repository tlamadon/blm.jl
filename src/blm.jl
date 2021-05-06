module blm

    export kmeansw

    # allow convenient multiple definitions in one line
    macro multidef(ex)
        @assert(ex isa Expr)
        @assert(ex.head == :(=))
        vars = ex.args[1].args
        what = ex.args[2]
        rex = quote end
        for var in vars
            push!(rex.args, :( $(esc(var)) = $(esc(what)) ))
        end
        rex
    end

    include("modules/clustering.jl")
    include("modules/constraints.jl")
    include("modules/model.jl")
end