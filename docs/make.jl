push!(LOAD_PATH,"../src/") # To add package to filepath
using Documenter, blm

makedocs(
    sitename="BLM Documentation",
    pages = [
        "Model" => "functions-model.md",
        "Clustering" => "functions-clustering.md",
        "Constraints" => "functions-constraints.md"
    ]
)
