doc:
	rm -rf docs/build
	julia -e 'using Pkg; Pkg.activate("."); include("docs/make.jl")'
	cd docs/build && \
	git init && \
	git add . && \
	git commit -m "Initial commit" && \
	git remote add origin git@github.com:tlamadon/blm.jl.git && \
	git push --force origin master:gh-pages