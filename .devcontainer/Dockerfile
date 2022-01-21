# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.205.2/containers/ubuntu/.devcontainer/base.Dockerfile

# [Choice] Ubuntu version (use hirsuite or bionic on local arm64/Apple Silicon): hirsute, focal, bionic
ARG VARIANT="hirsute"
FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends python pip  
RUN pip install numpy sympy matplotlib jill scipy qutip\
    && sh -c '/bin/echo -e "Y" | jill install'\
    && pip install julia
# RUN echo 'packs=["PyCall", "Zygote", "DifferentialEquations", "JLD","DelimitedFiles", "StatsBase", "ReinforcementLearning", "Convex", "SCS", "StableRNGs"];for i in packs;using Pkg;Pkg.add(i);end' >> packs.jl \
    # && julia packs.jl
RUN echo 'using Pkg;Pkg.instantiate()' >>  /home/vscode/instantiate.jl
RUN julia --project=/workspaces/QuanEstimation/quanestimation/JuliaSrc  /home/vscode/instantiate.jl 

