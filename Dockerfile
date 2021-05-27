FROM python
RUN pip install numpy sympy matplotlib jill scipy \
    && sh -c '/bin/echo -e "Y" | jill install'\
    && pip install julia
RUN echo 'packs=["PyCall", "Zygote"];for i in packs;using Pkg;Pkg.add(i);end' >> packs.jl \
    && julia packs.jl 