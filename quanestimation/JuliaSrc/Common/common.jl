destroy(N) = diagm(1 => [1/sqrt(n) for n in 1:N-1])

bases(dim; T=ComplexF64) = [e for e in I(dim).|>T|>eachrow]

function vec2mat(x::Vector{T}) where {T <: Number}
    reshape(x, x |> length |> sqrt |> Int, :)  
end

function vec2mat(x)
    vec2mat.(x)
end

function vec2mat(x::Matrix)
    throw(ErrorException("vec2mating a matrix of size $(size(x))"))
end

function Base.repeat(system, N)
    [deepcopy(system) for i in 1:N]
end

function Base.repeat(system, M, N)
    reshape(repeat(system, M * N), M, N)
end

function filterZeros!(x::Matrix{T}) where {T <: Complex}
    x[abs.(x) .< eps()] .= zero(T)
    x
end
function filterZeros!(x) 
    filterZeros!.(x)
end

function filterZeros(x::AbstractVecOrMat{T}) where T<:Number
    [x+1≈1 ? zero(T) : x for x in x]
end

function t2Num(t0, dt, t)
    Int(round((t - t0) / dt)) + 1 
end

function basis(dim, si, ::T)::Array{T} where {T <: Complex}
    result = zeros(T, dim)
    result[si] = 1.0
    result
end

function suN_generatorU(n, k)
    tmp1, tmp2 = ceil((1+sqrt(1+8k))/2), ceil((-1+sqrt(1+8k))/2) 
    i = k - tmp2*(tmp2-1)/2 |> Int
    j =  tmp1 |> Int
    return sparse([i, j], [j,i], [1, 1], n, n)
end

function suN_generatorV(n, k)
    tmp1, tmp2 = ceil((1+sqrt(1+8k))/2), ceil((-1+sqrt(1+8k))/2) 
    i = k - tmp2*(tmp2-1)/2 |> Int
    j =  tmp1 |> Int 
    return sparse([i, j], [j,i], [-im, im], n, n)
end

function suN_generatorW(n, k)
    diagw = spzeros(n)
    diagw[1:k] .=1
    diagw[k+1] = -k
    return spdiagm(n,n,diagw)
end

function suN_generator(n)
    result = Vector{SparseMatrixCSC{ComplexF64, Int64}}(undef, n^2-1)
    idx = 2
    itr = 1

    for i in 1:n-1
       idx_t = idx
       while idx_t > 0
            result[itr] = iseven(idx_t) ? suN_generatorU(n, (i*(i-1)+idx-idx_t+2)/2) : suN_generatorV(n, (i*(i-1)+idx-idx_t+1)/2)
            itr += 1
            idx_t -= 1
       end
       result[itr] = sqrt(2/(i+i*i))*suN_generatorW(n, i)
       itr += 1
       idx += 2
    end
    return result
end

function get_basis(dim, index)
    x = zeros(dim)
    x[index] = 1.0
    return x
end

function sic_povm(fiducial)
    """
    Generate a set of POVMs by applying the d^2 Weyl-Heisenberg displacement operators to a
    fiducial state. 
    The Weyl-Heisenberg displacement operators are constructioned by Fuchs et al. in the article
    https://doi.org/10.3390/axioms6030021 and it is realized in QBism.
    
    """
    d = length(fiducial)
    w = exp(2.0*pi*1.0im/d)
    Z = diagm([w^(i-1) for i in 1:d])
    X = zeros(ComplexF64, d, d)
    for i in 1:d
        for j in 1:d
            if j != d
                X += get_basis(d, j+1)*get_basis(d,j)'
            else
                X += get_basis(d, 1)* get_basis(d,j)'
            end
        end
    end
    X = X/d
    
    D = [[Matrix{ComplexF64}(undef,d,d) for i in 1:d] for j in 1:d]
    for a in 1:d
        for b in 1:d
            X_a = X^(b-1)
            Z_b = Z^(a-1)
            D[a][b] = (-exp(1.0im*pi/d))^((a-1)*(b-1))*X_a*Z_b
        end
    end
            
    res = Vector{Matrix{ComplexF64}}()
    for m in 1:d
        for n in 1:d
            res_tp = D[m][n]*fiducial
            res_tp = res_tp/norm(res_tp)
            push!(res, res_tp*res_tp'/d)
        end
    end
    return res
end

function SIC(dim)
    data = readdlm("$(Main.pkgpath)/sic_fiducial_vectors/d$(dim).txt", '\t', Float64, '\n')
    fiducial = data[:,1]+1.0im*data[:,2]
    M = sic_povm(fiducial)
end

function AdaptiveInput(x, func, dfunc; channel="dynamics")
    para_num = length(x)
    x_size = [x[i] for i in 1:para_num]
    x_list  =  Iterators.product(x...)
    if channel == "dynamics"
        H = [func(xi) for xi in x_list]
        dH = [dfunc(xi) for xi in x_list]
        return H, dH
    elseif channel == "kraus"
        K = [func(xi) for xi in x_list]
        dK = [dfunc(xi) for xi in x_list]
        return K, dK
    else
        throw("Supported values for channel are 'dynamics' and 'kraus'")
    end
end
