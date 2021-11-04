
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