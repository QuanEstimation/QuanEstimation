using SparseArrays
"""
Pauli matrces
"""
sigmax() = [.0im 1.;1. 0.]
sigmay() = [0. -1.0im;1.0im 0.]
sigmaz() = [1.0  .0im;0. -1.]
sigmap() = [.0im 1.;0. 0.]
sigmam() = [.0im 0.;1. 0.]
sigmax(i, N) = kron(I(2^(i - 1)), sigmax(), I(2^(N - i)))
sigmay(i, N) = kron(I(2^(i - 1)), sigmay(), I(2^(N - i)))
sigmaz(i, N) = kron(I(2^(i - 1)), sigmaz(), I(2^(N - i)))
sigmap(i, N) = kron(I(2^(i - 1)), sigmap(), I(2^(N - i)))
sigmam(i, N) = kron(I(2^(i - 1)), sigmam(), I(2^(N - i)))

function destroy(M)
    spdiagm(M, M, 1 => map(x -> x |> sqrt, 1:(M - 1)))
end

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
    [system for i in 1:N]
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

function Adam(gt, t, para, m_t, v_t, alpha=0.01, beta1=0.90, beta2=0.99, epsilon=1e-8)
    t = t+1
    m_t = beta1*m_t + (1-beta1)*gt
    v_t = beta2*v_t + (1-beta2)*(gt*gt)
    m_cap = m_t/(1-(beta1^t))
    v_cap = v_t/(1-(beta2^t))
    para = para+(alpha*m_cap)/(sqrt(v_cap)+epsilon)
    return para, m_t, v_t
end

function Adam!(system, δ, mt=0.0, vt=0.0)
    for ctrl in 1:length(δ)
        for ti in 1:length(system.times)
            system.control_coefficients[ctrl][ti], mt, vt = Adam(δ[ctrl][ti], ti, system.control_coefficients[ctrl][ti], mt, vt)
        end
    end
end

