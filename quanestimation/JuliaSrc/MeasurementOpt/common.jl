mutable struct projection_Mopt{T<:Complex, M <:Real} <:ControlSystem
    freeHamiltonian
    Hamiltonian_derivative::Vector{Matrix{T}}
    ρ0::Matrix{T}
    tspan::Vector{M}
    decay_opt::Vector{Matrix{T}}
    γ::Vector{M}
    C::Vector{Vector{T}}
    W::Matrix{M}
    eps::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    projection_Mopt(freeHamiltonian, Hamiltonian_derivative::Vector{Matrix{T}}, ρ0::Matrix{T}, tspan::Vector{M}, 
    decay_opt::Vector{Matrix{T}},γ::Vector{M}, C::Vector{Vector{T}}, W::Matrix{M}, eps::M, 
    ρ=Vector{Matrix{T}}(undef, 1), ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T<:Complex, M<:Real}=
    new{T,M}(freeHamiltonian, Hamiltonian_derivative, ρ0, tspan, decay_opt, γ, C, W, eps, ρ, ∂ρ_∂x) 
end

mutable struct LinearComb_Mopt{T<:Complex, M <:Real} <:ControlSystem
    freeHamiltonian
    Hamiltonian_derivative::Vector{Matrix{T}}
    ρ0::Matrix{T}
    tspan::Vector{M}
    decay_opt::Vector{Matrix{T}}
    γ::Vector{M}
    povm_basis::Vector{Matrix{T}}
    M_num::Int64
    W::Matrix{M}
    eps::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    LinearComb_Mopt(freeHamiltonian, Hamiltonian_derivative::Vector{Matrix{T}}, ρ0::Matrix{T}, tspan::Vector{M}, 
    decay_opt::Vector{Matrix{T}},γ::Vector{M}, povm_basis::Vector{Matrix{T}}, M_num::Int64, W::Matrix{M}, eps::M, 
    ρ=Vector{Matrix{T}}(undef, 1), ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T<:Complex, M<:Real}=
    new{T,M}(freeHamiltonian, Hamiltonian_derivative, ρ0, tspan, decay_opt, γ, povm_basis, M_num, W, eps, ρ, ∂ρ_∂x) 
end

mutable struct RotateBasis_Mopt{T<:Complex, M <:Real} <:ControlSystem
    freeHamiltonian
    Hamiltonian_derivative::Vector{Matrix{T}}
    ρ0::Matrix{T}
    tspan::Vector{M}
    decay_opt::Vector{Matrix{T}}
    γ::Vector{M}
    povm_basis::Vector{Matrix{T}}
    W::Matrix{M}
    eps::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    RotateBasis_Mopt(freeHamiltonian, Hamiltonian_derivative::Vector{Matrix{T}}, ρ0::Matrix{T}, tspan::Vector{M}, 
    decay_opt::Vector{Matrix{T}},γ::Vector{M}, povm_basis::Vector{Matrix{T}}, W::Matrix{M}, eps::M, 
    ρ=Vector{Matrix{T}}(undef, 1), ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T<:Complex, M<:Real}=
    new{T,M}(freeHamiltonian, Hamiltonian_derivative, ρ0, tspan, decay_opt, γ, povm_basis, W, eps, ρ, ∂ρ_∂x) 
end

function bound_LC_coeff(coefficients::Vector{Vector{Float64}})
    M_num = length(coefficients)
    basis_num = length(coefficients[1])
    for ck in 1:M_num
        for tk in 1:basis_num
            coefficients[ck][tk] = (x-> x < 0.0 ? 0.0 : x > 1.0 ? 1.0 : x)(coefficients[ck][tk])
        end 
    end

    Sum_col = [sum([coefficients[m][n] for m in 1:M_num])  for n in 1:basis_num]
    for si in 1:basis_num
        if Sum_col[si] == 0.0
            int_num = sample(1:M_num, 1, replace=false)[1]
            coefficients[int_num][si] = 1.0
        end
    end

    Sum_row = [sum([coefficients[m][n] for n in 1:basis_num])  for m in 1:M_num]
    for mi in 1:M_num
        if Sum_row[mi] == 0.0
            int_num = sample(1:basis_num, 1, replace=false)[1]
            coefficients[mi][int_num] = rand()
        end
    end 

    Sum_col = [sum([coefficients[m][n] for m in 1:M_num])  for n in 1:basis_num]
    for i in 1:M_num
        for j in 1:basis_num
            coefficients[i][j] = coefficients[i][j]/Sum_col[j]
        end
    end
    coefficients
end

function bound_rot_coeff(coefficients::Vector{Float64})
    n = length(coefficients)
    for tk in 1:n
        coefficients[tk] = (x-> x < 0.0 ? 0.0 : x > 2*pi ? 2*pi : x)(coefficients[tk])
    end 
    coefficients
end

function MOpt_Adam(gt, t, para, m_t, v_t, ϵ, beta1, beta2, eps)
    t = t+1
    m_t = beta1*m_t + (1-beta1)*gt
    v_t = beta2*v_t + (1-beta2)*(gt*gt)
    m_cap = m_t/(1-(beta1^t))
    v_cap = v_t/(1-(beta2^t))
    para = para+(ϵ*m_cap)/(sqrt(v_cap)+eps)
    return para, m_t, v_t
end

function MOpt_Adam!(coefficients::Vector{Float64}, δ, ϵ, mt, vt, beta1, beta2, eps)
    for ci in 1:length(δ)
        coefficients[ci], mt, vt = MOpt_Adam(δ[ci], ci, coefficients[ci], mt, vt, ϵ, beta1, beta2, eps)
    end
    coefficients
end

function MOpt_Adam!(coefficients::Vector{Vector{Float64}}, δ, ϵ, mt, vt, beta1, beta2, eps)
    mt_in = mt
    vt_in = vt
    for ci in 1:length(δ)
        mt = mt_in
        vt = vt_in
        for cj in 1:length(δ[1])
            coefficients[ci][cj], mt, vt = MOpt_Adam(δ[ci][cj], cj, coefficients[ci][cj], mt, vt, ϵ, beta1, beta2, eps)
        end
    end
    coefficients
end

function MOpt_Adam!(system, δ, ϵ, mt, vt, beta1, beta2, eps)
    for cj in 1:length(δ[1])
        system.M[1][cj], mt, vt = MOpt_Adam(δ[1][cj], cj, system.M[1][cj], mt, vt, ϵ, beta1, beta2, eps)
    end
end

function SaveFile_meas(f_now::Float64, M)
    open("f.csv","a") do f
        writedlm(f, [f_now])
    end
    open("measurements.csv","a") do g
        writedlm(g, M)
    end
end

function SaveFile_meas(f_now::Vector{Float64}, M)
    open("f.csv","w") do f
        writedlm(f, f_now)
    end
    open("measurements.csv","w") do g
        writedlm(g, M)
    end
end

function gramschmidt(A::Vector{Vector{ComplexF64}})
    n = length(A[1])
    m = length(A)
    Q = [zeros(ComplexF64,n) for i in 1:m]
    for j in 1:m
        q = A[j]
        for i in 1:j
            rij = dot(Q[i], q)
            q = q - rij*Q[i]
        end
        Q[j] = q/norm(q)
    end
    Q
end

function rotation_matrix(coefficients, Lambda)
    dim = size(Lambda[1])[1]
    U = Matrix{ComplexF64}(I,dim,dim)
    for i in 1:length(Lambda)
        U = U*exp(1.0im*coefficients[i]*Lambda[i])
    end
    return U
end
