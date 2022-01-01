mutable struct projection_Mopt{T<:Complex, M <:Real} <:ControlSystem
    freeHamiltonian
    Hamiltonian_derivative::Vector{Matrix{T}}
    ρ0::Matrix{T}
    tspan::Vector{M}
    decay_opt::Vector{Matrix{T}}
    γ::Vector{M}
    Measurement::Vector{Vector{T}}
    W::Matrix{M}
    accuracy::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    projection_Mopt(freeHamiltonian, Hamiltonian_derivative::Vector{Matrix{T}}, ρ0::Matrix{T}, tspan::Vector{M}, 
    decay_opt::Vector{Matrix{T}},γ::Vector{M}, Measurement::Vector{Vector{T}}, W::Matrix{M}, accuracy::M, 
    ρ=Vector{Matrix{T}}(undef, 1), ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T<:Complex, M<:Real}=
    new{T,M}(freeHamiltonian, Hamiltonian_derivative, ρ0, tspan, decay_opt, γ, Measurement, W, accuracy, ρ, ∂ρ_∂x) 
end

mutable struct givenpovm_Mopt{T<:Complex, M <:Real} <:ControlSystem
    freeHamiltonian
    Hamiltonian_derivative::Vector{Matrix{T}}
    ρ0::Matrix{T}
    tspan::Vector{M}
    decay_opt::Vector{Matrix{T}}
    γ::Vector{M}
    povm_basis::Vector{Matrix{T}}
    M_num::Int64
    W::Matrix{M}
    accuracy::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    givenpovm_Mopt(freeHamiltonian, Hamiltonian_derivative::Vector{Matrix{T}}, ρ0::Matrix{T}, tspan::Vector{M}, 
    decay_opt::Vector{Matrix{T}},γ::Vector{M}, povm_basis::Vector{Matrix{T}}, M_num::Int64, W::Matrix{M}, accuracy::M, 
    ρ=Vector{Matrix{T}}(undef, 1), ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T<:Complex, M<:Real}=
    new{T,M}(freeHamiltonian, Hamiltonian_derivative, ρ0, tspan, decay_opt, γ, povm_basis, M_num, W, accuracy, ρ, ∂ρ_∂x) 
end

function bound!(coefficients::Vector{Vector{Float64}})
    M_num = length(coefficients)
    n = length(coefficients[1])
    for ck in 1:M_num
        for tk in 1:n
            coefficients[ck][tk] = (x-> x < 0.0 ? 0.001 : x > 1.0 ? 1.0 : x)(coefficients[ck][tk])
        end 
    end
end
function bound!(coefficients::Vector{Float64})
    n = length(coefficients)
    for tk in 1:n
        coefficients[tk] = (x-> x < 0.0 ? 0.001 : x > 1.0 ? 1.0 : x)(coefficients[tk])
    end 
end

function MOpt_Adam(gt, t, para, m_t, v_t, ϵ, beta1, beta2, accuracy)
    t = t+1
    m_t = beta1*m_t + (1-beta1)*gt
    v_t = beta2*v_t + (1-beta2)*(gt*gt)
    m_cap = m_t/(1-(beta1^t))
    v_cap = v_t/(1-(beta2^t))
    para = para+(ϵ*m_cap)/(sqrt(v_cap)+accuracy)
    return para, m_t, v_t
end

function MOpt_Adam!(Mcoeff::Vector{Float64}, δ, ϵ, mt, vt, beta1, beta2, accuracy)
    for ci in 1:length(δ)
        Mcoeff[ci], mt, vt = MOpt_Adam(δ[ci], ci, Mcoeff[ci], mt, vt, ϵ, beta1, beta2, accuracy)
    end
    Mcoeff
end

function MOpt_Adam!(Mcoeff::Vector{Vector{Float64}}, δ, ϵ, mt, vt, beta1, beta2, accuracy)
    mt_in = mt
    vt_in = vt
    for ci in 1:length(δ)
        mt = mt_in
        vt = vt_in
        for cj in 1:length(δ[1])
            Mcoeff[ci][cj], mt, vt = MOpt_Adam(δ[ci][cj], cj, Mcoeff[ci][cj], mt, vt, ϵ, beta1, beta2, accuracy)
        end
    end
    Mcoeff
end

function MOpt_Adam!(system, δ, ϵ, mt, vt, beta1, beta2, accuracy)
    mt_in = mt
    vt_in = vt
    for ci in 1:length(δ)
        mt = mt_in
        vt = vt_in
        for cj in 1:length(δ[1])
            system.Measurement[ci][cj], mt, vt = MOpt_Adam(δ[ci][cj], cj, system.Measurement[ci][cj], mt, vt, ϵ, beta1, beta2, accuracy)
        end
    end
end

function SaveFile_meas(f_now::Float64, Measurement)
    open("f.csv","a") do f
        writedlm(f, [f_now])
    end
    open("measurements.csv","a") do g
        writedlm(g, Measurement)
    end
end

function SaveFile_meas(f_now::Vector{Float64}, Measurement)
    open("f.csv","w") do f
        writedlm(f, f_now)
    end
    open("measurements.csv","w") do g
        writedlm(g, Measurement)
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
