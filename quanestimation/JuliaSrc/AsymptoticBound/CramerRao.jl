using Zygote: @adjoint

############## logarrithmic derivative ###############
function SLD(ρ::Matrix{T}, dρ::Matrix{T}, eps; rep="original") where {T<:Complex}
    dim = size(ρ)[1]
    SLD = Matrix{ComplexF64}(undef, dim, dim)

    val, vec = eigen(ρ)
    val = val |> real
    SLD_eig = zeros(T, dim, dim)
    for fi in 1:dim
        for fj in 1:dim
            if abs(val[fi] + val[fj]) > eps
                SLD_eig[fi,fj] = 2 * (vec[:,fi]' * dρ * vec[:,fj])/(val[fi] + val[fj])
            end
        end
    end
    SLD_eig[findall(SLD_eig == Inf)] .= 0.

    if rep=="original"
        SLD = vec*(SLD_eig*vec')
    elseif rep=="eigen"
        SLD = SLD_eig
    end
    SLD
end

@adjoint function SLD(ρ::Matrix{T}, dρ::Matrix{T}, eps) where {T <: Complex}
    L = SLD(ρ, dρ, eps)
    SLD_pullback = L̄ -> (Ḡ -> (-Ḡ*L-L*Ḡ, 2*Ḡ, nothing))(SLD((ρ)|>Array, L̄/2, eps))
    L, SLD_pullback
end 

function SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}, eps) where {T<:Complex}
    (x->SLD(ρ, x, eps)).(dρ)
end


function SLD_liouville(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T<:Complex}
    2 * pinv(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ)) * vec(∂ρ_∂x) |> vec2mat
end

function SLD_liouville(ρ::Vector{T},∂ρ_∂x::Vector{T}) where {T<:Complex}
    SLD_liouville(ρ |> vec2mat, ∂ρ_∂x |> vec2mat)
end

function SLD_liouville(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}) where {T<:Complex}
    (x->SLD_liouville(ρ, x)).(∂ρ_∂x)
end

function SLD_qr(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T<:Complex}
    2 * (qr(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ), Val(true)) \ vec(∂ρ_∂x)) |> vec2mat
end

function RLD(ρ::Matrix{T}, dρ::Matrix{T}, eps) where {T<:Complex}
    pinv(ρ, rtol=eps) * dρ
end

function LLD(ρ::Matrix{T}, dρ::Matrix{T}, eps) where {T<:Complex}
    (dρ * pinv(ρ, rtol=eps))'
end

#========================================================#
####################### calculate QFI ####################
function QFI(ρ, dρ, eps)
    SLD_tp = SLD(ρ, dρ, eps)
    SLD2_tp = SLD_tp * SLD_tp
    F = tr(ρ * SLD2_tp)
    F |> real
end

function QFI_RLD(ρ, dρ::Matrix{T}, eps) where {T<:Complex}
    RLD_tp = RLD(ρ, dρ, eps)
    F = tr(ρ * RLD_tp * RLD_tp')
    F |> real
end

function QFI_LLD(ρ, dρ::Matrix{T}, eps) where {T<:Complex}
    LLD_tp = LLD(ρ, dρ, eps)
    F = tr(ρ * LLD_tp * LLD_tp')
    F |> real
end

function QFIM_pure(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T<:Complex}
    SLD = 2*∂ρ_∂x
    SLD2_tp = SLD * SLD
    F = tr(ρ * SLD2_tp)
    F |> real
end

#### quantum dynamics and calcalate QFI ####
function QFI(arr, H0, ∂H_∂x::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, tspan, eps) where {T<:Complex,R<:Real}
    ρ0 = arr[1]*arr[1]'
    control_coefficients = arr[2]
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    QFI(ρt, ∂ρt_∂x, eps)
end

function QFI(H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    QFI(ρt, ∂ρt_∂x, eps)
end

function QFI(H0, ∂H_∂x::Matrix{T}, psi0::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, eps) where {T<:Complex,R<:Real}
    ρ0 = psi0*psi0'
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    QFI(ρt, ∂ρt_∂x, eps)
end

function QFI_liouville(H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ,control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    QFI_liouville(ρt, ∂ρt_∂x)
end

function QFIM_TimeIndepend(H0, ∂H_∂x::Matrix{T}, psi0::Vector{T}, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi0, tspan)
    QFIM_pure(ρt, ∂ρt_∂x)
end

function QFIM_TimeIndepend(H0, ∂H_∂x::Matrix{T}, psi0::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi0, decay_opt, γ, tspan)
    QFIM(ρt, ∂ρt_∂x, eps)
end

function QFIM_TimeIndepend(H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    QFIM(ρt, ∂ρt_∂x, eps)
end

function QFIM_TimeIndepend_liouville(H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    QFI_liouville(ρt, ∂ρt_∂x)
end

function QFI(system)
    QFI(system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ0, system.decay_opt, system.γ, system.control_Hamiltonian, system.control_coefficients, system.tspan, system.eps)
end

function QFI_liouville(system)
    QFI_liouville(system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ0, system.decay_opt, system.γ, system.control_Hamiltonian, system.control_coefficients, system.tspan, system.eps)
end


#==========================================================#
####################### calculate QFIM #####################
function QFIM(ρ, dρ, eps=1e-8)
    p_num = length(dρ)
    LD_tp = SLD(ρ, dρ, eps)
    qfim = ([0.5*ρ] .* (kron(LD_tp, reshape(LD_tp,1,p_num)) + kron(reshape(LD_tp,1,p_num), LD_tp))).|> tr .|>real 
end

function QFIM_RLD(ρ, dρ, eps=1e-8)
    p_num = length(dρ)
    LD_tp = RLD(ρ, dρ, eps)
    LD_dag = [LD_tp[i]' for i in 1:p_num]
    qfim = ([ρ] .* (kron(LD_tp, reshape(LD_dag,1,p_num)))).|> tr .|>real 
end

function QFIM_LLD(ρ, dρ, eps=1e-8)
    p_num = length(dρ)
    LD_tp = LLD(ρ, dρ, eps)
    LD_dag = [LD_tp[i]' for i in 1:p_num]
    qfim = ([ρ] .* (kron(LD_tp, reshape(LD_dag,1,p_num)))).|> tr .|>real 
end

function QFIM_liouville(ρ, dρ)
    p_num = length(dρ)
    LD_tp = SLD_liouville(ρ, dρ)
    qfim = ([0.5*ρ] .* (kron(LD_tp, reshape(LD_tp,1,p_num)) + kron(reshape(LD_tp,1,p_num), LD_tp))).|> tr .|>real 
end

function QFIM_pure(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}) where {T<:Complex}
    p_num = length(∂ρ_∂x)
    SLD = [2*∂ρ_∂x[i] for i in 1:p_num]
    qfim = ([0.5*ρ] .* (kron(SLD, reshape(SLD,1,p_num)) + kron(reshape(SLD,1,p_num), SLD))) .|> tr .|>real
end

function obj_func(x::Val{:QFIM}, ρ, dρ, W, M, eps)
    F = QFIM(ρ, dρ, eps)
    return (abs(det(F)) < eps ? (1.0/eps) : real(tr(W*inv(F))))
end

#### quantum dynamics and calcalate QFIM ####

function QFIM(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    QFIM(ρt, ∂ρt_∂x, eps)
end

function QFIM(H0, ∂H_∂x::Vector{Matrix{T}}, psi0::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, eps) where {T<:Complex,R<:Real}
    ρ0 = psi0*psi0'
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    QFIM(ρt, ∂ρt_∂x, eps)
end

function QFIM(arr, H0, ∂H_∂x::Vector{Matrix{T}}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, tspan, eps) where {T<:Complex,R<:Real}
    ρ0 = arr[1]*arr[1]'
    control_coefficients = arr[2]
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    QFIM(ρt, ∂ρt_∂x, eps)
end

function QFIM(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    QFIM(ρt, ∂ρt_∂x, eps)
end

function QFIM_liouville(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    QFIM_liouville(ρt, ∂ρt_∂x)
end

function QFIM_TimeIndepend(H0, ∂H_∂x::Vector{Matrix{T}}, psi0::Vector{T}, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi0, tspan)
    QFIM_pure(ρt, ∂ρt_∂x)
end

function QFIM_TimeIndepend(H0, ∂H_∂x::Vector{Matrix{T}}, psi0::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi0, decay_opt, γ, tspan)
    QFIM(ρt, ∂ρt_∂x, eps)
end

function QFIM_TimeIndepend(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    QFIM(ρt, ∂ρt_∂x, eps)
end

function QFIM_TimeIndepend_liouville(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    QFIM_liouville(ρt, ∂ρt_∂x)
end

function QFIM_saveall(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = expm(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    F = [Matrix{Float64}(undef, length(∂H_∂x), length(∂H_∂x)) for i in 1:length(tspan)] 
    for t in 2:length(tspan)
        F[t] = QFIM(ρt, ∂ρt_∂x, eps)
    end
    return F
end

function obj_func(x::Val{:QFIM}, system, M)
    F = QFIM(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, system.control_Hamiltonian, 
                system.control_coefficients, system.tspan, system.eps)
    return (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:QFIM}, system, M, control_coeff)
    F = QFIM(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, system.control_Hamiltonian, 
                control_coeff, system.tspan, system.eps)
    return (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:QFIM_noctrl}, system, M, psi)
    ρ0 = psi*psi'
    F = QFIM(system.freeHamiltonian, system.Hamiltonian_derivative, ρ0, system.decay_opt, system.γ, system.tspan, system.eps)
    return  (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:QFIM_TimeIndepend_noiseless}, system, M)
    F = QFIM_TimeIndepend(system.freeHamiltonian, system.Hamiltonian_derivative, system.psi, system.tspan, system.eps)
    return (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:QFIM_TimeIndepend_noiseless}, system, M, psi)
    F = QFIM_TimeIndepend(system.freeHamiltonian, system.Hamiltonian_derivative, psi, system.tspan, system.eps)
    return (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:QFIM_TimeIndepend_noise}, system, M)
    F = QFIM_TimeIndepend(system.freeHamiltonian, system.Hamiltonian_derivative, system.psi, system.decay_opt, 
                             system.γ, system.tspan, system.eps)
    return (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:QFIM_TimeIndepend_noise}, system, M, psi)
    F = QFIM_TimeIndepend(system.freeHamiltonian, system.Hamiltonian_derivative, psi, system.decay_opt, 
                             system.γ, system.tspan, system.eps)
    return (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:QFIM_noctrl}, system, M)
    F = QFIM(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, system.tspan, system.eps)
    return (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function QFIM_liouville(system)
    QFIM_liouville(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, 
              system.control_Hamiltonian, system.control_coefficients, system.tspan, system.eps)
end

function QFIM_saveall(system)
    QFIM_saveall(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, 
                 system.control_Hamiltonian, system.control_coefficients, system.tspan, system.eps)
end

#==========================================================#
####################### calculate CFI ######################
function CFI(ρ::Matrix{T}, dρ::Matrix{T}; eps=1e-8) where {T<:Complex}
    data = readdlm("$(Main.pkgpath)/sic_fiducial_vectors/d$(size(ρ)[1]).txt", '\t', Float64, '\n')
    fiducial = data[:,1]+1.0im*data[:,2]
    M = sic_povm(fiducial)

    m_num = length(M)
    F = 0.
    for i in 1:m_num
        mp = M[i]
        p = real(tr(ρ * mp))
        dp = real(tr(dρ * mp))
        cadd = 0.
        if p > eps
            cadd = (dp*dp) / p
        end
        F += cadd
    end 
    real(F)
end

function CFI(ρ::Matrix{T}, dρ::Matrix{T}, M::Vector{Matrix{T}}; eps=1e-8) where {T<:Complex}
    m_num = length(M)
    F = 0.
    for i in 1:m_num
        mp = M[i]
        p = real(tr(ρ * mp))
        dp = real(tr(dρ * mp))
        cadd = 0.
        if p > eps
            cadd = (dp*dp) / p
        end
        F += cadd
    end 
    real(F)
end

#### quantum dynamics and calcalate CFI ####
function CFI(M::Vector{Matrix{T}}, H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    CFI(ρt, ∂ρt_∂x, M, eps)
end

function CFI(M::Vector{Matrix{T}}, H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    CFI(ρt, ∂ρt_∂x, M, eps)
end

function CFI_AD(Mbasis::Vector{Vector{T}}, Mcoeff::Vector{R}, Lambda, H0::Matrix{T}, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps) where {T<:Complex,R<:Real}
    dim = size(ρ0)[1]
    U = rotation_matrix(Mcoeff, Lambda)
    M = [U*Mbasis[i]*Mbasis[i]'*U' for i in 1:length(Mbasis)]

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero 
    expL = evolute(H0, decay_opt, γ, Δt, 1)
    ∂H_L = liouville_commu(∂H_∂x)
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    ρt = exp(vec(H0)' * zero(ρt)) * ρt
    CFI(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, M, eps)
end

function CFI_AD(Mbasis::Vector{Matrix{T}}, Mcoeff::Vector{R}, Lambda, H0::Matrix{T}, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps) where {T<:Complex,R<:Real}
    dim = size(ρ0)[1]
    U = rotation_matrix(Mcoeff, Lambda)
    M = [U*Mbasis[i]*U' for i in 1:length(Mbasis)]

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero 
    expL = evolute(H0, decay_opt, γ, Δt, 1)
    ∂H_L = liouville_commu(∂H_∂x)
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    ρt = exp(vec(H0)' * zero(ρt)) * ρt
    CFI(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, M, eps)
end

function CFIM_TimeIndepend(M::Vector{Matrix{T}}, H0, ∂H_∂x::Matrix{T}, psi0::Vector{T}, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi0, tspan)
    CFI(ρt, ∂ρt_∂x, M, eps)
end

function CFIM_TimeIndepend(M::Vector{Matrix{T}}, H0, ∂H_∂x::Matrix{T}, psi::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi, decay_opt, γ, tspan)
    CFI(ρt, ∂ρt_∂x, M, eps)
end

function CFIM_TimeIndepend(M::Vector{Matrix{T}}, H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    CFI(ρt, ∂ρt_∂x, M, eps)
end

function CFI(M, system)
    CFI(M, system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ0, system.decay_opt, system.γ, 
        system.control_Hamiltonian, system.control_coefficients, system.tspan, system.eps)
end


#======================================================#
#################### calculate CFIM ####################
function CFIM(ρ, dρ, eps)
    M = load_M(size(ρ)[1])
    m_num = length(M)
    p_num = length(dρ)
    cfim = [real(tr(ρ*M[i])) < eps ? zeros(ComplexF64, p_num, p_num) : (kron(tr.(dρ.*[M[i]]),reshape(tr.(dρ.*[M[i]]), 1, p_num))/tr(ρ*M[i])) for i in 1:m_num] |> sum .|>real
end

function CFIM(ρ, dρ, M, eps)
    m_num = length(M)
    p_num = length(dρ)
    cfim = [real(tr(ρ*M[i])) < eps ? zeros(ComplexF64, p_num, p_num) : (kron(tr.(dρ.*[M[i]]),reshape(tr.(dρ.*[M[i]]), 1, p_num))/tr(ρ*M[i])) for i in 1:m_num] |> sum .|>real
end

function obj_func(x::Val{:CFIM}, ρ, dρ, W, M, eps)
    F = CFIM(ρ, dρ, M, eps)
    return (abs(det(F)) < eps ? (1.0/eps) : real(tr(W*inv(F))))
end

#### quantum dynamics and calcalate CFIM ####
function CFIM(M::Vector{Matrix{T}}, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    CFIM(ρt, ∂ρt_∂x, M, eps)
end

function CFIM(M::Vector{Matrix{T}}, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    CFIM(ρt, ∂ρt_∂x, M, eps)
end

function CFIM_AD(Mbasis::Vector{Vector{T}}, Mcoeff::Vector{R}, Lambda, H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps) where {T<:Complex,R<:Real}
    dim = size(ρ0)[1]
    U = Matrix{ComplexF64}(I,dim,dim)
    U = rotation_matrix(Mcoeff, Lambda)
    M = [U*Mbasis[i]*Mbasis[i]'*U' for i in 1:length(Mbasis)]

    Δt = tspan[2] - tspan[1]
    para_num = length(∂H_∂x)
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    expL = evolute(H0, decay_opt, γ, Δt, 1)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]
    F = [Matrix{Float64}(undef, para_num, para_num) for i in 1:length(tspan)] 
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    CFIM(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, M, eps)
end

function CFIM_AD(Mbasis::Vector{Matrix{T}}, Mcoeff::Vector{R}, Lambda, H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps) where {T<:Complex,R<:Real}
    dim = size(ρ0)[1]
    U = Matrix{ComplexF64}(I,dim,dim)
    U = rotation_matrix(Mcoeff, Lambda)
    M = [U*Mbasis[i]*U' for i in 1:length(Mbasis)]

    Δt = tspan[2] - tspan[1]
    para_num = length(∂H_∂x)
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    expL = evolute(H0, decay_opt, γ, Δt, 1)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]
    F = [Matrix{Float64}(undef, para_num, para_num) for i in 1:length(tspan)] 
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    CFIM(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, M, eps)
end

function CFIM_TimeIndepend(M::Vector{Matrix{T}}, H0, ∂H_∂x::Vector{Matrix{T}}, psi::Vector{T}, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi, tspan)
    CFIM(ρt, ∂ρt_∂x, M, eps)
end

function CFIM_TimeIndepend(M::Vector{Matrix{T}}, H0, ∂H_∂x::Vector{Matrix{T}}, psi::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi, decay_opt, γ, tspan)
    CFIM(ρt, ∂ρt_∂x, M, eps)
end

function CFIM_TimeIndepend(M::Vector{Matrix{T}}, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    CFIM(ρt, ∂ρt_∂x, M, eps)
end

function CFIM_saveall(M, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, eps) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = expm(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    F = [Matrix{Float64}(undef, length(∂H_∂x), length(∂H_∂x)) for i in 1:length(tspan)]
    for t in 2:length(tspan)
        F[t] = CFIM(ρt, ∂ρt_∂x_all, M, eps)
    end
    return F
end

function obj_func(x::Val{:CFIM}, system, M)
    F = CFIM(M, system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, 
            system.control_Hamiltonian, system.control_coefficients, system.tspan, system.eps)
    return (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:CFIM}, system, M, control_coeff)
    F = CFIM(M, system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ,  
            system.control_Hamiltonian, control_coeff, system.tspan, system.eps)
    return (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:CFIM_TimeIndepend_noiseless}, system, M)
    F = CFIM_TimeIndepend(M, system.freeHamiltonian, system.Hamiltonian_derivative, system.psi, system.tspan, system.eps)
    return (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:CFIM_TimeIndepend_noiseless}, system, M, psi)
    F = CFIM_TimeIndepend(M, system.freeHamiltonian, system.Hamiltonian_derivative, psi, system.tspan, system.eps)
    return (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:CFIM_TimeIndepend_noise}, system, M)
    F = CFIM_TimeIndepend(M, system.freeHamiltonian, system.Hamiltonian_derivative, system.psi, system.decay_opt, 
                             system.γ, system.tspan, system.eps)
    return (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:CFIM_TimeIndepend_noise}, system, M, psi)
    F = CFIM_TimeIndepend(M, system.freeHamiltonian, system.Hamiltonian_derivative, psi, system.decay_opt, 
                             system.γ, system.tspan, system.eps)
    return (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:CFIM_noctrl}, system, M)
    F = CFIM(M, system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, system.tspan, system.eps)
    return  (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:QFIM_SCopt}, system, M, psi, control_coefficients)
    rho = psi*psi'
    F = QFIM(system.freeHamiltonian, system.Hamiltonian_derivative, rho, system.decay_opt, system.γ, system.control_Hamiltonian, control_coefficients, system.tspan, system.eps)
    return  (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:CFIM_SCopt}, system, M, psi, control_coefficients)
    rho = psi*psi'
    F = CFIM(M, system.freeHamiltonian, system.Hamiltonian_derivative, rho, system.decay_opt, system.γ, system.control_Hamiltonian, control_coefficients, system.tspan, system.eps)
    return  (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:CFIM_SMopt}, system, psi, M)
    rho = psi*psi'
    F = CFIM(M, system.freeHamiltonian, system.Hamiltonian_derivative, rho, system.decay_opt, system.γ, system.tspan, system.eps)
    return  (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:CFIM_CMopt}, system, M, rho, control_coefficients)
    F = CFIM(M, system.freeHamiltonian, system.Hamiltonian_derivative, rho, system.decay_opt, system.γ, system.control_Hamiltonian, control_coefficients, system.tspan, system.eps)
    return  (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function QFIM(ρ, dρ::Matrix{T}; dtype="SLD", exportLD=false, eps=1e-8) where {T<:Complex}
    if exportLD==false
        if dtype=="SLD"
            F = QFI(ρ, dρ, eps)
        elseif dtype=="RLD"
            F = QFI_RLD(ρ, dρ, eps)
        elseif dtype=="LLD"
            F = QFI_LLD(ρ, dρ, eps)
        end
        return F
    else
        if dtype=="SLD"
            LD = SLD(ρ, dρ, eps)
            F = QFI(ρ, dρ, eps)
        elseif dtype=="RLD"
            LD = RLD(ρ, dρ, eps)
            F = QFI_RLD(ρ, dρ, eps)
        elseif dtype=="LLD"
            LD = LLD(ρ, dρ, eps)
            F = QFI_LLD(ρ, dρ, eps)
        end
        return F, LD
    end
end

function QFIM(ρ, dρ::Vector{Matrix{T}}; dtype="SLD", exportLD=false, eps=1e-8) where {T<:Complex}
    if exportLD==false
        if length(dρ) == 1
            if dtype=="SLD"
                F = QFI(ρ, dρ[1], eps)
            elseif dtype=="RLD"
                F = QFI_RLD(ρ, dρ[1], eps)
            elseif dtype=="LLD"
                F = QFI_LLD(ρ, dρ[1], eps)
            end
            return F
        else
            if dtype=="SLD"
                F = QFIM(ρ, dρ, eps)
            elseif dtype=="RLD"
                F = QFIM_RLD(ρ, dρ, eps)
            elseif dtype=="LLD"
                F = QFIM_LLD(ρ, dρ, eps)
            end
            return F
        end
    else
        if length(dρ) == 1
            if dtype=="SLD"
                LD = SLD(ρ, dρ[1], eps)
                F = QFI(ρ, dρ[1], eps)
            elseif dtype=="RLD"
                LD = RLD(ρ, dρ[1], eps)
                F = QFI_RLD(ρ, dρ[1], eps)
            elseif dtype=="LLD"
                LD = LLD(ρ, dρ[1], eps)
                F = QFI_LLD(ρ, dρ[1], eps)
            end
            return F
        else
            if dtype=="SLD"
                LD = SLD(ρ, dρ, eps)
                F = QFIM(ρ, dρ, eps)
            elseif dtype=="RLD"
                LD = RLD(ρ, dρ, eps)
                F = QFIM_RLD(ρ, dρ, eps)
            elseif dtype=="LLD"
                LD = LLD(ρ, dρ, eps)
                F = QFIM_LLD(ρ, dρ, eps)
            end
            return F, LD
        end
    end
end
