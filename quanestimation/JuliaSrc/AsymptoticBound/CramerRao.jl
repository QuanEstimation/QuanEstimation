using Zygote: @adjoint
const σ_x = [0.0 1.0; 1.0 0.0im]
const σ_y = [0.0 -1.0im; 1.0im 0.0]
const σ_z = [1.0 0.0im; 0.0 -1.0]


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

function QFIM_TimeIndepend(H0, ∂H_∂x::Vector{Matrix{T}}, psi0::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps::Number) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi0, decay_opt, γ, tspan)
    QFIM(ρt, ∂ρt_∂x, eps)
end

function QFIM_TimeIndepend(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps::Number) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    QFIM(ρt, ∂ρt_∂x, eps)
end

function QFIM_TimeIndepend_liouville(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps::Number) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    QFIM_liouville(ρt, ∂ρt_∂x)
end

function QFIM_saveall(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, eps::Number) where {T<:Complex,R<:Real}
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
function CFI(ρ::Matrix{T}, dρ::Matrix{T}, eps=1e-8) where {T<:Complex}
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

function CFI(ρ::Matrix{T}, dρ::Matrix{T}, M::Vector{Matrix{T}}, eps=1e-8) where {T<:Complex}
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
function CFI(M::Vector{Matrix{T}}, H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, eps::Number) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    CFI(ρt, ∂ρt_∂x, M, eps)
end

function CFI(M::Vector{Matrix{T}}, H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps::Number) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    CFI(ρt, ∂ρt_∂x, M, eps)
end

function CFI_AD(Mbasis::Vector{Vector{T}}, Mcoeff::Vector{R}, Lambda, H0::Matrix{T}, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps::Number) where {T<:Complex,R<:Real}
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

function CFI_AD(Mbasis::Vector{Matrix{T}}, Mcoeff::Vector{R}, Lambda, H0::Matrix{T}, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps::Number) where {T<:Complex,R<:Real}
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

function CFIM_TimeIndepend(M::Vector{Matrix{T}}, H0, ∂H_∂x::Matrix{T}, psi0::Vector{T}, tspan, eps::Number) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi0, tspan)
    CFI(ρt, ∂ρt_∂x, M, eps)
end

function CFIM_TimeIndepend(M::Vector{Matrix{T}}, H0, ∂H_∂x::Matrix{T}, psi::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps::Number) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi, decay_opt, γ, tspan)
    CFI(ρt, ∂ρt_∂x, M, eps)
end

function CFIM_TimeIndepend(M::Vector{Matrix{T}}, H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps::Number) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    CFI(ρt, ∂ρt_∂x, M, eps)
end

function CFI(M, system)
    CFI(M, system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ0, system.decay_opt, system.γ, 
        system.control_Hamiltonian, system.control_coefficients, system.tspan, system.eps)
end


#======================================================#
#################### calculate CFIM ####################
function CFIM(ρ, dρ, eps::Number)
    M = SIC(size(ρ)[1])
    m_num = length(M)
    p_num = length(dρ)
    cfim = [real(tr(ρ*M[i])) < eps ? zeros(ComplexF64, p_num, p_num) : (kron(tr.(dρ.*[M[i]]),reshape(tr.(dρ.*[M[i]]), 1, p_num))/tr(ρ*M[i])) for i in 1:m_num] |> sum .|>real
end

function CFIM(ρ, dρ, M, eps::Number)
    m_num = length(M)
    p_num = length(dρ)
    cfim = [real(tr(ρ*M[i])) < eps ? zeros(ComplexF64, p_num, p_num) : (kron(tr.(dρ.*[M[i]]),reshape(tr.(dρ.*[M[i]]), 1, p_num))/tr(ρ*M[i])) for i in 1:m_num] |> sum .|>real
end

function obj_func(x::Val{:CFIM}, ρ, dρ, W, M, eps::Number)
    F = CFIM(ρ, dρ, M, eps)
    return (abs(det(F)) < eps ? (1.0/eps) : real(tr(W*inv(F))))
end

#### quantum dynamics and calcalate CFIM ####
function CFIM(M::Vector{Matrix{T}}, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, eps::Number) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    CFIM(ρt, ∂ρt_∂x, M, eps)
end

function CFIM(M::Vector{Matrix{T}}, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps::Number) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    CFIM(ρt, ∂ρt_∂x, M, eps)
end

function CFIM_AD(Mbasis::Vector{Vector{T}}, Mcoeff::Vector{R}, Lambda, H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps::Number) where {T<:Complex,R<:Real}
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

function CFIM_AD(Mbasis::Vector{Matrix{T}}, Mcoeff::Vector{R}, Lambda, H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps::Number) where {T<:Complex,R<:Real}
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

function CFIM_TimeIndepend(M::Vector{Matrix{T}}, H0, ∂H_∂x::Vector{Matrix{T}}, psi::Vector{T}, tspan, eps::Number) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi, tspan)
    CFIM(ρt, ∂ρt_∂x, M, eps)
end

# CFIM for pure state with kraus rep.
function CFIM_TimeIndepend(M::Vector{Matrix{T}}, H0, ∂H_∂x::Vector{Matrix{T}}, psi::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps::Number) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi, decay_opt, γ, tspan)
    CFIM(ρt, ∂ρt_∂x, M, eps)
end

function CFIM_TimeIndepend(M::Vector{Matrix{T}}, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, eps::Number) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    CFIM(ρt, ∂ρt_∂x, M, eps)
end

function CFIM_saveall(M, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, eps::Number) where {T<:Complex,R<:Real}
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

function QFIM_Bloch(r, dr; eps=1e-8)
    para_num = length(dr)
    QFIM_res = zeros(para_num, para_num)
        
    dim = Int(sqrt(length(r)+1))
    Lambda = suN_generator(dim)
    if dim == 2
        r_norm = norm(r)^2
        if abs(r_norm-1.0) < eps
            for para_i in 1:para_num
                for para_j in para_i:para_num
                    QFIM_res[para_i, para_j] = dr[para_i]'*dr[para_j]
                    QFIM_res[para_j, para_i] = QFIM_res[para_i, para_j]
                end
            end
        else
            for para_i in 1:para_num
                for para_j in para_i:para_num
                    QFIM_res[para_i, para_j] = dr[para_i]'*dr[para_j]+(r'*dr[para_i])*(r'*dr[para_j])/(1-r_norm)
                    QFIM_res[para_j, para_i] = QFIM_res[para_i, para_j]
                end
            end
        end
    else
        rho = (Matrix(I,dim,dim)+sqrt(dim*(dim-1)/2)*r'*Lambda)/dim
        G = zeros(ComplexF64, dim^2-1, dim^2-1)
        for row_i in 1:dim^2-1
            for col_j in row_i:dim^2-1
                anti_commu = Lambda[row_i]*Lambda[col_j]+Lambda[col_j]*Lambda[row_i]
                G[row_i, col_j] = 0.5*tr(rho*anti_commu)
                G[col_j, row_i] = G[row_i, col_j]
            end
        end

        mat_tp = G*dim/(2*(dim-1))-r*r'
        mat_inv = pinv(mat_tp) 

        for para_m in 1:para_num
            for para_n in para_m:para_num
                # println(dr[para_n]*mat_inv*dr[para_m]')
                QFIM_res[para_m, para_n] = dr[para_n]'*mat_inv*dr[para_m]
                QFIM_res[para_n, para_m] = QFIM_res[para_m, para_n]
            end
        end
    end
    if para_num == 1
        return QFIM_res[1,1]
    else
        return QFIM_res
    end
end

#======================================================#
################# Gaussian States QFIM #################
function Williamson_form(A::AbstractMatrix)
    n = size(A)[1]//2 |>Int
    J = zeros(n,n)|>x->[x one(x); -one(x) x]
    A_sqrt = sqrt(A)
    B = A_sqrt*J*A_sqrt
    P = one(A)|>x->[x[:,1:2:2n-1] x[:,2:2:2n]]
    t, Q, vals = schur(B)
    c = vals[1:2:2n-1].|>imag
    D = c|>diagm|>x->x^(-0.5)
    S = (J*A_sqrt*Q*P*[zeros(n,n) -D; D zeros(n,n)]|>transpose|>inv)*transpose(P)
    return S, c
end

# const a_Gauss = [im*σ_y,σ_z,σ_x|>one, σ_x]

function A_Gauss(m::Int)
    e = bases(m)
    s = e.*e'
    a_Gauss .|> x -> [kron(s, x)/sqrt(2) for s in s]
end

function G_Gauss(S::M, dC::VM, c::V) where {M<:AbstractMatrix, V,VM<:AbstractVector}
    para_num = length(dC)
    m = size(S)[1]//2 |>Int
    As = A_Gauss(m)
    gs =  [[[inv(S)*∂ₓC*inv(transpose(S))*a'|>tr for a in A] for A in As] for ∂ₓC in dC]
    #[[inv(S)*∂ₓC*inv(transpose(S))*As[l][j,k]|>tr for j in 1:m, k in 1:m] for l in 1:4]
    G = [zero(S) for _ in 1:para_num]
    
    for i in 1:para_num
        for j in 1:m
            for k in 1:m 
                for l in 1:4
                    G[i]+=gs[i][l][j,k]/(4*c[j]c[k]+(-1)^l)*inv(transpose(S))*As[l][j,k]*inv(S)
                end
            end 
        end
    end
    G
end

function QFIM_Gauss(R̄::V, dR̄::VV, D::M, dD::VM) where {V,VV,M,VM <:AbstractVecOrMat}
    para_num = length(dR̄)
    quad_num = length(R̄)
    C = [(D[i,j] + D[j,i])/2 - R̄[i]R̄[j] for i in 1:quad_num, j in 1:quad_num]
    dC = [[(dD[k][i,j] + dD[k][j,i])/2 - dR̄[k][i]R̄[j] - R̄[i]dR̄[k][j] for i in 1:quad_num, j in 1:quad_num] for k in 1:para_num]
    S, cs = Williamson_form(C)
    Gs = G_Gauss(S, dC, cs)    
    F = [tr(Gs[i]*dC[j])+transpose(dR̄[i])*inv(C)*dR̄[j] for i in 1:para_num, j in 1:para_num]
    F|>real
end
