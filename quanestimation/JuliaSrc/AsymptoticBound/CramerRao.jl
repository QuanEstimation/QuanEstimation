using Zygote: @adjoint

############## logarrithmic derivative ###############
function SLD(ρ::Matrix{T}, dρ::Matrix{T}, accuracy; rep="original") where {T<:Complex}
    dim = size(ρ)[1]
    SLD = Matrix{ComplexF64}(undef, dim, dim)

    val, vec = eigen(ρ)
    val = val |> real
    SLD_eig = zeros(T, dim, dim)
    for fi in 1:dim
        for fj in 1:dim
            if abs(val[fi] + val[fj]) > accuracy
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

@adjoint function SLD(ρ::Matrix{T}, dρ::Matrix{T}, accuracy) where {T <: Complex}
    L = SLD(ρ, dρ, accuracy)
    SLD_pullback = L̄ -> (Ḡ -> (-Ḡ*L-L*Ḡ, 2*Ḡ, nothing))(SLD((ρ)|>Array, L̄/2, accuracy))
    L, SLD_pullback
end 

function SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}, accuracy; rep="original") where {T<:Complex}
    (x->SLD(ρ, x, accuracy, rep=rep)).(dρ)
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

function RLD(ρ::Matrix{T}, dρ::Matrix{T}, accuracy) where {T<:Complex}
    pinv(ρ, rtol=accuracy) * dρ
end

function LLD(ρ::Matrix{T}, dρ::Matrix{T}, accuracy) where {T<:Complex}
    (dρ * pinv(ρ, rtol=accuracy))'
end

#========================================================#
####################### calculate QFI ####################
function QFI(ρ, dρ, accuracy=1e-8)
    SLD_tp = SLD(ρ, dρ, accuracy)
    SLD2_tp = SLD_tp * SLD_tp
    F = tr(ρ * SLD2_tp)
    F |> real
end

function QFI_RLD(ρ, dρ)
    p_num = length(dρ)
    RLD_tp = RLD(ρ, dρ)
    F = tr(ρ * RLD_tp * reshape(RLD_tp,1,p_num))
    F |> real
end

function QFIM_pure(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T<:Complex}
    SLD = 2*∂ρ_∂x
    SLD2_tp = SLD * SLD
    F = tr(ρ * SLD2_tp)
    F |> real
end

#### quantum dynamics and calcalate QFI ####

function QFI(H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    QFI(ρt, ∂ρt_∂x, accuracy)
end

function QFI_liouville(H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ,control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    QFI_liouville(ρt, ∂ρt_∂x)
end

function QFIM_TimeIndepend(H0, ∂H_∂x::Matrix{T}, psi0::Vector{T}, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi0, tspan)
    QFIM_pure(ρt, ∂ρt_∂x)
end

function QFIM_TimeIndepend(H0, ∂H_∂x::Matrix{T}, psi0::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi0, decay_opt, γ, tspan)
    QFIM(ρt, ∂ρt_∂x, accuracy)
end

function QFIM_TimeIndepend(H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    QFIM(ρt, ∂ρt_∂x, accuracy)
end

function QFIM_TimeIndepend_liouville(H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    QFI_liouville(ρt, ∂ρt_∂x)
end

function QFI(system)
    QFI(system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ0, system.decay_opt, system.γ, system.control_Hamiltonian, system.control_coefficients, system.tspan, system.accuracy)
end

function QFI_liouville(system)
    QFI_liouville(system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ0, system.decay_opt, system.γ, system.control_Hamiltonian, system.control_coefficients, system.tspan, system.accuracy)
end


#==========================================================#
####################### calculate QFIM #####################
function QFIM(ρ, dρ, accuracy)
    p_num = length(dρ)
    SLD_tp = SLD(ρ, dρ, accuracy)
    qfim = ([0.5*ρ] .* (kron(SLD_tp, reshape(SLD_tp,1,p_num)) + kron(reshape(SLD_tp,1,p_num), SLD_tp))).|> tr .|>real 
end

function QFIM_liouville(ρ, dρ)
    p_num = length(dρ)
    SLD_tp = SLD_liouville(ρ, dρ)
    qfim = ([0.5*ρ] .* (kron(SLD_tp, reshape(SLD_tp,1,p_num)) + kron(reshape(SLD_tp,1,p_num), SLD_tp))).|> tr .|>real 
end

function QFIM_pure(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}) where {T<:Complex}
    p_num = length(∂ρ_∂x)
    SLD = [2*∂ρ_∂x[i] for i in 1:p_num]
    qfim = ([0.5*ρ] .* (kron(SLD, reshape(SLD,1,p_num)) + kron(reshape(SLD,1,p_num), SLD))) .|> tr .|>real
end

function obj_func(x::Val{:QFIM}, ρ, dρ, W, Measurement, accuracy)
    F = QFIM(ρ, dρ, accuracy)
    return (abs(det(F)) < accuracy ? (1.0/accuracy) : real(tr(W*inv(F))))
end

#### quantum dynamics and calcalate QFIM ####

function QFIM(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    QFIM(ρt, ∂ρt_∂x, accuracy)
end

function QFIM(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    QFIM(ρt, ∂ρt_∂x, accuracy)
end

function QFIM_liouville(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    QFIM_liouville(ρt, ∂ρt_∂x)
end

function QFIM_TimeIndepend(H0, ∂H_∂x::Vector{Matrix{T}}, psi0::Vector{T}, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi0, tspan)
    QFIM_pure(ρt, ∂ρt_∂x)
end

function QFIM_TimeIndepend(H0, ∂H_∂x::Vector{Matrix{T}}, psi0::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi0, decay_opt, γ, tspan)
    QFIM(ρt, ∂ρt_∂x, accuracy)
end

function QFIM_TimeIndepend(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    QFIM(ρt, ∂ρt_∂x, accuracy)
end

function QFIM_TimeIndepend_liouville(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    QFIM_liouville(ρt, ∂ρt_∂x)
end

function QFIM_saveall(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = expm(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    F = [Matrix{Float64}(undef, length(∂H_∂x), length(∂H_∂x)) for i in 1:length(tspan)] 
    for t in 2:length(tspan)
        F[t] = QFIM(ρt, ∂ρt_∂x, accuracy)
    end
    return F
end

function obj_func(x::Val{:QFIM}, system, Measurement)
    F = QFIM(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, system.control_Hamiltonian, 
                system.control_coefficients, system.tspan, system.accuracy)
    return (abs(det(F)) < system.accuracy ? (1.0/system.accuracy) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:QFIM}, system, Measurement, control_coeff)
    F = QFIM(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, system.control_Hamiltonian, 
                control_coeff, system.tspan, system.accuracy)
    return (abs(det(F)) < system.accuracy ? (1.0/system.accuracy) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:QFIM_TimeIndepend_noiseless}, system, Measurement)
    F = QFIM_TimeIndepend(system.freeHamiltonian, system.Hamiltonian_derivative, system.psi, system.tspan, system.accuracy)
    return (abs(det(F)) < system.accuracy ? (1.0/system.accuracy) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:QFIM_TimeIndepend_noiseless}, system, Measurement, psi)
    F = QFIM_TimeIndepend(system.freeHamiltonian, system.Hamiltonian_derivative, psi, system.tspan, system.accuracy)
    return (abs(det(F)) < system.accuracy ? (1.0/system.accuracy) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:QFIM_TimeIndepend_noise}, system, Measurement)
    F = QFIM_TimeIndepend(system.freeHamiltonian, system.Hamiltonian_derivative, system.psi, system.decay_opt, 
                             system.γ, system.tspan, system.accuracy)
    return (abs(det(F)) < system.accuracy ? (1.0/system.accuracy) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:QFIM_TimeIndepend_noise}, system, Measurement, psi)
    F = QFIM_TimeIndepend(system.freeHamiltonian, system.Hamiltonian_derivative, psi, system.decay_opt, 
                             system.γ, system.tspan, system.accuracy)
    return (abs(det(F)) < system.accuracy ? (1.0/system.accuracy) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:QFIM_noctrl}, system, Measurement)
    F = QFIM(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, system.tspan, system.accuracy)
    return (abs(det(F)) < system.accuracy ? (1.0/system.accuracy) : real(tr(system.W*inv(F))))
end

function QFIM_liouville(system)
    QFIM_liouville(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, 
              system.control_Hamiltonian, system.control_coefficients, system.tspan, system.accuracy)
end

function QFIM_saveall(system)
    QFIM_saveall(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, 
                 system.control_Hamiltonian, system.control_coefficients, system.tspan, system.accuracy)
end

#==========================================================#
####################### calculate CFI ######################
function CFI(ρ, dρ, accuracy=1e-8)
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
        if p > accuracy
            cadd = (dp*dp) / p
        end
        F += cadd
    end 
    real(F)
end

function CFI(ρ, dρ, M, accuracy=1e-8)

    m_num = length(M)
    F = 0.
    for i in 1:m_num
        mp = M[i]
        p = real(tr(ρ * mp))
        dp = real(tr(dρ * mp))
        cadd = 0.
        if p > accuracy
            cadd = (dp*dp) / p
        end
        F += cadd
    end 
    real(F)
end

#### quantum dynamics and calcalate CFI ####
function CFI(Measurement::Vector{Matrix{T}}, H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    CFI(ρt, ∂ρt_∂x, Measurement, accuracy)
end

function CFI(Measurement::Vector{Matrix{T}}, H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    CFI(ρt, ∂ρt_∂x, Measurement, accuracy)
end

function CFI_AD(Mbasis::Vector{Vector{T}}, Mcoeff::Vector{R}, Lambda, H0::Matrix{T}, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    dim = size(ρ0)[1]
    U = rotation_matrix(Mcoeff, Lambda)
    Measurement = [U*Mbasis[i]*Mbasis[i]'*U' for i in 1:length(Mbasis)]

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
    CFI(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, Measurement, accuracy)
end

function CFI_AD(Mbasis::Vector{Matrix{T}}, Mcoeff::Vector{R}, Lambda, H0::Matrix{T}, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    dim = size(ρ0)[1]
    U = rotation_matrix(Mcoeff, Lambda)
    Measurement = [U*Mbasis[i]*U' for i in 1:length(Mbasis)]

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
    CFI(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, Measurement, accuracy)
end

function CFIM_TimeIndepend(Measurement::Vector{Matrix{T}}, H0, ∂H_∂x::Matrix{T}, psi0::Vector{T}, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi0, tspan)
    CFI(ρt, ∂ρt_∂x, Measurement, accuracy)
end

function CFIM_TimeIndepend(Measurement::Vector{Matrix{T}}, H0, ∂H_∂x::Matrix{T}, psi::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi, decay_opt, γ, tspan)
    CFI(ρt, ∂ρt_∂x, Measurement, accuracy)
end

function CFIM_TimeIndepend(Measurement::Vector{Matrix{T}}, H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    CFI(ρt, ∂ρt_∂x, Measurement, accuracy)
end

function CFI(Measurement, system)
    CFI(Measurement, system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ0, system.decay_opt, system.γ, 
        system.control_Hamiltonian, system.control_coefficients, system.tspan, system.accuracy)
end


#======================================================#
#################### calculate CFIM ####################
function CFIM(ρ, dρ, accuracy)
    data = readdlm("$(Main.pkgpath)/sic_fiducial_vectors/d$(size(ρ)[1]).txt", '\t', Float64, '\n')
    fiducial = data[:,1]+1.0im*data[:,2]
    M = sic_povm(fiducial)
    m_num = length(M)
    p_num = length(dρ)
    cfim = [real(tr(ρ*M[i])) < accuracy ? zeros(ComplexF64, p_num, p_num) : (kron(tr.(dρ.*[M[i]]),reshape(tr.(dρ.*[M[i]]), 1, p_num))/tr(ρ*M[i])) for i in 1:m_num] |> sum .|>real
end

function CFIM(ρ, dρ, M, accuracy)
    m_num = length(M)
    p_num = length(dρ)
    cfim = [real(tr(ρ*M[i])) < accuracy ? zeros(ComplexF64, p_num, p_num) : (kron(tr.(dρ.*[M[i]]),reshape(tr.(dρ.*[M[i]]), 1, p_num))/tr(ρ*M[i])) for i in 1:m_num] |> sum .|>real
end

function obj_func(x::Val{:CFIM}, ρ, dρ, W, Measurement, accuracy)
    F = CFIM(ρ, dρ, Measurement, accuracy)
    return (abs(det(F)) < accuracy ? (1.0/accuracy) : real(tr(W*inv(F))))
end

#### quantum dynamics and calcalate CFIM ####
function CFIM(Measurement::Vector{Matrix{T}}, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    CFIM(ρt, ∂ρt_∂x, Measurement, accuracy)
end

function CFIM(Measurement::Vector{Matrix{T}}, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    CFIM(ρt, ∂ρt_∂x, Measurement, accuracy)
end

function CFIM_AD(Mbasis::Vector{Vector{T}}, Mcoeff::Vector{R}, Lambda, H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    dim = size(ρ0)[1]
    U = Matrix{ComplexF64}(I,dim,dim)
    U = rotation_matrix(Mcoeff, Lambda)
    Measurement = [U*Mbasis[i]*Mbasis[i]'*U' for i in 1:length(Mbasis)]

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
    CFIM(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, Measurement, accuracy)
end

function CFIM_AD(Mbasis::Vector{Matrix{T}}, Mcoeff::Vector{R}, Lambda, H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    dim = size(ρ0)[1]
    U = Matrix{ComplexF64}(I,dim,dim)
    U = rotation_matrix(Mcoeff, Lambda)
    Measurement = [U*Mbasis[i]*U' for i in 1:length(Mbasis)]

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
    CFIM(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, Measurement, accuracy)
end

function CFIM_TimeIndepend(Measurement::Vector{Matrix{T}}, H0, ∂H_∂x::Vector{Matrix{T}}, psi::Vector{T}, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi, tspan)
    CFIM(ρt, ∂ρt_∂x, Measurement, accuracy)
end

function CFIM_TimeIndepend(Measurement::Vector{Matrix{T}}, H0, ∂H_∂x::Vector{Matrix{T}}, psi::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, psi, decay_opt, γ, tspan)
    CFIM(ρt, ∂ρt_∂x, Measurement, accuracy)
end

function CFIM_TimeIndepend(Measurement::Vector{Matrix{T}}, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(H0, ∂H_∂x, ρ0, decay_opt, γ, tspan)
    CFIM(ρt, ∂ρt_∂x, Measurement, accuracy)
end

function CFIM_saveall(Measurement, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, accuracy) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = expm(H0, ∂H_∂x, ρ0, decay_opt, γ, control_Hamiltonian, control_coefficients, tspan)
    F = [Matrix{Float64}(undef, length(∂H_∂x), length(∂H_∂x)) for i in 1:length(tspan)]
    for t in 2:length(tspan)
        F[t] = CFIM(ρt, ∂ρt_∂x_all, Measurement, accuracy)
    end
    return F
end

function obj_func(x::Val{:CFIM}, system, Measurement)
    F = CFIM(Measurement, system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, 
            system.control_Hamiltonian, system.control_coefficients, system.tspan, system.accuracy)
    return (abs(det(F)) < system.accuracy ? (1.0/system.accuracy) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:CFIM}, system, Measurement, control_coeff)
    F = CFIM(Measurement, system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ,  
            system.control_Hamiltonian, control_coeff, system.tspan, system.accuracy)
    return (abs(det(F)) < system.accuracy ? (1.0/system.accuracy) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:CFIM_TimeIndepend_noiseless}, system, Measurement)
    F = CFIM_TimeIndepend(Measurement, system.freeHamiltonian, system.Hamiltonian_derivative, system.psi, system.tspan, system.accuracy)
    return (abs(det(F)) < system.accuracy ? (1.0/system.accuracy) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:CFIM_TimeIndepend_noiseless}, system, Measurement, psi)
    F = CFIM_TimeIndepend(Measurement, system.freeHamiltonian, system.Hamiltonian_derivative, psi, system.tspan, system.accuracy)
    return (abs(det(F)) < system.accuracy ? (1.0/system.accuracy) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:CFIM_TimeIndepend_noise}, system, Measurement)
    F = CFIM_TimeIndepend(Measurement, system.freeHamiltonian, system.Hamiltonian_derivative, system.psi, system.decay_opt, 
                             system.γ, system.tspan, system.accuracy)
    return (abs(det(F)) < system.accuracy ? (1.0/system.accuracy) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:CFIM_TimeIndepend_noise}, system, Measurement, psi)
    F = CFIM_TimeIndepend(Measurement, system.freeHamiltonian, system.Hamiltonian_derivative, psi, system.decay_opt, 
                             system.γ, system.tspan, system.accuracy)
    return (abs(det(F)) < system.accuracy ? (1.0/system.accuracy) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:CFIM_noctrl}, system, Measurement)
    F = CFIM(Measurement, system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, system.tspan, system.accuracy)
    return  (abs(det(F)) < system.accuracy ? (1.0/system.accuracy) : real(tr(system.W*inv(F))))
end
