############## logarrithmic derivative ###############
function SLD(ρ::Matrix{T}, dρ::Matrix{T}; rep="original", accuracy=1e-8) where {T<:Complex}
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
    else
        SLD = SLD_eig
    end
    SLD
end

function SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep="original", accuracy=1e-8) where {T<:Complex}
    (x->SLD(ρ, x, rep=rep, accuracy=accuracy)).(dρ)
end

function SLD_auto(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T<:Complex}
    2 * pinv(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ)) * vec(∂ρ_∂x) |> vec2mat
end

function SLD_auto(ρ::Vector{T},∂ρ_∂x::Vector{T}) where {T<:Complex}
    SLD_auto(ρ |> vec2mat, ∂ρ_∂x |> vec2mat)
end

function SLD_auto(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}) where {T<:Complex}
    (x->SLD_auto(ρ, x)).(∂ρ_∂x)
end

function SLD_qr(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T<:Complex}
    2 * (qr(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ), Val(true)) \ vec(∂ρ_∂x)) |> vec2mat
end

function RLD(ρ::Matrix{T}, dρ::Matrix{T}) where {T<:Complex}
    pinv(ρ) * dρ
end

function LLD(ρ::Matrix{T}, dρ::Matrix{T}) where {T<:Complex}
    (dρ * pinv(ρ))'
end

#========================================================#
####################### calculate QFI ####################
function QFI(ρ, dρ, accuracy)
    SLD_tp = SLD(ρ, dρ, accuracy=accuracy)
    SLD2_tp = SLD_tp * SLD_tp
    F = tr(ρ * SLD2_tp)
    F |> real
end

function QFI_auto(ρ, dρ)
    SLD_tp = SLD_auto(ρ, dρ)
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

    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = liouville_commu(∂H_∂x)

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero

    for t in 2:length(tspan)
        expL = evolute(H[t-1], decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    ρt = exp( vec(H[end])' * zero(ρt) ) * ρt
    QFI(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, accuracy)
end

function QFI_auto(H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ,control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T<:Complex,R<:Real}

    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [transpose(repeat(control_coefficients[i], 1, ctrl_interval))[:] for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = liouville_commu(∂H_∂x)

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero

    for t in 2:length(tspan)
        expL = evolute(H[t-1], decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    QFI_auto(ρt |> vec2mat, ∂ρt_∂x |> vec2mat)
end

function QFIM_TimeIndepend(H0::Matrix{T}, ∂H_∂x::Matrix{T}, psi_initial::Vector{T}, tspan, accuracy) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    U = exp(-im * H0 * Δt)
    psi_t = psi_initial
    ∂psi_∂x = psi_initial |> zero 
    for t in 2:length(tspan)
        psi_t = U * psi_t
        ∂psi_∂x = -im * Δt * ∂H_∂x * psi_t + U * ∂psi_∂x
    end
    ρt = psi_t*psi_t'
    ∂ρt_∂x = ∂psi_∂x*psi_t'+psi_t*∂psi_∂x'
    QFIM_pure(ρt, ∂ρt_∂x)
end

function QFIM_TimeIndepend(H0::Matrix{T}, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero 
    expL = evolute(H0, decay_opt, γ, Δt, 1)
    ∂H_L = liouville_commu(∂H_∂x)
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    QFI(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, accuracy)
end

function QFIM_TimeIndepend_AD(H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero 
    expL = evolute(H0, decay_opt, γ, Δt, 1)
    ∂H_L = liouville_commu(∂H_∂x)
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    QFI_auto(ρt |> vec2mat, ∂ρt_∂x |> vec2mat)
end

function QFI(system)
    QFI(system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ0, system.decay_opt, system.γ, system.control_Hamiltonian, system.control_coefficients, system.tspan, system.accuracy)
end

function QFI_auto(system)
    QFI_auto(system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ0, system.decay_opt, system.γ, system.control_Hamiltonian, system.control_coefficients, system.tspan)
end


#==========================================================#
####################### calculate QFIM #####################
function QFIM(ρ, dρ, accuracy)
    p_num = length(dρ)
    SLD_tp = SLD(ρ, dρ, accuracy=accuracy)
    qfim = ([0.5*ρ] .* (kron(SLD_tp, reshape(SLD_tp,1,p_num)) + kron(reshape(SLD_tp,1,p_num), SLD_tp))).|> tr .|>real 
end

function QFIM_auto(ρ, dρ)
    p_num = length(dρ)
    SLD_tp = SLD_auto(ρ, dρ)
    qfim = ([0.5*ρ] .* (kron(SLD_tp, reshape(SLD_tp,1,p_num)) + kron(reshape(SLD_tp,1,p_num), SLD_tp))).|> tr .|>real 
end

function QFIM_pure(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}) where {T<:Complex}
    p_num = length(∂ρ_∂x)
    SLD = [2*∂ρ_∂x[i] for i in 1:p_num]
    qfim = ([0.5*ρ] .* (kron(SLD, reshape(SLD,1,p_num)) + kron(reshape(SLD,1,p_num), SLD))) .|> tr .|>real
end


#### quantum dynamics and calcalate QFIM ####

function QFIM(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, accuracy) where {T<:Complex,R<:Real}

    para_num = length(∂H_∂x)
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    for t in 2:length(tspan)
        expL = evolute(H[t-1], decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    QFIM(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, accuracy)
end

function QFIM(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
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
    QFIM(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, accuracy)
end

function QFIM(H0::Vector{Matrix{T}}, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    para_num = length(∂H_∂x)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    for t in 2:length(tspan)
        expL = evolute(H0[t-1], decay_opt, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp( vec(H0[end])' * zero(ρt) ) * ρt
    QFIM(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, accuracy)
end

function QFIM_auto(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T<:Complex,R<:Real}
 
    para_num = length(∂H_∂x)
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [transpose(repeat(control_coefficients[i], 1, ctrl_interval))[:] for i in 1:ctrl_num]
    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]

    for t in 2:length(tspan)
        expL = evolute(H[t-1], decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end

    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    QFIM_auto(ρt |> vec2mat, ∂ρt_∂x |> vec2mat)
end

function QFIM_TimeIndepend(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, psi_initial::Vector{T}, tspan, accuracy) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    para_num = length(∂H_∂x)
    U = exp(-im * H0 * Δt)
    psi_t = psi_initial
    ∂psi_∂x = [psi_initial |> zero for i in 1:para_num]
    for t in 2:length(tspan)
        psi_t = U * psi_t
        ∂psi_∂x = [-im * Δt * ∂H_∂x[i] * psi_t for i in 1:para_num] + [U] .* ∂psi_∂x
    end
    ρt = psi_t * psi_t'
    ∂ρt_∂x = [(∂psi_∂x[i] * psi_t' + psi_t * ∂psi_∂x[i]') for i in 1:para_num]
    QFIM_pure(ρt, ∂ρt_∂x)
end

function QFIM_TimeIndepend(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    para_num = length(∂H_∂x)
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    expL = evolute(H0, decay_opt, γ, Δt, 1)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    QFIM(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, accuracy)
end

function QFIM_TimeIndepend(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, psi::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    para_num = length(∂H_∂x)
    ρ0 = psi*psi'
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    expL = evolute(H0, decay_opt, γ, Δt, 1)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    QFIM(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, accuracy)
end

function QFIM_TimeIndepend_AD(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    para_num = length(∂H_∂x)
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    expL = evolute(H0, decay_opt, γ, Δt, 1)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    QFIM_auto(ρt |> vec2mat, ∂ρt_∂x |> vec2mat)
end

function QFIM_saveall(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, accuracy) where {T<:Complex,R<:Real}
 
    para_num = length(∂H_∂x)
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [transpose(repeat(control_coefficients[i], 1, ctrl_interval))[:] for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    F = [Matrix{Float64}(undef, length(∂H_∂x), length(∂H_∂x)) for i in 1:length(tspan)] 
    for t in 2:length(tspan)
        expL = evolute(H[t-1], decay_opt, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
        F[t] = QFIM(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, accuracy)
    end
    return F
end

function obj_func(x::Val{:QFIM}, system, Measurement)
    return QFIM(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, system.control_Hamiltonian, 
                system.control_coefficients, system.tspan, system.accuracy)
end

function obj_func(x::Val{:QFIM}, system, Measurement, control_coeff)
    return QFIM(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, system.control_Hamiltonian, 
                control_coeff, system.tspan, system.accuracy)
end

function obj_func(x::Val{:QFIM_TimeIndepend_noiseless}, system, Measurement)
    return QFIM_TimeIndepend(system.freeHamiltonian, system.Hamiltonian_derivative, system.psi, system.tspan, system.accuracy)
end

function obj_func(x::Val{:QFIM_TimeIndepend_noiseless}, system, Measurement, psi)
    return QFIM_TimeIndepend(system.freeHamiltonian, system.Hamiltonian_derivative, psi, system.tspan, system.accuracy)
end

function obj_func(x::Val{:QFIM_TimeIndepend_noise}, system, Measurement)
    return QFIM_TimeIndepend(system.freeHamiltonian, system.Hamiltonian_derivative, system.psi, system.decay_opt, 
                             system.γ, system.tspan, system.accuracy)
end

function obj_func(x::Val{:QFIM_TimeIndepend_noise}, system, Measurement, psi)
    return QFIM_TimeIndepend(system.freeHamiltonian, system.Hamiltonian_derivative, psi, system.decay_opt, 
                             system.γ, system.tspan, system.accuracy)
end

function obj_func(x::Val{:QFIM_noctrl}, system, Measurement)
    return QFIM(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, system.tspan, system.accuracy)
end

function QFIM_auto(system)
    QFIM_auto(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, 
              system.control_Hamiltonian, system.control_coefficients, system.tspan)
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
    
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]
    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = liouville_commu(∂H_∂x)

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero
    for t in 2:length(tspan)
        expL = evolute(H[t-1], decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    ρt = exp( vec(H[end])' * zero(ρt) ) * ρt
    CFI(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, Measurement, accuracy)
end

function CFI(Measurement::Vector{Matrix{T}}, H0::Matrix{T}, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
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

function CFI_AD(Mbasis::Vector{Vector{T}}, Mcoeff::Vector{R}, Lambda, H0::Matrix{T}, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    dim = size(ρ0)[1]
    U = Matrix{ComplexF64}(I,dim,dim)
    for i in 1:length(Lambda)
        U = U*exp(1.0im*Mcoeff[i]*Lambda[i])
    end
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


function CFI(Measurement::Vector{Matrix{T}}, H0::Vector{Matrix{T}}, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    ∂H_L = liouville_commu(∂H_∂x)

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero
    for t in 2:length(tspan)
        expL = evolute(H0[t-1], decay_opt, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    ρt = exp( vec(H0[end])' * zero(ρt) ) * ρt
    CFI(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, Measurement, accuracy)
end

function CFIM_TimeIndepend(Measurement, H0::Matrix{T}, ∂H_∂x::Matrix{T}, psi_initial::Vector{T}, tspan, accuracy) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    U = exp(-im * H0 * Δt)
    psi_t = psi_initial
    ∂psi_∂x = psi_initial |> zero 
    for t in 2:length(tspan)
        psi_t = U * psi_t
        ∂psi_∂x = -im * Δt * ∂H_∂x * psi_t + U * ∂psi_∂x 
    end
    ρt = psi_t * psi_t'
    ∂ρt_∂x = ∂psi_∂x * psi_t' + psi_t * ∂psi_∂x'
    CFI(ρt, ∂ρt_∂x, Measurement, accuracy)
end

function CFIM_TimeIndepend(Measurement, H0::Matrix{T}, ∂H_∂x::Matrix{T}, psi::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    ρ0 = psi*psi'
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero 
    expL = evolute(H0, decay_opt, γ, Δt, 1)
    ∂H_L = liouville_commu(∂H_∂x)
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    CFI(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, Measurement, accuracy)
end

function CFIM_TimeIndepend(Measurement, H0::Matrix{T}, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero 
    expL = evolute(H0, decay_opt, γ, Δt, 1)
    ∂H_L = liouville_commu(∂H_∂x)
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    CFI(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, Measurement, accuracy)
end

function CFI(Measurement, system)
    CFI(Measurement, system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ0, system.decay_opt, system.γ, 
        system.control_Hamiltonian, system.control_coefficients, system.tspan, system.accuracy)
end


#======================================================#
#################### calculate CFIM ####################
function CFIM(ρ, dρ)
    data = readdlm("$(Main.pkgpath)/sic_fiducial_vectors/d$(size(ρ)[1]).txt", '\t', Float64, '\n')
    fiducial = data[:,1]+1.0im*data[:,2]
    M = sic_povm(fiducial)
    m_num = length(M)
    p_num = length(dρ)
    cfim = [kron(tr.(dρ.*[M[i]]),reshape(tr.(dρ.*[M[i]]), 1, p_num))/ tr(ρ*M[i]) for i in 1:m_num] |> sum .|>real
end

function CFIM(ρ, dρ, M)
    m_num = length(M)
    p_num = length(dρ)
    cfim = [kron(tr.(dρ.*[M[i]]),reshape(tr.(dρ.*[M[i]]), 1, p_num))/ tr(ρ*M[i]) for i in 1:m_num] |> sum .|>real
end


#### quantum dynamics and calcalate CFIM ####
function CFIM(Measurement::Vector{Matrix{T}}, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, accuracy) where {T<:Complex,R<:Real}
    para_num = length(∂H_∂x)
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [transpose(repeat(control_coefficients[i], 1, ctrl_interval))[:] for i in 1:ctrl_num]
    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    for t in 2:length(tspan)
        expL = evolute(H[t-1], decay_opt, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp( vec(H[end])' * zero(ρt) ) * ρt
    CFIM(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, Measurement)
end

function CFIM(Measurement::Vector{Matrix{T}}, H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
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
    CFIM(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, Measurement)
end

function CFIM_AD(Mbasis::Vector{Vector{T}}, Mcoeff::Vector{R}, Lambda, H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    dim = size(ρ0)[1]
    U = Matrix{ComplexF64}(I,dim,dim)
    for i in 1:length(Lambda)
        U = U*exp(1.0im*Mcoeff[i]*Lambda[i])
    end
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
    CFIM(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, Measurement)
end

function CFIM(Measurement::Vector{Matrix{T}}, H0::Vector{Matrix{T}}, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    para_num = length(∂H_∂x)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    for t in 2:length(tspan)
        expL = evolute(H0[t-1], decay_opt, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp( vec(H0[end])' * zero(ρt) ) * ρt
    CFIM(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, Measurement)
end

function CFIM_TimeIndepend(Measurement, H0, ∂H_∂x::Vector{Matrix{T}}, psi_initial::Vector{T}, tspan, accuracy) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    para_num = length(∂H_∂x)
    U = exp(-im * H0 * Δt)
    psi_t = psi_initial
    ∂psi_∂x = [psi_initial |> zero for i in 1:para_num]
    F = [Matrix{Float64}(undef, length(∂H_∂x), length(∂H_∂x)) for i in 1:length(tspan)] 
    for t in 2:length(tspan)
        psi_t = U * psi_t
        ∂psi_∂x = [-im * Δt * ∂H_∂x[i] * psi_t for i in 1:para_num] + [U] .* ∂psi_∂x 
    end
    ρt = psi_t * psi_t'
    ∂ρt_∂x = [(∂psi_∂x[i] * psi_t' + psi_t * ∂psi_∂x[i]') for i in 1:para_num]
    CFIM(ρt, ∂ρt_∂x, Measurement)
end

function CFIM_TimeIndepend(Measurement, H0, ∂H_∂x::Vector{Matrix{T}}, psi::Vector{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
    Δt = tspan[2] - tspan[1]
    para_num = length(∂H_∂x)
    ρ0 = psi*psi'
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    expL = evolute(H0, decay_opt, γ, Δt, 1)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]
    F = [Matrix{Float64}(undef, para_num, para_num) for i in 1:length(tspan)] 
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    CFIM(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, Measurement)
end

function CFIM_TimeIndepend(Measurement, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, tspan, accuracy) where {T<:Complex,R<:Real}
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
    CFIM(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, Measurement)
end

function CFIM_saveall(Measurement, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan, accuracy) where {T<:Complex,R<:Real}
 
    para_num = length(∂H_∂x)
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [transpose(repeat(control_coefficients[i], 1, ctrl_interval))[:] for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    F = [Matrix{Float64}(undef, length(∂H_∂x), length(∂H_∂x)) for i in 1:length(tspan)] 
    for t in 2:length(tspan)
        expL = evolute(H[t-1], decay_opt, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
        F[t] = CFIM(ρt |> vec2mat, ∂ρt_∂x |> vec2mat, Measurement)
    end
    return F
end

function obj_func(x::Val{:CFIM}, system, Measurement)
    return CFIM(Measurement, system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, 
                system.control_Hamiltonian, system.control_coefficients, system.tspan, system.accuracy)
end

function obj_func(x::Val{:CFIM}, system, Measurement, control_coeff)
    return CFIM(Measurement, system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ,  
                system.control_Hamiltonian, control_coeff, system.tspan, system.accuracy)
end

function obj_func(x::Val{:CFIM_TimeIndepend_noiseless}, system, Measurement)
    return CFIM_TimeIndepend(Measurement, system.freeHamiltonian, system.Hamiltonian_derivative, system.psi, system.tspan, system.accuracy)
end

function obj_func(x::Val{:CFIM_TimeIndepend_noiseless}, system, Measurement, psi)
    return CFIM_TimeIndepend(Measurement, system.freeHamiltonian, system.Hamiltonian_derivative, psi, system.tspan, system.accuracy)
end

function obj_func(x::Val{:CFIM_TimeIndepend_noise}, system, Measurement)
    return CFIM_TimeIndepend(Measurement, system.freeHamiltonian, system.Hamiltonian_derivative, system.psi, system.decay_opt, 
                             system.γ, system.tspan, system.accuracy)
end

function obj_func(x::Val{:CFIM_TimeIndepend_noise}, system, Measurement, psi)
    return CFIM_TimeIndepend(Measurement, system.freeHamiltonian, system.Hamiltonian_derivative, psi, system.decay_opt, 
                             system.γ, system.tspan, system.accuracy)
end

function obj_func(x::Val{:CFIM_noctrl}, system, Measurement)
    return CFIM(Measurement, system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, 
                system.tspan, system.accuracy)
end
