############## logarrithmic derivative ###############
function SLD(ρ::Matrix{T}, dρ::Matrix{T}, rep="original", precision=1e-8) where {T <: Complex}
    dim = size(ρ)[1]
    SLD = Matrix{ComplexF64}(undef, dim, dim)

    val, vec = eigen(ρ)
    SLD_eig = zeros(T, dim, dim)
    for fi in 1:dim
        for fj in 1:dim
            if abs(val[fi] + val[fj]) > precision
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

function SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}, rep="original", precision=1e-8) where {T <: Complex}
    dim = size(ρ)[1]
    para_num = length(dρ)
    SLD = [Matrix{ComplexF64}(undef, dim, dim) for i in 1:para_num]

    val, vec = eigen(ρ)
    for pj in 1:para_num
        SLD_eig = zeros(T, dim, dim)
        for fi in 1:dim
            for fj in 1:dim
                if abs(val[fi] + val[fj]) > precision
                    SLD_eig[fi,fj] = 2 * (vec[:,fi]' * dρ[pj] * vec[:,fj])/(val[fi] + val[fj])
                end
            end
        end
        SLD_eig[findall(SLD_eig == Inf)] .= 0.

        if rep=="original"
            SLD[pj] = vec*(SLD_eig*vec')
        else
            SLD[pj] = SLD_eig
        end
    end
    SLD
end

function SLD_auto(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T <: Complex}
    2 * pinv(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ)) * vec(∂ρ_∂x) |> vec2mat
end

function SLD_auto(ρ::Vector{T},∂ρ_∂x::Vector{T}) where {T <: Complex}
    SLD_auto(ρ |> vec2mat, ∂ρ_∂x |> vec2mat)
end

function SLD_auto(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}) where {T <: Complex}
    (x->SLD_auto(ρ, x)).(∂ρ_∂x)
end

function SLD_qr(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T <: Complex}
    2 * (qr(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ), Val(true)) \ vec(∂ρ_∂x)) |> vec2mat
end

function RLD(ρ::Matrix{T}, dρ::Matrix{T}) where {T <: Complex}
    pinv(ρ) * dρ
end

function LLD(ρ::Matrix{T}, dρ::Matrix{T}) where {T <: Complex}
    (dρ * pinv(ρ))'
end

#========================================================#
####################### calculate QFI ####################
function QFI(ρ, dρ)
    SLD_tp = SLD(ρ, dρ)
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

function QFIM_pure(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T <: Complex}
    SLD = 2*∂ρ_∂x
    SLD2_tp = SLD * SLD
    F = tr(ρ * SLD2_tp)
    F |> real
end

#### quantum dynamics and calcalate QFI ####
function QFI(H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, Decay_opt::Vector{Matrix{T}}, γ,control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T <: Complex,R <: Real}

    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = liouville_commu(∂H_∂x)

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero

    for t in 2:length(tspan)
        expL = evolute(H[t-1], Decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    ρt = exp( vec(H[end])' * zero(ρt) ) * ρt
    QFI(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
end

function QFI_auto(H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, Decay_opt::Vector{Matrix{T}}, γ,control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T <: Complex,R <: Real}

    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [transpose(repeat(control_coefficients[i], 1, ctrl_interval))[:] for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = liouville_commu(∂H_∂x)

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero

    for t in 2:length(tspan)
        expL = evolute(H[t-1], Decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    ρt = exp( vec(H[end])' * zero(ρt) ) * ρt
    QFI_auto(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
end

function QFIM_TimeIndepend(H0, ∂H_∂x::Matrix{T}, psi_initial::Vector{T}, tspan) where {T <: Complex,R <: Real}
    Δt = tspan[2] - tspan[1]
    U = exp(-im*H0*Δt)
    psi_t = psi_initial
    ∂psi_∂x = psi_initial |> zero 
    for t in 2:length(tspan)
        psi_t = U*psi_t
        ∂psi_∂x = -im*Δt*∂H_∂x*psi_t + U*∂psi_∂x
    end
    ρt = psi_t*psi_t'
    ∂ρt_∂x = ∂psi_∂x*psi_t'+psi_t*∂psi_∂x'
    QFIM_pure(ρt, ∂ρt_∂x)
end

function QFIM_TimeIndepend(H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, Decay_opt::Vector{Matrix{T}}, γ, tspan) where {T <: Complex,R <: Real}
    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero 
    expL = evolute(H0, Decay_opt, γ, Δt, 1)
    ∂H_L = liouville_commu(∂H_∂x)
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL*∂ρt_∂x
    end
    QFI(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
end

function QFIM_TimeIndepend_AD(H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, Decay_opt::Vector{Matrix{T}}, γ, tspan) where {T <: Complex,R <: Real}
    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero 
    expL = evolute(H0, Decay_opt, γ, Δt, 1)
    ∂H_L = liouville_commu(∂H_∂x)
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL*∂ρt_∂x
    end
    QFI_auto(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
end

function QFI(system)
    QFI(system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ0, system.Decay_opt, system.γ, system.control_Hamiltonian, system.control_coefficients, system.tspan)
end

function QFI_auto(system)
    QFI_auto(system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ0, system.Decay_opt, system.γ, system.control_Hamiltonian, system.control_coefficients, system.tspan)
end

function QFI_bfgs(system, control_coefficients)
    QFI(system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ0, system.Decay_opt, system.γ, system.control_Hamiltonian, control_coefficients, system.tspan)
end


#==========================================================#
####################### calculate QFIM #####################
function QFIM(ρ, dρ)
    p_num = length(dρ)
    SLD_tp = SLD(ρ, dρ)
    qfim = ([0.5*ρ] .* (kron(SLD_tp, reshape(SLD_tp,1,p_num)) + kron(reshape(SLD_tp,1,p_num), SLD_tp))).|> tr .|>real 
end

function QFIM_auto(ρ, dρ)
    p_num = length(dρ)
    SLD_tp = SLD_auto(ρ, dρ)
    qfim = ([0.5*ρ] .* (kron(SLD_tp, reshape(SLD_tp,1,p_num)) + kron(reshape(SLD_tp,1,p_num), SLD_tp))).|> tr .|>real 
end

function QFIM_pure(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}) where {T <: Complex}
    p_num = length(∂ρ_∂x)
    SLD = [2*∂ρ_∂x[i] for i in 1:p_num]
    qfim = ([0.5*ρ] .* (kron(SLD, reshape(SLD,1,p_num)) + kron(reshape(SLD,1,p_num), SLD))).|> tr .|>real
end

#### quantum dynamics and calcalate QFIM ####
function QFIM(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, Decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T <: Complex,R <: Real}

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
        expL = evolute(H[t-1], Decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    QFIM(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
end

function QFIM_auto(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, Decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T <: Complex,R <: Real}
 
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
        expL = evolute(H[t-1], Decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end

    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    QFIM_auto(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
end

function QFIM_TimeIndepend(H0, ∂H_∂x::Vector{Matrix{T}}, psi_initial::Vector{T}, tspan) where {T <: Complex,R <: Real}
    Δt = tspan[2] - tspan[1]
    para_num = length(∂H_∂x)
    U = exp(-im*H0*Δt)
    psi_t = psi_initial
    ∂psi_∂x = [psi_initial |> zero for i in 1:para_num]
    for t in 2:length(tspan)
        psi_t = U*psi_t
        ∂psi_∂x = [-im*Δt*∂H_∂x[i]*psi_t for i in 1:para_num] + [U].*∂psi_∂x
    end
    ρt = psi_t*psi_t'
    ∂ρt_∂x = [(∂psi_∂x[i]*psi_t'+psi_t*∂psi_∂x[i]') for i in 1:para_num]
    QFIM_pure(ρt, ∂ρt_∂x)
end

function QFIM_TimeIndepend(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, Decay_opt::Vector{Matrix{T}}, γ, tspan) where {T <: Complex,R <: Real}
    Δt = tspan[2] - tspan[1]
    para_num = length(∂H_∂x)
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    expL = evolute(H0, Decay_opt, γ, Δt, 1)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]
    for t in 2:length(tspan)
        ρt = expL*ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL].*∂ρt_∂x
    end
    QFIM(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
end

function QFIM_TimeIndepend_AD(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, Decay_opt::Vector{Matrix{T}}, γ, tspan) where {T <: Complex,R <: Real}
    Δt = tspan[2] - tspan[1]
    para_num = length(∂H_∂x)
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    expL = evolute(H0, Decay_opt, γ, Δt, 1)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]
    for t in 2:length(tspan)
        ρt = expL*ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL].*∂ρt_∂x
    end
    QFIM_auto(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
end

function QFIM_saveall(H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, Decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T <: Complex,R <: Real}
 
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
        expL = evolute(H[t-1], Decay_opt, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
        F[t] = QFIM(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
    end
    return F
end

function QFIM(system)
    QFIM(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.Decay_opt, system.γ, system.control_Hamiltonian, system.control_coefficients, system.tspan)
end

function QFIM_auto(system)
    QFIM_auto(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.Decay_opt, system.γ, system.control_Hamiltonian, system.control_coefficients, system.tspan)
end

function QFIM_saveall(system)
    QFIM_saveall(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.Decay_opt, system.γ, system.control_Hamiltonian, system.control_coefficients, system.tspan)
end

#==========================================================#
####################### calculate CFI ######################
function CFI(ρ, dρ, M)
    m_num = length(M)
    p = zero(ComplexF64)
    dp = zero(ComplexF64)
    F = 0.
    for i in 1:m_num
        mp = M[i]
        p = tr(ρ * mp)
        dp = tr(dρ * mp)
        cadd = 0.
        if p != 0
            cadd = (dp^2) / p
        end
        F += cadd
    end 
    real(F)
end

#### quantum dynamics and calcalate CFI ####
function CFI(M::Vector{Matrix{T}}, H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, Decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T <: Complex,R <: Real}
    
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]
    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = liouville_commu(∂H_∂x)

    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero
    for t in 2:length(tspan)
        expL = evolute(H[t-1], Decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    ρt = exp( vec(H[end])' * zero(ρt) ) * ρt
    CFI(ρt|> vec2mat, ∂ρt_∂x|> vec2mat, M)
end

function CFIM_TimeIndepend(M, H0, ∂H_∂x::Matrix{T}, psi_initial::Vector{T}, tspan) where {T <: Complex,R <: Real}
    Δt = tspan[2] - tspan[1]
    U = exp(-im*H0*Δt)
    psi_t = psi_initial
    ∂psi_∂x = psi_initial |> zero 
    for t in 2:length(tspan)
        psi_t = U*psi_t
        ∂psi_∂x = -im*Δt*∂H_∂x*psi_t + U*∂psi_∂x 
    end
    ρt = psi_t*psi_t'
    ∂ρt_∂x = ∂psi_∂x*psi_t'+psi_t*∂psi_∂x'
    CFI(ρt, ∂ρt_∂x, M)
end

function CFIM_TimeIndepend(M, H0, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, Decay_opt::Vector{Matrix{T}}, γ, tspan) where {T <: Complex,R <: Real}
    Δt = tspan[2] - tspan[1]
    ρt = ρ0 |> vec
    ∂ρt_∂x = ρt |> zero 
    expL = evolute(H0, Decay_opt, γ, Δt, 1)
    ∂H_L = liouville_commu(∂H_∂x)
    for t in 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL*∂ρt_∂x
    end
    CFI(ρt|> vec2mat, ∂ρt_∂x|> vec2mat, M)
end

function CFI(M, system)
    CFI(M,system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ0, system.Decay_opt, system.γ, system.control_Hamiltonian, system.control_coefficients, system.tspan)
end

#======================================================#
#################### calculate CFIM ####################
function CFIM(ρ, dρ, M)
    m_num = length(M)
    p_num = length(dρ)
    cfim = [kron(tr.(dρ.*[M[i]]),reshape(tr.(dρ.*[M[i]]), 1, p_num))/ tr(ρ*M[i]) for i in 1:m_num] |> sum .|>real
end

#### quantum dynamics and calcalate CFIM ####
function CFIM(M::Vector{Matrix{T}}, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, Decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T <: Complex,R <: Real}
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
        expL = evolute(H[t-1], Decay_opt, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp( vec(H[end])' * zero(ρt) ) * ρt
    CFIM(ρt|> vec2mat, ∂ρt_∂x|> vec2mat, M)
end

function CFIM_TimeIndepend(M, H0, ∂H_∂x::Vector{Matrix{T}}, psi_initial::Vector{T}, tspan) where {T <: Complex,R <: Real}
    Δt = tspan[2] - tspan[1]
    para_num = length(∂H_∂x)
    U = exp(-im*H0*Δt)
    psi_t = psi_initial
    ∂psi_∂x = [psi_initial |> zero for i in 1:para_num]
    F = [Matrix{Float64}(undef, length(∂H_∂x), length(∂H_∂x)) for i in 1:length(tspan)] 
    for t in 2:length(tspan)
        psi_t = U*psi_t
        ∂psi_∂x = [-im*Δt*∂H_∂x[i]*psi_t for i in 1:para_num] + [U].*∂psi_∂x 
    end
    ρt = psi_t*psi_t'
    ∂ρt_∂x = [(∂psi_∂x[i]*psi_t'+psi_t*∂psi_∂x[i]') for i in 1:para_num]
    CFIM(ρt, ∂ρt_∂x, M)
end

function CFIM_TimeIndepend(M, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, Decay_opt::Vector{Matrix{T}}, γ, tspan) where {T <: Complex,R <: Real}
    Δt = tspan[2] - tspan[1]
    para_num = length(∂H_∂x)
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    expL = evolute(H0, Decay_opt, γ, Δt, 1)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]
    F = [Matrix{Float64}(undef, para_num, para_num) for i in 1:length(tspan)] 
    for t in 2:length(tspan)
        ρt = expL*ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL].*∂ρt_∂x
    end
    CFIM(ρt|> vec2mat, ∂ρt_∂x|> vec2mat, M)
end

function CFIM_saveall(M, H0, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, Decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T <: Complex,R <: Real}
 
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
        expL = evolute(H[t-1], Decay_opt, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
        F[t] = CFIM(ρt|> vec2mat, ∂ρt_∂x|> vec2mat, M)
    end
    return F
end

function CFIM(M, system)
    CFIM(M, system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.Decay_opt, system.γ, system.control_Hamiltonian, system.control_coefficients, system.tspan)
end
