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

function CFI(M::Vector{Matrix{T}}, H0::Matrix{T}, ∂H_∂x::Matrix{T},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,  control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, times) where {T <: Complex,R <: Real}
    
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = (length(times)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]
    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = liouville_commu(∂H_∂x)

    Δt = times[2] - times[1]
    ρt = ρ_initial |> vec
    ∂ρt_∂x = ρt |> zero

    for t in 2:length(times)
        expL = evolute(H[t-1], Liouville_operator, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    ρt = exp( vec(H[end])' * zero(ρt) ) * ρt
    CFI(ρt|> vec2mat, ∂ρt_∂x|> vec2mat, M)
end

function CFIM(ρ, dρ, M)
    m_num = length(M)
    p_num = length(dρ)
    cfim = [kron(tr.(dρ.*[M[i]]),reshape(tr.(dρ.*[M[i]]), 1, p_num))/ tr(ρ*M[i])  for i in 1:m_num] |> sum .|>real
end

function CFIM(M::Vector{Matrix{T}}, H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,  control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, times) where {T <: Complex,R <: Real}
    para_num = length(∂H_∂x)
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = (length(times)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]
    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]

    Δt = times[2] - times[1]
    ρt = ρ_initial |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]

    for t in 2:length(times)
        expL = evolute(H[t-1], Liouville_operator, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp( vec(H[end])' * zero(ρt) ) * ρt
    CFIM(ρt|> vec2mat, ∂ρt_∂x|> vec2mat, M)
end

function SLD(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T <: Complex}
    2 * pinv(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ)) * vec(∂ρ_∂x) |> vec2mat
end

function SLD(ρ::Vector{T},∂ρ_∂x::Vector{T}) where {T <: Complex}
    SLD(ρ |> vec2mat, ∂ρ_∂x |> vec2mat)
end

function SLD(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}) where {T <: Complex}
    (x->SLD(ρ, x)).(∂ρ_∂x)
end

function SLD_qr(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T <: Complex}
    2 * (qr(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ), Val(true)) \ vec(∂ρ_∂x)) |> vec2mat
end

function SLD_ori(ρ::Matrix{T}, dρ::Vector{Matrix{T}}, rep="original", precision=1e-6) where {T <: Complex}
    dim = size(ρ)[1]
    para_num = length(dρ)
    SLD = [Matrix{ComplexF64}(undef, dim, dim) for i in 1:para_num]

    val, vec = eigen(ρ)
    for pi in 1:para_num
        SLD_eig = zeros(T, dim, dim)
        for fi in 1:dim
            for fj in 1:dim
                if abs(val[fi] + val[fj]) > precision
                    SLD_eig[fi,fj] = 2 * (vec[:,fi]' * dρ[pi] * vec[:,fj])/(val[fi] + val[fj])
                end
            end
        end
        SLD_eig[findall(SLD_eig == Inf)] .= 0.

        if rep=="original"
            SLD[pi] = vec*(SLD_eig*vec')
        else
            SLD[pi] = SLD_eig
        end
    end
    SLD
end

function SLD_ori(ρ::Matrix{T}, dρ::Matrix{T}, rep="original", precision=1e-6) where {T <: Complex}
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

# function SLD_ori(ρ::Matrix{T}, dρ::Vector{Matrix{T}}, rep="original", precision=1e-6) where {T <: Complex}
#     (x->SLD_ori(ρ, x, rep, precision)).(dρ)   
# end

function RLD(ρ::Matrix{T}, dρ::Matrix{T}) where {T <: Complex}
    dρ * pinv(ρ)
end

function QFI(ρ, dρ)
    SLD_tp = SLD(ρ, dρ)
    SLD2_tp = SLD_tp * SLD_tp
    F = tr(ρ * SLD2_tp)
    F |> real
end

function QFI_ori(ρ, dρ)
    SLD_tp = SLD_ori(ρ, dρ)
    SLD2_tp = SLD_tp * SLD_tp
    F = tr(ρ * SLD2_tp)
    F |> real
end

function QFI(H0::Matrix{T}, ∂H_∂x::Matrix{T},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, times) where {T <: Complex,R <: Real}

    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = (length(times)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = liouville_commu(∂H_∂x)

    Δt = times[2] - times[1]
    ρt = ρ_initial |> vec
    ∂ρt_∂x = ρt |> zero

    for t in 2:length(times)
        expL = evolute(H[t-1], Liouville_operator, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end
    ρt = exp( vec(H[end])' * zero(ρt) ) * ρt
    QFI(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
end

function QFI_ori(H0::Matrix{T}, ∂H_∂x::Matrix{T},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, times) where {T <: Complex,R <: Real}

    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = (length(times)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = liouville_commu(∂H_∂x)

    Δt = times[2] - times[1]
    ρt = ρ_initial |> vec
    ∂ρt_∂x = ρt |> zero

    for t in 2:length(times)
        expL = evolute(H[t-1], Liouville_operator, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = -im * Δt * ∂H_L * ρt + expL * ∂ρt_∂x
    end

    ρt = exp( vec(H[end])' * zero(ρt) ) * ρt
    QFI_ori(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
end

function QFI_RLD(ρ, dρ)
    p_num = length(dρ)
    RLD_tp = RLD(ρ, dρ)
    F = tr(ρ * RLD_tp * reshape(RLD_tp,1,p_num))
    F |> real
end

function QFIM(ρ, dρ)
    p_num = length(dρ)
    SLD_tp = SLD(ρ, dρ)
    qfim = ([0.5*ρ] .* (kron(SLD_tp, reshape(SLD_tp,1,p_num)) + kron(reshape(SLD_tp,1,p_num), SLD_tp))).|> tr .|>real 
end

function QFIM_ori(ρ, dρ)
    p_num = length(dρ)
    SLD_tp = SLD_ori(ρ, dρ)
    qfim = ([0.5*ρ] .* (kron(SLD_tp, reshape(SLD_tp,1,p_num)) + kron(reshape(SLD_tp,1,p_num), SLD_tp))).|> tr .|>real 
end

function QFIM(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, times) where {T <: Complex,R <: Real}
 
    para_num = length(∂H_∂x)
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = (length(times)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]
    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]

    Δt = times[2] - times[1]
    ρt = ρ_initial |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]

    for t in 2:length(times)
        expL = evolute(H[t-1], Liouville_operator, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end

    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    QFIM(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
end

function QFIM_ori(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, times) where {T <: Complex,R <: Real}

    para_num = length(∂H_∂x)
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = (length(times)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]

    Δt = times[2] - times[1]
    ρt = ρ_initial |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]

    for t in 2:length(times)
        expL = evolute(H[t-1], Liouville_operator, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end

    ρt = exp( vec(H[end])' * zero(ρt) ) * ρt
    QFIM_ori(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
end

function QFIM_ana(ρ::Array{T}, dρ::Vector{Matrix{T}}, rep="original") where {T <: Complex}
    para_num = length(dρ)
    QFIM_res = zeros(T, para_num, para_num)
    LD_tp = SLD_ana(ρ, dρ, rep)
    for para_i in 1:para_num
        for para_j in para_i:para_num
            SLD_ac = LD_tp[para_i]*LD_tp[para_j]+LD_tp[para_j]*LD_tp[para_i]
            QFIM_res[para_i,para_j] = 0.5*tr(ρ*SLD_ac)
            QFIM_res[para_j,para_i] = QFIM_res[para_i,para_j]
        end
    end
    QFIM_res |> real
end

function CFI(M, system)
    CFI(M,system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ_initial, system.Liouville_operator, system.γ, system.control_Hamiltonian, system.control_coefficients, system.times)
end

function CFIM(M, system)
    CFIM(M, system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ_initial, system.Liouville_operator, system.γ, system.control_Hamiltonian, system.control_coefficients, system.times)
end

function QFI(system)
    QFI(system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ_initial, system.Liouville_operator, system.γ, system.control_Hamiltonian, system.control_coefficients, system.times)
end

function QFIM(system)
    QFIM(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ_initial, system.Liouville_operator, system.γ, system.control_Hamiltonian, system.control_coefficients, system.times)
end

function QFI_ori(system)
    QFI_ori(system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ_initial, system.Liouville_operator, system.γ, system.control_Hamiltonian, system.control_coefficients, system.times)
end

function QFIM_ori(system)
    QFIM_ori(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ_initial, system.Liouville_operator, system.γ, system.control_Hamiltonian, system.control_coefficients, system.times)
end

function QFI_bfgs(system, control_coefficients)
    QFI_ori(system.freeHamiltonian, system.Hamiltonian_derivative[1], system.ρ_initial, system.Liouville_operator, system.γ, system.control_Hamiltonian, control_coefficients, system.times)
end

function QFIM_saveall(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, times) where {T <: Complex,R <: Real}
 
    para_num = length(∂H_∂x)
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = (length(times)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]
    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]

    Δt = times[2] - times[1]
    ρt = ρ_initial |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    F = [Matrix{Float64}(undef, length(∂H_∂x), length(∂H_∂x)) for i in 1:length(times)] 
    for t in 2:length(times)
        expL = evolute(H[t-1], Liouville_operator, γ, Δt, t)
        ρt =  expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
        F[t] = QFIM(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
    end
    return F
end