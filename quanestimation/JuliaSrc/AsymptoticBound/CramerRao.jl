function CFI(ρ, dρ, M)
    m_num = length(M)
    p = zero(ComplexF64)
    dp = zero(ComplexF64)
    F = 0.
    for i in 1:m_num
        mp = M[i]
        p += tr(ρ * mp)
        dp = tr(dρ * mp)
        cadd = 0.
        if p != 0
            cadd = (dp^2) / p
        end
        F += cadd
    end 
    real(F)
end

function CFI(M::Vector{Matrix{T}}, H::Vector{Matrix{T}}, ∂H_∂x::Matrix{T},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,  times) where {T <: Complex,R <: Real}
    dim = size(H[1])[1]
    Δt = times[2] - times[1]
    ρt = evolute(H[1], Liouville_operator, γ, Δt, 1) * (ρ_initial |> vec)
    ∂ρt_∂x = -im * Δt * liouville_commu(∂H_∂x) * ρt
    for t in 2:length(times)
        expL = evolute(H[t], Liouville_operator, γ, Δt, t)
        ρt=  expL * ρt
        ∂ρt_∂x= -im * Δt * liouville_commu(∂H_∂x) * ρt + expL * ∂ρt_∂x
    end
    CFI(ρt|> vec2mat, ∂ρt_∂x|> vec2mat, M)
end

function CFIM(ρ, dρ, M)
    m_num = length(M)
    cfim = [tr.(kron(dρ', dρ).*M[i]) / tr(ρ*M[i])  for i in 1:m_num] |> sum
end

function CFIM(M::Vector{Matrix{T}}, H::Vector{Matrix{T}}, ∂H_∂x::Matrix{T},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,  times) where {T <: Complex,R <: Real}
    dim = size(H[1])[1]
    Δt = times[2] - times[1]
    para_num = length(∂H_∂x)
    ρt = evolute(H[1], Liouville_operator, γ, Δt, 1) * ρ_initial[:]
    ∂ρt_∂x = [-im * Δt * liouville_commu(∂H_∂x[i]) * ρt for i in 1:para_num]
    for t in 2:length(times)
        expL = evolute(H[t], Liouville_operator, γ, Δt, t)
        ρt=  expL * ρt
        ∂ρt_∂x= [-im * Δt * liouville_commu(∂H_∂x[i]) * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
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

function SLD_ana(ρ::Array{T}, dρ::Vector{Matrix{T}}, rep="original", precision=1e-6) where {T <: Complex}
    dim = size(ρ)[1]
    para_num = length(dρ)
    SLD = [Matrix{ComplexF64}(undef, dim, dim) for i in 1:para_num]

    val, vec = eigen(ρ)
    for para_i in 1:para_num
        SLD_eig = zeros(T, dim, dim)
        for fi in 1:dim
            for fj in 1:dim
                if abs(val[fi] + val[fj]) > precision
                    SLD_eig[fi,fj] = 2*(vec[:,fi]' *dρ[para_i] * vec[:,fj])/(val[fi] + val[fj])
                end
            end
        end
        SLD_eig[findall(SLD_eig == Inf)] .= 0.

        if rep=="original"
            SLD[para_i] = vec*(SLD_eig*vec')
        else
            SLD[para_i] = SLD_eig
        end
    end
    SLD
end

function RLD(ρ::Matrix{T}, dρ::Matrix{T}) where {T <: Complex}
    dρ * pinv(ρ)
end

function QFI(ρ, dρ)
    SLD_tp = SLD(ρ, dρ)
    SLD2_tp = SLD_tp * SLD_tp
    F = tr(ρ * SLD2_tp)
    F |> real
end

function QFI(H::Vector{Matrix{T}}, ∂H_∂x::Matrix{T},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,  times) where {T <: Complex,R <: Real}
    dim = size(H[1])[1]
    Δt = times[2] - times[1]
    ρt = ρ_initial |> vec
    ∂ρt_∂x = ρt |> zero
    for t in 2:length(times)
        expL = evolute(H[t-1], Liouville_operator, γ, Δt, t)
        ρt=  expL * ρt
        ∂ρt_∂x= -im * Δt * liouville_commu(∂H_∂x) * ρt + expL * ∂ρt_∂x
    end
    QFI(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
end

function QFI_RLD(ρ, dρ)
    RLD_tp = RLD(ρ, dρ)
    F = tr(ρ * RLD_tp * RLD_tp')
    F |> real
end

function QFIM(ρ, dρ)
    SLD_tp = SLD(ρ, dρ)
    [0.5*ρ] .* (kron(SLD_tp, SLD_tp') + kron(SLD_tp', SLD_tp)).|> tr .|>real 
end

function QFIM(H::Vector{Matrix{T}}, ∂H_∂x::Vector{Matrix{T}},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,  times) where {T <: Complex,R <: Real}
    dim = size(H[1])[1]
    Δt = times[2] - times[1]
    para_num = length(∂H_∂x)
    ρt = evolute(H[1], Liouville_operator, γ, Δt, 1) * ρ_initial[:]
    ∂ρt_∂x = [-im * Δt * liouville_commu(∂H_∂x[i]) * ρt for i in 1:para_num]
    for t in 2:length(times)
        expL = evolute(H[t], Liouville_operator, γ, Δt, t)
        ρt=  expL * ρt
        ∂ρt_∂x= [-im * Δt * liouville_commu(∂H_∂x[i]) * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    QFIM(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
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
    CFI(M,Htot(system.freeHamiltonian, system.control_Hamiltonian, system.control_coefficients), system.Hamiltonian_derivative[1], system.ρ_initial, system.Liouville_operator, system.γ, system.times)
end

function CFIM(M, system)
    CFIM(M,Htot(system.freeHamiltonian, system.control_Hamiltonian, system.control_coefficients), system.Hamiltonian_derivative, system.ρ_initial, system.Liouville_operator, system.γ, system.times)
end

function QFI(system)
    QFI(Htot(system.freeHamiltonian, system.control_Hamiltonian, system.control_coefficients), system.Hamiltonian_derivative[1], system.ρ_initial, system.Liouville_operator, system.γ, system.times)
end

function QFIM(system)
    QFIM(Htot(system.freeHamiltonian, system.control_Hamiltonian, system.control_coefficients), system.Hamiltonian_derivative, system.ρ_initial, system.Liouville_operator, system.γ, system.times)
end
