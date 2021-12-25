abstract type ControlSystem end
mutable struct GRAPE_Copt{T<:Complex,M <: Real} <: ControlSystem
    freeHamiltonian
    Hamiltonian_derivative::Vector{Matrix{T}}
    ρ0::Matrix{T}
    tspan::Vector{M}
    decay_opt::Vector{Matrix{T}}
    γ::Vector{M}
    control_Hamiltonian::Vector{Matrix{T}}
    control_coefficients::Vector{Vector{M}}
    ctrl_bound::Vector{M}
    W::Matrix{M}
    mt::M
    vt::M
    ϵ::M
    beta1::M
    beta2::M
    accuracy::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    GRAPE_Copt(freeHamiltonian, Hamiltonian_derivative::Vector{Matrix{T}}, ρ0::Matrix{T}, tspan::Vector{M}, decay_opt::Vector{Matrix{T}}, 
             γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{M}}, ctrl_bound::Vector{M}, 
             W::Matrix{M}, mt::M, vt::M, ϵ::M, beta1::M, beta2::M, accuracy::M, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1), ∂ρ_∂V=Vector{Vector{Matrix{T}}}(undef, 1)) where {T<:Complex,M <: Real}= 
             new{T,M}(freeHamiltonian, Hamiltonian_derivative, ρ0, tspan, decay_opt, γ, control_Hamiltonian, control_coefficients,
                      ctrl_bound, W, mt, vt, ϵ, beta1, beta2, accuracy, ρ, ∂ρ_∂x) 
end

function gradient_QFI!(grape::GRAPE_Copt{T}) where {T<:Complex}
    δF = gradient(x->QFI_auto(grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan), grape.control_coefficients)[1].|>real
    grape.control_coefficients += grape.ϵ*δF
    bound!(grape.control_coefficients, grape.ctrl_bound)
    return δF
end

function gradient_QFI_Adam!(grape::GRAPE_Copt{T}) where {T<:Complex}
    δF = gradient(x->QFI_auto(grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan), grape.control_coefficients)[1].|>real
    Adam!(grape, δF)
    bound!(grape.control_coefficients, grape.ctrl_bound)
    return δF
end

function gradient_QFI_bfgs(grape::GRAPE_Copt{T}) where {T<:Complex}
    δF = gradient(x->QFI_auto(grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan), grape.control_coefficients)[1].|>real
    return δF
end

function gradient_QFIM!(grape::GRAPE_Copt{T}) where {T<:Complex}
    δF = gradient(x->1/(grape.W*(QFIM_auto(grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan) |> pinv) |> tr |>real), grape.control_coefficients).|>real |>sum
    grape.control_coefficients += grape.ϵ*δF
    bound!(grape.control_coefficients, grape.ctrl_bound)
    return δF
end

function gradient_QFIM_Adam!(grape::GRAPE_Copt{T}) where {T<:Complex}
    δF = gradient(x->1/(grape.W*(QFIM_auto(grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan) |> pinv) |> tr |>real), grape.control_coefficients).|>real |>sum
    Adam!(grape, δF)
    bound!(grape.control_coefficients, grape.ctrl_bound)
    return δF
end

function gradient_QFIM_bfgs(grape::GRAPE_Copt{T}) where {T<:Complex}
    δF = gradient(x->1/(grape.W*(QFIM_auto(grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan) |> pinv) |> tr |>real), grape.control_coefficients).|>real |>sum
    return δF
end

function gradient_CFI!(grape::GRAPE_Copt{T}, Measurement) where {T<:Complex}
    δI = gradient(x->CFI(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan, grape.accuracy), grape.control_coefficients)[1].|>real
    grape.control_coefficients += grape.ϵ*δI
    bound!(grape.control_coefficients, grape.ctrl_bound)
    return δI
end

function gradient_CFI_Adam!(grape::GRAPE_Copt{T}, Measurement) where {T<:Complex}
    δI = gradient(x->CFI(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan, grape.accuracy), grape.control_coefficients)[1].|>real
    Adam!(grape, δI)
    bound!(grape.control_coefficients, grape.ctrl_bound)
    return δI
end

function gradient_CFI_bfgs(grape::GRAPE_Copt{T}, Measurement) where {T<:Complex}
    δF = gradient(x->CFI(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan, grape.accuracy), grape.control_coefficients)[1].|>real
    return δF
end

function gradient_CFIM!(grape::GRAPE_Copt{T}, Measurement) where {T<:Complex}
    δI = gradient(x->1/(grape.W*(CFIM(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan, grape.accuracy) |> pinv) |> tr |>real), grape.control_coefficients).|>real |>sum
    grape.control_coefficients += grape.ϵ*δI
    bound!(grape.control_coefficients, grape.ctrl_bound)
    return δI
end

function gradient_CFIM_Adam!(grape::GRAPE_Copt{T}, Measurement) where {T<:Complex}
    δI = gradient(x->1/(grape.W*(CFIM(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan, grape.accuracy) |> pinv) |> tr |>real), grape.control_coefficients).|>real |>sum
    Adam!(grape, δI)
    bound!(grape.control_coefficients, grape.ctrl_bound)
    return δI
end

function gradient_CFIM_bfgs(grape::GRAPE_Copt{T}, Measurement) where {T<:Complex}
    δF = gradient(x->1/(grape.W*(CFIM(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan, grape.accuracy) |> pinv) |> tr |>real), grape.control_coefficients).|>real |>sum
    return δF
end


function dynamics_analy(grape::GRAPE_Copt{T}, dim, tnum, para_num, ctrl_num) where {T<:Complex}
    Δt = grape.tspan[2] - grape.tspan[1]
    H = Htot(grape.freeHamiltonian, grape.control_Hamiltonian, grape.control_coefficients)

    ρt = [Vector{ComplexF64}(undef, dim^2) for i in 1:tnum]
    ∂ρt_∂x = [[Vector{ComplexF64}(undef, dim^2) for para in 1:para_num] for i in 1:tnum]
    δρt_δV = [[] for ctrl in 1:ctrl_num]
    ∂xδρt_δV = [[[] for ctrl in 1:ctrl_num] for i in 1:para_num]
    ∂H_L = [Matrix{ComplexF64}(undef, dim^2,dim^2) for i in 1:para_num]
    Hc_L = [Matrix{ComplexF64}(undef, dim^2,dim^2) for i in 1:ctrl_num]

    ρt[1] = grape.ρ0 |> vec
    for cj in 1:ctrl_num
        Hc_L[cj] = liouville_commu(grape.control_Hamiltonian[cj])
        append!(δρt_δV[cj], [-im*Δt*Hc_L[cj]*ρt[1]])
    end

    for pj in 1:para_num
        ∂ρt_∂x[1][pj] = ρt[1] |> zero
        ∂H_L[pj] = liouville_commu(grape.Hamiltonian_derivative[pj])
        for ci in 1:ctrl_num
            append!(∂xδρt_δV[pj][ci], [-im*Δt*Hc_L[ci]*∂ρt_∂x[1][pj]])
        end
    end

    for ti in 2:tnum
        
        expL = evolute(H[ti-1], grape.decay_opt, grape.γ, Δt, ti)
        ρt[ti] = expL * ρt[ti-1]
        for pk in 1:para_num
            ∂ρt_∂x[ti][pk] = -im * Δt * ∂H_L[pk] * ρt[ti] + expL * ∂ρt_∂x[ti-1][pk]
        end
        
        for ck in 1:ctrl_num
            for tk in 1:(ti-1)
                δρt_δV_first = popfirst!(δρt_δV[ck])
                δρt_δV_tp = expL * δρt_δV_first
                append!(δρt_δV[ck], [δρt_δV_tp])
                for pk in 1:para_num
                    ∂xδρt_δV_first = popfirst!(∂xδρt_δV[pk][ck])
                    ∂xδρt_δV_tp = -im * Δt * ∂H_L[pk] * expL * δρt_δV_first + expL * ∂xδρt_δV_first
                    append!(∂xδρt_δV[pk][ck], [∂xδρt_δV_tp])
                end
            end
            δρt_δV_last = -im * Δt * Hc_L[ck] * ρt[ti]
            append!(δρt_δV[ck], [δρt_δV_last])
            for pk in 1:para_num
                ∂xδρt_δV_last = -im * Δt * Hc_L[ck] * ∂ρt_∂x[ti][pk]
                append!(∂xδρt_δV[pk][ck], [∂xδρt_δV_last])
            end
        end
    end

    ρt_T = ρt[end] |> vec2mat
    ∂ρt_T = [(∂ρt_∂x[end][para] |> vec2mat) for para in 1:para_num]

    return ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV
end

function gradient_QFIM_analy_Adam(grape::GRAPE_Copt{T}) where {T<:Complex}
    dim = size(grape.ρ0)[1]
    tnum = length(grape.tspan)
    para_num = length(grape.Hamiltonian_derivative)
    ctrl_num = length(grape.control_Hamiltonian)
    
    ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV = dynamics_analy(grape, dim, tnum, para_num, ctrl_num)

    Lx = SLD(ρt_T, ∂ρt_T)
    F_T = QFIM(ρt_T, ∂ρt_T, grape.accuracy)

    if para_num == 1
        cost_function = F_T[1]
        anti_commu = 2*Lx[1]*Lx[1]
        for cm in 1:ctrl_num
            mt = grape.mt
            vt = grape.vt
            for tm in 1:(tnum-1)
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                ∂xδρt_T_δV = ∂xδρt_δV[1][cm][tm] |> vec2mat
                term1 = tr(∂xδρt_T_δV*Lx[1])
                term2 = tr(∂ρt_T_δV*anti_commu)
                δF = ((2*term1-0.5*term2) |> real)
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.accuracy)
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)

    elseif para_num == 2
        coeff1 = real(det(F))
        coeff2 = grape.W[1,1]*F_T[2,2]+grape.W[2,2]*F_T[1,1]-grape.W[1,2]*F_T[2,1]-grape.W[2,1]*F_T[1,2]
        cost_function = real(tr(grape.W*pinv(F_T)))
        for cm in 1:ctrl_num
            mt = grape.mt
            vt = grape.vt
            for tm in 1:(tnum-1)
                δF_all = [[0.0 for i in 1:para_num] for j in 1:para_num]
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                for pm in 1:para_num
                    for pn in 1:para_num
                        ∂xδρt_T_δV_a = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                        ∂xδρt_T_δV_b = ∂xδρt_δV[pn][cm][tm] |> vec2mat
                        term1 = tr(∂xδρt_T_δV_a * Lx[pn])
                        term2 = tr(∂xδρt_T_δV_b * Lx[pm])

                        anti_commu = Lx[pm] * Lx[pn] + Lx[pn] * Lx[pm]
                        term2 = tr(∂ρt_T_δV * anti_commu)
                        δF_all[pm][pn] = ((2*term1-0.5*term2) |> real)
                    end
                end
                item1 = -coeff2*(F_T[2,2]*δF_all[1][1]+F_T[1,1]*δF_all[2][2]-F_T[2,1]*δF_all[1][2]-F_T[1,2]*δF_all[2][1])/coeff1^2
                item2 = (grape.W[1,1]*δF_all[2][2]+grape.W[2,2]*δF_all[1][1]-grape.W[1,2]*δF_all[2][1]-grape.W[2,1]*δF_all[1][2])/coeff1
                δF = -(item1+item2)*cost_function^2
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.accuracy)
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)
    else       
        cost_function = real(tr(grape.W*pinv(F_T)))
        coeff = [grape.W[para,para]/F_T[para,para] for para in 1:para_num] |>sum
        coeff = coeff^(-2)
        for cm in 1:ctrl_num
            mt = grape.mt
            vt = grape.vt
            for tm in 1:(tnum-1)
                δF = 0.0
                for pm in 1:para_num
                    ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                    ∂xδρt_T_δV = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                    term1 = tr(∂xδρt_T_δV * Lx[pm])
                    anti_commu = 2 * Lx[pm] * Lx[pm]
                    term2 = tr(∂ρt_T_δV * anti_commu)
                    δF = δF + grape.W[pm,pm]*(1.0/F_T[pm,pm]/F_T[pm,pm])*((2*term1-0.5*term2) |> real)
                end
                δF = δF*coeff
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.accuracy)
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)
    end
    grape.control_coefficients, cost_function
end

function gradient_QFIM_analy(grape::GRAPE_Copt{T}) where {T<:Complex}
    dim = size(grape.ρ0)[1]
    tnum = length(grape.tspan)
    para_num = length(grape.Hamiltonian_derivative)
    ctrl_num = length(grape.control_Hamiltonian)
    
    ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV = dynamics_analy(grape, dim, tnum, para_num, ctrl_num)

    Lx = SLD(ρt_T, ∂ρt_T)
    F_T = QFIM(ρt_T, ∂ρt_T, grape.accuracy)

    cost_function = F_T[1]
    
    if para_num == 1
        anti_commu = 2*Lx[1]*Lx[1]
        for cm in 1:ctrl_num
            mt = grape.mt
            vt = grape.vt
            for tm in 1:(tnum-1)
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                ∂xδρt_T_δV = ∂xδρt_δV[1][cm][tm] |> vec2mat
                term1 = tr(∂xδρt_T_δV*Lx[1])
                term2 = tr(∂ρt_T_δV*anti_commu)
                δF = ((2*term1-0.5*term2) |> real)
                grape.control_coefficients[cm][tm] = grape.control_coefficients[cm][tm] + grape.ϵ*δF
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)

    elseif para_num == 2
        coeff1 = real(det(F))
        coeff2 = grape.W[1,1]*F_T[2,2]+grape.W[2,2]*F_T[1,1]-grape.W[1,2]*F_T[2,1]-grape.W[2,1]*F_T[1,2]
        cost_function = real(tr(grape.W*pinv(F_T)))
        for cm in 1:ctrl_num
            for tm in 1:(tnum-1)
                δF_all = [[0.0 for i in 1:para_num] for j in 1:para_num]
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                for pm in 1:para_num
                    for pn in 1:para_num
                        ∂xδρt_T_δV_a = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                        ∂xδρt_T_δV_b = ∂xδρt_δV[pn][cm][tm] |> vec2mat
                        term1 = tr(∂xδρt_T_δV_a * Lx[pn])
                        term2 = tr(∂xδρt_T_δV_b * Lx[pm])

                        anti_commu = Lx[pm] * Lx[pn] + Lx[pn] * Lx[pm]
                        term2 = tr(∂ρt_T_δV * anti_commu)
                        δF_all[pm][pn] = ((2*term1-0.5*term2) |> real)
                    end
                end
                item1 = -coeff2*(F_T[2,2]*δF_all[1][1]+F_T[1,1]*δF_all[2][2]-F_T[2,1]*δF_all[1][2]-F_T[1,2]*δF_all[2][1])/coeff1^2
                item2 = (grape.W[1,1]*δF_all[2][2]+grape.W[2,2]*δF_all[1][1]-grape.W[1,2]*δF_all[2][1]-grape.W[2,1]*δF_all[1][2])/coeff1
                δF = -(item1+item2)*cost_function^2
                grape.control_coefficients[cm][tm] = grape.control_coefficients[cm][tm] + grape.ϵ*δF
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)

    else
        cost_function = real(tr(grape.W*pinv(F_T)))
        coeff = [grape.W[para,para]/F_T[para,para] for para in 1:para_num] |>sum
        coeff = coeff^(-2)
        for cm in 1:ctrl_num
            for tm in 1:(tnum-1)
                δF = 0.0
                for pm in 1:para_num
                    ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                    ∂xδρt_T_δV = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                    term1 = tr(∂xδρt_T_δV * Lx[pm])
                    anti_commu = 2 * Lx[pm] * Lx[pm]
                    term2 = tr(∂ρt_T_δV * anti_commu)
                    δF = δF + grape.W[pm,pm]*(1.0/F_T[pm,pm]/F_T[pm,pm])*((2*term1-0.5*term2) |> real)
                end
                δF = δF*coeff
                grape.control_coefficients[cm][tm] = grape.control_coefficients[cm][tm] + grape.ϵ*δF
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)
    end
    grape.control_coefficients, cost_function
end

function gradient_CFIM_analy_Adam(Measurement::Vector{Matrix{T}}, grape::GRAPE_Copt{T}) where {T<:Complex}
    dim = size(grape.ρ0)[1]
    tnum = length(grape.tspan)
    para_num = length(grape.Hamiltonian_derivative)
    ctrl_num = length(grape.control_Hamiltonian)
    
    ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV = dynamics_analy(grape, dim, tnum, para_num, ctrl_num)

    if para_num == 1
        F_T = CFI(ρt_T, ∂ρt_T[1], Measurement, grape.accuracy)
        cost_function = F_T
        L1_tidle = zeros(ComplexF64, dim, dim)
        L2_tidle = zeros(ComplexF64, dim, dim)

        for mi in 1:dim
            p = (tr(ρt_T*Measurement[mi]) |> real)
            dp = (tr(∂ρt_T[1]*Measurement[mi]) |> real)
            if p > grape.accuracy
                L1_tidle = L1_tidle + dp*Measurement[mi]/p
                L2_tidle = L2_tidle + dp*dp*Measurement[mi]/p^2
            end
        end

        for cm in 1:ctrl_num
            mt = grape.mt
            vt = grape.vt
            for tm in 1:(tnum-1)
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                ∂xδρt_T_δV = ∂xδρt_δV[1][cm][tm] |> vec2mat
                term1 = tr(∂xδρt_T_δV*L1_tidle)
                term2 = tr(∂ρt_T_δV*L2_tidle)
                δF = ((2*term1-term2) |> real)
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.accuracy)
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)

    elseif para_num == 2
        F_T = CFIM(ρt_T, ∂ρt_T, Measurement, grape.accuracy)
        L1_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
        L2_tidle = [[zeros(ComplexF64, dim, dim) for i in 1:para_num] for j in 1:para_num]
    
        for para_i in 1:para_num
            for mi in 1:dim
                p = (tr(ρt_T*Measurement[mi]) |> real)
                dp = (tr(∂ρt_T[para_i]*Measurement[mi]) |> real)
                if p > grape.accuracy
                    L1_tidle[para_i] = L1_tidle[para_i] + dp*Measurement[mi]/p
                end
            end
        end
    
        for para_i in 1:para_num
            dp_a = (tr(∂ρt_T[para_i]*Measurement[mi]) |> real)
            for para_j in 1:para_num
                dp_b = (tr(∂ρt_T[para_j]*Measurement[mi]) |> real)
                for mi in 1:dim
                    p = (tr(ρt_T*Measurement[mi]) |> real)
                    if p > grape.accuracy
                        L2_tidle[para_i][para_j] = L2_tidle[para_i][para_j] + dp_a*dp_b*Measurement[mi]/p^2
                    end
                end
            end
        end
        coeff1 = real(det(F))
        coeff2 = grape.W[1,1]*F_T[2,2]+grape.W[2,2]*F_T[1,1]-grape.W[1,2]*F_T[2,1]-grape.W[2,1]*F_T[1,2]
        cost_function = real(tr(grape.W*pinv(F_T)))
        for cm in 1:ctrl_num
            mt = grape.mt
            vt = grape.vt
            for tm in 1:(tnum-1)
                δF_all = [[0.0 for i in 1:para_num] for j in 1:para_num]
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                for pm in 1:para_num
                    for pn in 1:para_num
                        ∂xδρt_T_δV_a = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                        ∂xδρt_T_δV_b = ∂xδρt_δV[pn][cm][tm] |> vec2mat
                        term1 = tr(∂xδρt_T_δV_a * L1_tidle[pn])
                        term2 = tr(∂xδρt_T_δV_b * L1_tidle[pm])
                        term3 = tr(∂ρt_T_δV * L2_tidle[pm][pn])
                        δF_all[pm][pn] = ((term1+term2-term3) |> real)
                    end
                end
                item1 = -coeff2*(F_T[2,2]*δF_all[1][1]+F_T[1,1]*δF_all[2][2]-F_T[2,1]*δF_all[1][2]-F_T[1,2]*δF_all[2][1])/coeff1^2
                item2 = (grape.W[1,1]*δF_all[2][2]+grape.W[2,2]*δF_all[1][1]-grape.W[1,2]*δF_all[2][1]-grape.W[2,1]*δF_all[1][2])/coeff1
                δF = -(item1+item2)*cost_function^2
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.accuracy)
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)

    else
        F_T = CFIM(ρt_T, ∂ρt_T, Measurement, grape.accuracy)
        L1_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
        L2_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
    
        for para_i in 1:para_num
            for mi in 1:dim
                p = (tr(ρt_T*Measurement[mi]) |> real)
                dp = (tr(∂ρt_T[para_i]*Measurement[mi]) |> real)
                if p > grape.accuracy
                    L1_tidle[para_i] = L1_tidle[para_i] + dp*Measurement[mi]/p
                    L2_tidle[para_i] = L2_tidle[para_i] + dp*dp*Measurement[mi]/p^2
                end
            end
        end

        cost_function = real(tr(grape.W*pinv(F_T)))
        coeff = [grape.W[para,para]/F_T[para,para] for para in 1:para_num] |>sum
        coeff = coeff^(-2)
        for cm in 1:ctrl_num
            mt = grape.mt
            vt = grape.vt
            for tm in 1:(tnum-1)
                δF = 0.0
                for pm in 1:para_num
                    ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                    ∂xδρt_T_δV = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                    term1 = tr(∂xδρt_T_δV * L1_tidle[pm])
                    term2 = tr(∂ρt_T_δV * L2_tidle[pm])
                    δF = δF + grape.W[pm,pm]*(1.0/F_T[pm,pm]/F_T[pm,pm])*((2*term1-term2) |> real)
                end
                δF = δF*coeff
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.accuracy)
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)
    end
    grape.control_coefficients, cost_function
end

function gradient_CFIM_analy(Measurement::Vector{Matrix{T}}, grape::GRAPE_Copt{T}) where {T<:Complex}
    dim = size(grape.ρ0)[1]
    tnum = length(grape.tspan)
    para_num = length(grape.Hamiltonian_derivative)
    ctrl_num = length(grape.control_Hamiltonian)
    
    ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV = dynamics_analy(grape, dim, tnum, para_num, ctrl_num)

    if para_num == 1
        F_T = CFI(ρt_T, ∂ρt_T[1], Measurement, grape.accuracy)
        cost_function = F_T
        L1_tidle = zeros(ComplexF64, dim, dim)
        L2_tidle = zeros(ComplexF64, dim, dim)

        for mi in 1:dim
            p = (tr(ρt_T*Measurement[mi]) |> real)
            dp = (tr(∂ρt_T[1]*Measurement[mi]) |> real)
            if p > grape.accuracy
                L1_tidle = L1_tidle + dp*Measurement[mi]/p
                L2_tidle = L2_tidle + dp*dp*Measurement[mi]/p^2
            end
        end

        for cm in 1:ctrl_num
            for tm in 1:(tnum-1)
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                ∂xδρt_T_δV = ∂xδρt_δV[1][cm][tm] |> vec2mat
                term1 = tr(∂xδρt_T_δV*L1_tidle)
                term2 = tr(∂ρt_T_δV*L2_tidle)
                δF = ((2*term1-term2) |> real)
                grape.control_coefficients[cm][tm] = grape.control_coefficients[cm][tm] + grape.ϵ*δF
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)

    elseif para_num == 2
        F_T = CFIM(ρt_T, ∂ρt_T, Measurement, grape.accuracy)
        L1_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
        L2_tidle = [[zeros(ComplexF64, dim, dim) for i in 1:para_num] for j in 1:para_num]
    
        for para_i in 1:para_num
            for mi in 1:dim
                p = (tr(ρt_T*Measurement[mi]) |> real)
                dp = (tr(∂ρt_T[para_i]*Measurement[mi]) |> real)
                if p > grape.accuracy
                    L1_tidle[para_i] = L1_tidle[para_i] + dp*Measurement[mi]/p
                end
            end
        end
    
        for para_i in 1:para_num
            dp_a = (tr(∂ρt_T[para_i]*Measurement[mi]) |> real)
            for para_j in 1:para_num
                dp_b = (tr(∂ρt_T[para_j]*Measurement[mi]) |> real)
                for mi in 1:dim
                    p = (tr(ρt_T*Measurement[mi]) |> real)
                    if p > grape.accuracy
                        L2_tidle[para_i][para_j] = L2_tidle[para_i][para_j] + dp_a*dp_b*Measurement[mi]/p^2
                    end
                end
            end
        end
        coeff1 = real(det(F))
        coeff2 = grape.W[1,1]*F_T[2,2]+grape.W[2,2]*F_T[1,1]-grape.W[1,2]*F_T[2,1]-grape.W[2,1]*F_T[1,2]
        cost_function = real(tr(grape.W*pinv(F_T)))
        for cm in 1:ctrl_num
            for tm in 1:(tnum-1)
                δF_all = [[0.0 for i in 1:para_num] for j in 1:para_num]
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                for pm in 1:para_num
                    for pn in 1:para_num
                        ∂xδρt_T_δV_a = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                        ∂xδρt_T_δV_b = ∂xδρt_δV[pn][cm][tm] |> vec2mat
                        term1 = tr(∂xδρt_T_δV_a * L1_tidle[pn])
                        term2 = tr(∂xδρt_T_δV_b * L1_tidle[pm])
                        term3 = tr(∂ρt_T_δV * L2_tidle[pm][pn])
                        δF_all[pm][pn] = ((term1+term2-term3) |> real)
                    end
                end
                item1 = -coeff2*(F_T[2,2]*δF_all[1][1]+F_T[1,1]*δF_all[2][2]-F_T[2,1]*δF_all[1][2]-F_T[1,2]*δF_all[2][1])/coeff1^2
                item2 = (grape.W[1,1]*δF_all[2][2]+grape.W[2,2]*δF_all[1][1]-grape.W[1,2]*δF_all[2][1]-grape.W[2,1]*δF_all[1][2])/coeff1
                δF = -(item1+item2)*cost_function^2
                grape.control_coefficients[cm][tm] = grape.control_coefficients[cm][tm] + grape.ϵ*δF
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)

    else
        F_T = CFIM(ρt_T, ∂ρt_T, Measurement, grape.accuracy)
        L1_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
        L2_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
    
        for para_i in 1:para_num
            for mi in 1:dim
                p = (tr(ρt_T*Measurement[mi]) |> real)
                dp = (tr(∂ρt_T[para_i]*Measurement[mi]) |> real)
                if p > grape.accuracy
                    L1_tidle[para_i] = L1_tidle[para_i] + dp*Measurement[mi]/p
                    L2_tidle[para_i] = L2_tidle[para_i] + dp*dp*Measurement[mi]/p^2
                end
            end
        end

        cost_function = real(tr(grape.W*pinv(F_T)))
        coeff = [grape.W[para,para]/F_T[para,para] for para in 1:para_num] |>sum
        coeff = coeff^(-2)
        for cm in 1:ctrl_num
            for tm in 1:(tnum-1)
                δF = 0.0
                for pm in 1:para_num
                    ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                    ∂xδρt_T_δV = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                    term1 = tr(∂xδρt_T_δV * L1_tidle[pm])
                    term2 = tr(∂ρt_T_δV * L2_tidle[pm])
                    δF = δF + grape.W[pm,pm]*(1.0/F_T[pm,pm]/F_T[pm,pm])*((2*term1-term2) |> real)
                end
                δF = δF*coeff
                grape.control_coefficients[cm][tm] = grape.control_coefficients[cm][tm] + grape.ϵ*δF
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)
    end
    grape.control_coefficients, cost_function
end

function QFIM_autoGRAPE_Copt(grape, max_episode, Adam, save_file)
    println("quantum parameter estimation")
    ctrl_num = length(grape.control_Hamiltonian)
    ctrl_length = length(grape.control_coefficients[1])
    episodes = 1
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: auto-GRAPE")
        f_noctrl = QFI(grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ0, grape.decay_opt, grape.γ, 
                        grape.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], grape.tspan, grape.accuracy)
        f_ini = QFI(grape)
        f_list = [f_ini]
        println("non-controlled QFI is $(f_noctrl)")
        println("initial QFI is $(f_ini)")
        if save_file == true
            SaveFile_ctrl(f_ini, grape.control_coefficients)
            if Adam == true
                gradient_QFI_Adam!(grape)
                while true
                    f_now = QFI(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        break
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        print("current QFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFI_Adam!(grape)
                end
            else
                gradient_QFI!(grape)
                while true
                    f_now = QFI(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        break
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        print("current QFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFI!(grape)
                end
            end
        else
            if Adam == true
                gradient_QFI_Adam!(grape)
                while true
                    f_now = QFI(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, grape.control_coefficients)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current QFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFI_Adam!(grape)
                end
            else
                gradient_QFI!(grape)
                while true
                    f_now = QFI(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, grape.control_coefficients)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current QFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFI!(grape)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("control algorithm: auto-GRAPE")
        F_noctrl = QFIM(grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ0, grape.decay_opt, grape.γ, 
                        grape.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], grape.tspan, grape.accuracy)
        f_noctrl = real(tr(grape.W*pinv(F_noctrl)))
        F_ini = QFIM(grape)
        f_ini = real(tr(grape.W*pinv(F_ini)))
        f_list = [f_ini]
        println("non-controlled value of Tr(WF^{-1}) is $(f_noctrl)")
        println("initial value of Tr(WF^{-1}) is $(f_ini)")
        if save_file == true
            SaveFile_ctrl(f_ini, grape.control_coefficients)
            if Adam == true
                gradient_QFIM_Adam!(grape)
                while true
                    f_now = real(tr(grape.W*pinv(QFIM(grape))))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        break
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(grape)
                end
            else
                gradient_QFIM!(grape)
                while true
                    f_now = real(tr(grape.W*pinv(QFIM(grape))))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        break
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFIM!(grape)
                end
            end
        else
            if Adam == true
                gradient_QFIM_Adam!(grape)
                while true
                    f_now = real(tr(grape.W*pinv(QFIM(grape))))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, grape.control_coefficients)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(grape)
                end
            else
                gradient_QFIM!(grape)
                while true
                    f_now = real(tr(grape.W*pinv(QFIM(grape))))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, grape.control_coefficients)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFIM!(grape)
                end
            end
        end
    end
end

function QFIM_GRAPE_Copt(grape, max_episode, Adam, save_file)
    println("quantum parameter estimation")
    ctrl_num = length(grape.control_Hamiltonian)
    ctrl_length = length(grape.control_coefficients[1])
    episodes = 1
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: GRAPE")
        f_noctrl = QFI(grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ0, grape.decay_opt, grape.γ, 
                        grape.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], grape.tspan, grape.accuracy)
        println("non-controlled QFI is $(f_noctrl)")
        ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
        if Adam == true
            grape.control_coefficients, f_ini = gradient_QFIM_analy_Adam(grape)
        else
            grape.control_coefficients, f_ini = gradient_QFIM_analy(grape)
        end
        f_list = [f_ini]
        println("initial QFI is $(f_ini)")
        if save_file == true
            SaveFile_ctrl(f_ini, ctrl_pre)
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy_Adam(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile_ctrl(f_now, ctrl_pre) 
                        break 
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, ctrl_pre)
                        print("current QFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end  
                end 
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile_ctrl(f_now, ctrl_pre) 
                        break 
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, ctrl_pre)
                        print("current QFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end 
            end                    
        else
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy_Adam(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, ctrl_pre)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current QFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, ctrl_pre)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current QFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            end
        end
    else
        println("multiparameter scenario")
        println("control algorithm: GRAPE")
        F_noctrl = QFIM(grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ0, grape.decay_opt, grape.γ, 
                        grape.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], grape.tspan, grape.accuracy)
        f_noctrl = real(tr(grape.W*pinv(F_noctrl)))
        println("non-controlled value of Tr(WF^{-1}) is $(f_noctrl)")
        ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
        if Adam == true
            grape.control_coefficients, f_ini = gradient_QFIM_analy_Adam(grape)
        else
            grape.control_coefficients, f_ini = gradient_QFIM_analy(grape)
        end
        f_list = [f_ini]
        println("initial value of Tr(WF^{-1}) is $(f_ini)")
        if save_file == true
            SaveFile_ctrl(f_ini, ctrl_pre)
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy_Adam(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile_ctrl(f_now, ctrl_pre)
                        break
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, ctrl_pre)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile_ctrl(f_now, ctrl_pre)
                        break
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, ctrl_pre)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy_Adam(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, ctrl_pre)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, ctrl_pre)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            end
        end
    end
end

function CFIM_autoGRAPE_Copt(Measurement, grape, max_episode, Adam, save_file)
    println("classical parameter estimation")
    ctrl_num = length(grape.control_Hamiltonian)
    ctrl_length = length(grape.control_coefficients[1])
    episodes = 1
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: auto_GRAPE")
        f_noctrl = CFI(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ0, grape.decay_opt, grape.γ, 
                        grape.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], grape.tspan, grape.accuracy)
        f_ini = CFI(Measurement, grape)
        f_list = [f_ini]
        println("non-controlled CFI is $(f_noctrl)")
        println("initial CFI is $(f_ini)")
        if save_file == true
            SaveFile_ctrl(f_ini, grape.control_coefficients)
            if Adam == true
                gradient_CFI_Adam!(grape, Measurement)
                while true
                    f_now = CFI(Measurement, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        break
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        print("current CFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFI_Adam!(grape, Measurement)
                end
            else
                gradient_CFI!(grape, Measurement)
                while true
                    f_now = CFI(Measurement, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        break
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        print("current CFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFI!(grape, Measurement)
                end
            end
        else
            if Adam == true
                gradient_CFI_Adam!(grape, Measurement)
                while true
                    f_now = CFI(Measurement, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, grape.control_coefficients)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFI_Adam!(grape, Measurement)
                end
            else
                gradient_CFI!(grape, Measurement)
                while true
                    f_now = CFI(Measurement, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, grape.control_coefficients)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFI!(grape, Measurement)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("control algorithm: auto-GRAPE")
        F_noctrl = CFIM(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ0, grape.decay_opt, grape.γ, 
                        grape.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], grape.tspan, grape.accuracy)
        f_noctrl = real(tr(grape.W*pinv(F_noctrl)))
        F_ini = CFIM(Measurement, grape)
        f_ini = real(tr(grape.W*pinv(F_ini)))
        f_list = [f_ini]
        println("non-controlled value of Tr(WI^{-1}) is $(f_noctrl)")
        println("initial value of Tr(WI^{-1}) is $(f_ini)")
        if save_file == true
            SaveFile_ctrl(f_ini, grape.control_coefficients)
            if Adam == true
                gradient_CFIM_Adam!(grape, Measurement)
                while true
                    f_now = real(tr(grape.W*pinv(CFIM(Measurement, grape))))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        break
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(grape, Measurement)
                end
            else
                gradient_CFIM!(grape, Measurement)
                while true
                    f_now = real(tr(grape.W*pinv(CFIM(Measurement, grape))))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        break
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFIM!(grape, Measurement)
                end
            end
        else
            if Adam == true
                gradient_CFIM_Adam!(grape, Measurement)
                while true
                    f_now = real(tr(grape.W*pinv(CFIM(Measurement, grape))))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, grape.control_coefficients)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(grape, Measurement)
                end
            else
                gradient_CFIM!(grape, Measurement)
                while true
                    f_now = real(tr(grape.W*pinv(CFIM(Measurement, grape))))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, grape.control_coefficients)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFIM!(grape, Measurement)
                end
            end
        end
    end
end

function CFIM_GRAPE_Copt(Measurement, grape, max_episode, Adam, save_file)
    println("classical parameter estimation")
    ctrl_num = length(grape.control_Hamiltonian)
    ctrl_length = length(grape.control_coefficients[1])
    episodes = 1
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: GRAPE")
        f_noctrl = CFI(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ0, grape.decay_opt, grape.γ, 
                     grape.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], grape.tspan, grape.accuracy)
        println("non-controlled CFI is $(f_noctrl)")
        ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
        if Adam == true
            grape.control_coefficients, f_ini = gradient_CFIM_analy_Adam(Measurement, grape)
        else
            grape.control_coefficients, f_ini = gradient_CFIM_analy(Measurement, grape)
        end
        f_list = [f_ini]
        println("initial CFI is $(f_ini)")
        if save_file == true
            SaveFile_ctrl(f_ini, ctrl_pre) 
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy_Adam(Measurement, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile_ctrl(f_now, ctrl_pre)                    
                        break 
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, ctrl_pre)
                        print("current CFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end 
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy(Measurement, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile_ctrl(f_now, ctrl_pre)                    
                        break 
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, ctrl_pre)
                        print("current CFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end 
                end  
            end                     
        else
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy_Adam(Measurement, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, ctrl_pre)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy(Measurement, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, ctrl_pre)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            end
        end
    else
        println("multiparameter scenario")
        println("control algorithm: GRAPE")
        F_noctrl = CFIM(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ0, grape.decay_opt, grape.γ, 
                        grape.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], grape.tspan, grape.accuracy)
        f_noctrl = real(tr(grape.W*pinv(F_noctrl)))
        println("non-controlled value of Tr(WI^{-1}) is $(f_noctrl)")
        ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
        if Adam == true
            grape.control_coefficients, f_ini = gradient_CFIM_analy_Adam(Measurement, grape)
        else
            grape.control_coefficients, f_ini = gradient_CFIM_analy(Measurement, grape)
        end
        f_list = [f_ini]
        println("initial value of Tr(WI^{-1}) is $(f_ini)")
        if save_file == true
            SaveFile_ctrl(f_ini, ctrl_pre)
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy_Adam(Measurement, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        SaveFile_ctrl(f_now, ctrl_pre)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        SaveFile_ctrl(f_now, ctrl_pre)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy(Measurement, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        SaveFile_ctrl(f_now, ctrl_pre)
                        break
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, ctrl_pre)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy_Adam(Measurement, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, ctrl_pre)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy(Measurement, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, ctrl_pre)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            end
        end
    end
end
