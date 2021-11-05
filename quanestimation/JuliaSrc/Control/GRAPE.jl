abstract type ControlSystem end
mutable struct Gradient{T <: Complex,M <: Real} <: ControlSystem
    freeHamiltonian::Matrix{T}
    Hamiltonian_derivative::Vector{Matrix{T}}
    ρ_initial::Matrix{T}
    times::Vector{M}
    Liouville_operator::Vector{Matrix{T}}
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
    precision::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    Gradient(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, ρ_initial::Matrix{T},
                 times::Vector{M}, Liouville_operator::Vector{Matrix{T}},γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}},
                 control_coefficients::Vector{Vector{M}}, ctrl_bound::Vector{M}, W::Matrix{M}, mt::M, vt::M, ϵ::M, beta1::M, beta2::M, precision::M, 
                 ρ=Vector{Matrix{T}}(undef, 1), ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1),∂ρ_∂V=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = 
                 new{T,M}(freeHamiltonian, Hamiltonian_derivative, ρ_initial, times, Liouville_operator, γ, control_Hamiltonian,
                          control_coefficients, ctrl_bound, W, mt, vt, ϵ, beta1, beta2, precision, ρ, ∂ρ_∂x) 
end

function gradient_CFI!(grape::Gradient{T}, Measurement) where {T <: Complex}
    δI = gradient(x->CFI(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.control_Hamiltonian, x, grape.times), grape.control_coefficients)[1].|>real
    grape.control_coefficients += grape.ϵ*δI
    bound!(grape.control_coefficients, grape.ctrl_bound)
end

function gradient_CFI_Adam!(grape::Gradient{T}, Measurement) where {T <: Complex}
    δI = gradient(x->CFI(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.control_Hamiltonian, x, grape.times), grape.control_coefficients)[1].|>real
    Adam!(grape, δI)
    bound!(grape.control_coefficients, grape.ctrl_bound)
end

function gradient_CFIM!(grape::Gradient{T}, Measurement) where {T <: Complex}
    δI = gradient(x->1/(grape.W*(CFIM(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.control_Hamiltonian, x, grape.times) |> pinv) |> tr |>real), grape.control_coefficients).|>real |>sum
    grape.control_coefficients += grape.ϵ*δI
    bound!(grape.control_coefficients, grape.ctrl_bound)
end

function gradient_CFIM_Adam!(grape::Gradient{T}, Measurement) where {T <: Complex}
    δI = gradient(x->1/(grape.W*(CFIM(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.control_Hamiltonian, x, grape.times) |> pinv) |> tr |>real), grape.control_coefficients).|>real |>sum
    Adam!(grape, δI)
    bound!(grape.control_coefficients, grape.ctrl_bound)
end

function gradient_QFI!(grape::Gradient{T}) where {T <: Complex}
    δF = gradient(x->QFI_auto(grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.control_Hamiltonian, x, grape.times), grape.control_coefficients)[1].|>real
    grape.control_coefficients += grape.ϵ*δF
    bound!(grape.control_coefficients, grape.ctrl_bound)
end

function gradient_QFI_Adam!(grape::Gradient{T}) where {T <: Complex}
    δF = gradient(x->QFI_auto(grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.control_Hamiltonian, x, grape.times), grape.control_coefficients)[1].|>real
    Adam!(grape, δF)
    bound!(grape.control_coefficients, grape.ctrl_bound)
end

function gradient_QFIM!(grape::Gradient{T}) where {T <: Complex}
    δF = gradient(x->1/(grape.W*(QFIM_auto(grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.control_Hamiltonian, x, grape.times) |> pinv) |> tr |>real), grape.control_coefficients).|>real |>sum
    grape.control_coefficients += grape.ϵ*δF
    bound!(grape.control_coefficients, grape.ctrl_bound)
end

function gradient_QFIM_Adam!(grape::Gradient{T}) where {T <: Complex}
    δF = gradient(x->1/(grape.W*(QFIM_auto(grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.control_Hamiltonian, x, grape.times) |> pinv) |> tr |>real), grape.control_coefficients).|>real |>sum
    Adam!(grape, δF)
    bound!(grape.control_coefficients, grape.ctrl_bound)
end

function gradient_QFI_bfgs(grape::Gradient{T}, control_coefficients) where {T <: Complex}
    δF = gradient(x->QFI_auto(grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.control_Hamiltonian, x, grape.times), control_coefficients)[1].|>real
end

function dynamics_analy(grape::Gradient{T}, dim, tnum, para_num, ctrl_num) where {T <: Complex}
    Δt = grape.times[2] - grape.times[1]
    H = Htot(grape.freeHamiltonian, grape.control_Hamiltonian, grape.control_coefficients)

    ρt = [Vector{ComplexF64}(undef, dim^2)  for i in 1:tnum]
    ∂ρt_∂x = [[Vector{ComplexF64}(undef, dim^2) for para in 1:para_num] for i in 1:tnum]
    δρt_δV = [[] for ctrl in 1:ctrl_num]
    ∂xδρt_δV = [[[] for ctrl in 1:ctrl_num] for i in 1:para_num]
    ∂H_L = [Matrix{ComplexF64}(undef, dim^2,dim^2)  for i in 1:para_num]
    Hc_L = [Matrix{ComplexF64}(undef, dim^2,dim^2)  for i in 1:ctrl_num]

    ρt[1] = grape.ρ_initial |> vec
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
        
        expL = evolute(H[ti-1], grape.Liouville_operator, grape.γ, Δt, ti)
        ρt[ti] =  expL * ρt[ti-1]
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

function gradient_QFIM_analy_Adam(grape::Gradient{T}) where {T <: Complex}
    dim = size(grape.freeHamiltonian)[1]
    tnum = length(grape.times)
    para_num = length(grape.Hamiltonian_derivative)
    ctrl_num = length(grape.control_Hamiltonian)
    
    ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV = dynamics_analy(grape, dim, tnum, para_num, ctrl_num)

    Lx = SLD(ρt_T, ∂ρt_T)
    F_T = QFIM(ρt_T, ∂ρt_T)

    if para_num == 1
        cost_function = F_T[1]
        anti_commu = 2*Lx[1]*Lx[1]
        for cm in 1:ctrl_num
            mt = grape.mt
            vt = grape.vt
            for tm in 1:tnum
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                ∂xδρt_T_δV = ∂xδρt_δV[1][cm][tm] |> vec2mat
                term1 = tr(∂xδρt_T_δV*Lx[1])
                term2 = tr(∂ρt_T_δV*anti_commu)
                δF = ((2*term1-0.5*term2) |> real)
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.precision)
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
            for tm in 1:tnum
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
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.precision)
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
            for tm in 1:tnum
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
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.precision)
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)
    end
    grape.control_coefficients, cost_function
end

function gradient_QFIM_analy(grape::Gradient{T}) where {T <: Complex}
    dim = size(grape.freeHamiltonian)[1]
    tnum = length(grape.times)
    para_num = length(grape.Hamiltonian_derivative)
    ctrl_num = length(grape.control_Hamiltonian)
    
    ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV = dynamics_analy(grape, dim, tnum, para_num, ctrl_num)

    Lx = SLD(ρt_T, ∂ρt_T)
    F_T = QFIM(ρt_T, ∂ρt_T)

    cost_function = F_T[1]
    
    if para_num == 1
        anti_commu = 2*Lx[1]*Lx[1]
        for cm in 1:ctrl_num
            mt = grape.mt
            vt = grape.vt
            for tm in 1:tnum
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
            for tm in 1:tnum
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
            for tm in 1:tnum
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

function gradient_CFIM_analy_Adam(Measurement::Vector{Matrix{T}}, grape::Gradient{T}) where {T <: Complex}
    dim = size(grape.freeHamiltonian)[1]
    tnum = length(grape.times)
    para_num = length(grape.Hamiltonian_derivative)
    ctrl_num = length(grape.control_Hamiltonian)
    
    ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV = dynamics_analy(grape, dim, tnum, para_num, ctrl_num)

    if para_num == 1
        F_T = CFI(ρt_T, ∂ρt_T[1], Measurement)
        cost_function = F_T
        L1_tidle = zeros(ComplexF64, dim, dim)
        L2_tidle = zeros(ComplexF64, dim, dim)

        for mi in 1:dim
            p = (tr(ρt_T*Measurement[mi]) |> real)
            dp = (tr(∂ρt_T[1]*Measurement[mi]) |> real)
            if p > grape.precision
                L1_tidle = L1_tidle + dp*Measurement[mi]/p
                L2_tidle = L2_tidle + dp*dp*Measurement[mi]/p^2
            end
        end

        for cm in 1:ctrl_num
            mt = grape.mt
            vt = grape.vt
            for tm in 1:tnum
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                ∂xδρt_T_δV = ∂xδρt_δV[1][cm][tm] |> vec2mat
                term1 = tr(∂xδρt_T_δV*L1_tidle)
                term2 = tr(∂ρt_T_δV*L2_tidle)
                δF = ((2*term1-term2) |> real)
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.precision)
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)

    elseif para_num == 2
        F_T = CFIM(ρt_T, ∂ρt_T, Measurement)
        L1_tidle = [zeros(ComplexF64, dim, dim)  for i in 1:para_num]
        L2_tidle = [[zeros(ComplexF64, dim, dim)  for i in 1:para_num] for j in 1:para_num]
    
        for para_i in 1:para_num
            for mi in 1:dim
                p = (tr(ρt_T*Measurement[mi]) |> real)
                dp = (tr(∂ρt_T[para_i]*Measurement[mi]) |> real)
                if p > grape.precision
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
                    if p > grape.precision
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
            for tm in 1:tnum
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
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.precision)
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)

    else
        F_T = CFIM(ρt_T, ∂ρt_T, Measurement)
        L1_tidle = [zeros(ComplexF64, dim, dim)  for i in 1:para_num]
        L2_tidle = [zeros(ComplexF64, dim, dim)  for i in 1:para_num]
    
        for para_i in 1:para_num
            for mi in 1:dim
                p = (tr(ρt_T*Measurement[mi]) |> real)
                dp = (tr(∂ρt_T[para_i]*Measurement[mi]) |> real)
                if p > grape.precision
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
            for tm in 1:tnum
                δF = 0.0
                for pm in 1:para_num
                    ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                    ∂xδρt_T_δV = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                    term1 = tr(∂xδρt_T_δV * L1_tidle[pm])
                    term2 = tr(∂ρt_T_δV * L2_tidle[pm])
                    δF = δF + grape.W[pm,pm]*(1.0/F_T[pm,pm]/F_T[pm,pm])*((2*term1-term2) |> real)
                end
                δF = δF*coeff
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.precision)
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)
    end
    grape.control_coefficients, cost_function
end

function gradient_CFIM_analy(Measurement::Vector{Matrix{T}}, grape::Gradient{T}) where {T <: Complex}

    dim = size(grape.freeHamiltonian)[1]
    tnum = length(grape.times)
    para_num = length(grape.Hamiltonian_derivative)
    ctrl_num = length(grape.control_Hamiltonian)
    
    ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV = dynamics_analy(grape, dim, tnum, para_num, ctrl_num)

    if para_num == 1
        F_T = CFI(ρt_T, ∂ρt_T[1], Measurement)
        cost_function = F_T
        L1_tidle = zeros(ComplexF64, dim, dim)
        L2_tidle = zeros(ComplexF64, dim, dim)

        for mi in 1:dim
            p = (tr(ρt_T*Measurement[mi]) |> real)
            dp = (tr(∂ρt_T[1]*Measurement[mi]) |> real)
            if p > grape.precision
                L1_tidle = L1_tidle + dp*Measurement[mi]/p
                L2_tidle = L2_tidle + dp*dp*Measurement[mi]/p^2
            end
        end

        for cm in 1:ctrl_num
            for tm in 1:tnum
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
        F_T = CFIM(ρt_T, ∂ρt_T, Measurement)
        L1_tidle = [zeros(ComplexF64, dim, dim)  for i in 1:para_num]
        L2_tidle = [[zeros(ComplexF64, dim, dim)  for i in 1:para_num] for j in 1:para_num]
    
        for para_i in 1:para_num
            for mi in 1:dim
                p = (tr(ρt_T*Measurement[mi]) |> real)
                dp = (tr(∂ρt_T[para_i]*Measurement[mi]) |> real)
                if p > grape.precision
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
                    if p > grape.precision
                        L2_tidle[para_i][para_j] = L2_tidle[para_i][para_j] + dp_a*dp_b*Measurement[mi]/p^2
                    end
                end
            end
        end
        coeff1 = real(det(F))
        coeff2 = grape.W[1,1]*F_T[2,2]+grape.W[2,2]*F_T[1,1]-grape.W[1,2]*F_T[2,1]-grape.W[2,1]*F_T[1,2]
        cost_function = real(tr(grape.W*pinv(F_T)))
        for cm in 1:ctrl_num
            for tm in 1:tnum
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
        F_T = CFIM(ρt_T, ∂ρt_T, Measurement)
        L1_tidle = [zeros(ComplexF64, dim, dim)  for i in 1:para_num]
        L2_tidle = [zeros(ComplexF64, dim, dim)  for i in 1:para_num]
    
        for para_i in 1:para_num
            for mi in 1:dim
                p = (tr(ρt_T*Measurement[mi]) |> real)
                dp = (tr(∂ρt_T[para_i]*Measurement[mi]) |> real)
                if p > grape.precision
                    L1_tidle[para_i] = L1_tidle[para_i] + dp*Measurement[mi]/p
                    L2_tidle[para_i] = L2_tidle[para_i] + dp*dp*Measurement[mi]/p^2
                end
            end
        end

        cost_function = real(tr(grape.W*pinv(F_T)))
        coeff = [grape.W[para,para]/F_T[para,para] for para in 1:para_num] |>sum
        coeff = coeff^(-2)
        for cm in 1:ctrl_num
            for tm in 1:tnum
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

function auto_GRAPE_QFIM(grape, epsilon, max_episodes, Adam, save_file)
    println("quantum parameter estimation")
    ctrl_num = length(grape.control_Hamiltonian)
    ctrl_length = length(grape.control_coefficients[1])
    episodes = 1
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: auto-GRAPE")
        f_noctrl = QFI(grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, 
                        grape.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], grape.times)
        f_ini = QFI(grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, 
                        grape.control_Hamiltonian, grape.control_coefficients, grape.times)
        f_list = [f_ini]
        println("non-controlled QFI is $(f_noctrl)")
        println("initial QFI is $(f_ini)")
        if Adam == true
            gradient_QFI_Adam!(grape)
        else
            gradient_QFI!(grape)
        end
        if save_file == true
            SaveFile(f_ini, grape.control_coefficients)
            if Adam == true
                while true
                    f_now = QFI(grape)
                    if  abs(f_now - f_ini) < epsilon  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(f_now, grape.control_coefficients)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        SaveFile(f_now, grape.control_coefficients)
                        print("current QFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFI_Adam!(grape)
                end
            else
                while true
                    f_now = QFI(grape)
                    if  abs(f_now - f_ini) < epsilon  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(f_now, grape.control_coefficients)

                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        SaveFile(f_now, grape.control_coefficients)
                        print("current QFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFI!(grape)
                end
            end
        else
            if Adam == true
                while true
                    f_now = QFI(grape)
                    if  abs(f_now - f_ini) < epsilon  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile(f_list, grape.control_coefficients)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current QFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFI_Adam!(grape)
                end
            else
                while true
                    f_now = QFI(grape)
                    if  abs(f_now - f_ini) < epsilon  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile(f_list, grape.control_coefficients)
                        break
                    else
                        f_ini = f_now
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
        F_noctrl = QFIM(grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, 
                        grape.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], grape.times)
        f_noctrl = real(tr(grape.W*pinv(F_noctrl)))
        F_ini = QFI(grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, 
                        grape.control_Hamiltonian, grape.control_coefficients, grape.times)
        f_ini = real(tr(grape.W*pinv(F_ini)))
        f_list = [f_ini]
        println("non-controlled value of Tr(WF^{-1}) is $(f_noctrl)")
        println("initial value of Tr(WF^{-1}) is $(f_ini)")
        if Adam == true
            gradient_QFIM_Adam!(grape)
        else
            gradient_QFIM!(grape)
        end
        if save_file == true
            SaveFile(f_ini, grape.control_coefficients)
            if Adam == true
                while true
                    f_now = real(tr(grape.W*pinv(QFIM(grape))))
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(f_now, grape.control_coefficients)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile(f_now, grape.control_coefficients)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(grape)
                end
            else
                while true
                    f_now = real(tr(grape.W*pinv(QFIM(grape))))
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(f_now, grape.control_coefficients)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile(f_now, grape.control_coefficients)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFIM!(grape)
                end
            end
        else
            if Adam == true
                while true
                    f_now = real(tr(grape.W*pinv(QFIM(grape))))
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile(f_list, grape.control_coefficients)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(grape)
                end
            else
                while true
                    f_now = real(tr(grape.W*pinv(QFIM(grape))))
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile(f_list, grape.control_coefficients)
                        break
                    else
                        f_ini = f_now
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

function GRAPE_QFIM(grape, epsilon, max_episodes, Adam, save_file)
    println("quantum parameter estimation")
    ctrl_num = length(grape.control_Hamiltonian)
    ctrl_length = length(grape.control_coefficients[1])
    episodes = 1
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: GRAPE")
        f_noctrl = QFI(grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, 
                        grape.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], grape.times)
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
            SaveFile(f_ini, ctrl_pre)
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy_Adam(grape)
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes  
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(f_now, ctrl_pre) 
                        break 
                    else
                        f_ini = f_now
                        episodes += 1
                        SaveFile(f_now, ctrl_pre)
                        append!(f_list, f_now)
                        print("current QFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end  
                end 
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy(grape)
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes  
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(f_now, ctrl_pre) 
                        break 
                    else
                        f_ini = f_now
                        episodes += 1
                        SaveFile(f_now, ctrl_pre)
                        append!(f_list, f_now)
                        print("current QFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end  
                end 
            end                    
        else
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy_Adam(grape)
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile(f_list, ctrl_pre)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current QFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy(grape)
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile(f_list, ctrl_pre)
                        break
                    else
                        f_ini = f_now
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
        F_noctrl = QFIM(grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, 
                        grape.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], grape.times)
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
            SaveFile(f_ini, ctrl_pre)
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy_Adam(grape)
                    if  abs(f_now - f_ini) < epsilon  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(f_now, ctrl_pre)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        SaveFile(f_now, ctrl_pre)
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy(grape)
                    if  abs(f_now - f_ini) < epsilon  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(f_now, ctrl_pre)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        SaveFile(f_now, ctrl_pre)
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy_Adam(grape)
                    if  abs(f_now - f_ini) < epsilon  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile(f_list, ctrl_pre)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy(grape)
                    if  abs(f_now - f_ini) < epsilon  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile(f_list, ctrl_pre)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            end
        end
    end
end

function auto_GRAPE_CFIM(Measurement, grape, epsilon, max_episodes, Adam, save_file)
    println("classical parameter estimation")
    ctrl_num = length(grape.control_Hamiltonian)
    ctrl_length = length(grape.control_coefficients[1])
    episodes = 1
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: auto_GRAPE")
        f_noctrl = CFI(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, 
                        grape.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], grape.times)
        f_ini = CFI(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, 
                        grape.control_Hamiltonian, grape.control_coefficients, grape.times)
        f_list = [f_ini]
        println("non-controlled CFI is $(f_noctrl)")
        println("initial CFI is $(f_ini)")
        if Adam == true
            gradient_CFI_Adam!(grape, Measurement)
        else
            gradient_CFI!(grape, Measurement)
        end
        if save_file == true
            SaveFile(f_ini, grape.control_coefficients)
            if Adam == true
                while true
                    f_now = CFI(Measurement, grape)
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile(f_now, grape.control_coefficients)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        SaveFile(f_now, grape.control_coefficients)
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFI_Adam!(grape, Measurement)
                end
            else
                while true
                    f_now = CFI(Measurement, grape)
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile(f_now, grape.control_coefficients)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        SaveFile(f_now, grape.control_coefficients)
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFI!(grape, Measurement)
                end
            end
        else
            if Adam == true
                while true
                    f_now = CFI(Measurement, grape)
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile(f_list, grape.control_coefficients)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFI_Adam!(grape, Measurement)
                end
            else
                while true
                    f_now = CFI(Measurement, grape)
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile(f_list, grape.control_coefficients)
                        break
                    else
                        f_ini = f_now
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
        F_noctrl = CFIM(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, 
                        grape.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], grape.times)
        f_noctrl = real(tr(grape.W*pinv(F_noctrl)))
        F_ini = CFIM(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, 
                        grape.control_Hamiltonian, grape.control_coefficients, grape.times)
        f_ini = real(tr(grape.W*pinv(F_ini)))
        f_list = [f_ini]
        println("non-controlled value of Tr(WF^{-1}) is $(f_noctrl)")
        println("initial value of Tr(WF^{-1}) is $(f_ini)")
        if Adam == true
            gradient_CFIM_Adam!(grape, Measurement)
        else
            gradient_CFIM!(grape, Measurement)
        end
        if save_file == true
            SaveFile(f_ini, grape.control_coefficients)
            if Adam == true
                while true
                    f_now = real(tr(grape.W*pinv(CFIM(Measurement, grape))))
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(f_now, grape.control_coefficients)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        SaveFile(f_now, grape.control_coefficients)
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(grape, Measurement)
                end
            else
                while true
                    f_now = real(tr(grape.W*pinv(CFIM(Measurement, grape))))
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(f_now, grape.control_coefficients)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        SaveFile(f_now, grape.control_coefficients)
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFIM!(grape, Measurement)
                end
            end
        else
            if Adam == true
                while true
                    f_now = real(tr(grape.W*pinv(CFIM(Measurement, grape))))
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile(f_list, grape.control_coefficients)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(grape, Measurement)
                end
            else
                while true
                    f_now = real(tr(grape.W*pinv(CFIM(Measurement, grape))))
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile(f_list, grape.control_coefficients)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFIM!(grape, Measurement)
                end
            end
        end
    end
end

function GRAPE_CFIM(Measurement, grape, epsilon, max_episodes, Adam, save_file)
    println("classical parameter estimation")
    ctrl_num = length(grape.control_Hamiltonian)
    ctrl_length = length(grape.control_coefficients[1])
    episodes = 1
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: GRAPE")
        f_noctrl = CFI(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, 
                     grape.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], grape.times)
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
            SaveFile(f_ini, ctrl_pre) 
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy_Adam(Measurement, grape)
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes   
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile(f_now, ctrl_pre)                    
                        break 
                    else
                        f_ini = f_now
                        episodes += 1
                        SaveFile(f_now, ctrl_pre)
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end 
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy(Measurement, grape)
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes   
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile(f_now, ctrl_pre)                    
                        break 
                    else
                        f_ini = f_now
                        episodes += 1
                        SaveFile(f_now, ctrl_pre)
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end 
                end  
            end                     
        else
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy_Adam(Measurement, grape)
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile(f_list, ctrl_pre)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy(Measurement, grape)
                    if  abs(f_now - f_ini) < epsilon || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile(f_list, ctrl_pre)
                        break
                    else
                        f_ini = f_now
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
        F_noctrl = CFIM(Measurement, grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, 
                        grape.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], grape.times)
        f_noctrl = real(tr(grape.W*pinv(F_noctrl)))
        println("non-controlled value of Tr(WF^{-1}) is $(f_noctrl)")
        ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
        if Adam == true
            grape.control_coefficients, f_ini = gradient_CFIM_analy_Adam(Measurement, grape)
        else
            grape.control_coefficients, f_ini = gradient_CFIM_analy(Measurement, grape)
        end
        f_list = [f_ini]
        println("initial value of Tr(WF^{-1}) is $(f_ini)")
        if save_file == true
            SaveFile(f_ini, ctrl_pre)
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy_Adam(Measurement, grape)
                    if  abs(f_now - f_ini) < epsilon  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(f_now, ctrl_pre)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        SaveFile(f_now, ctrl_pre)
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy(Measurement, grape)
                    if  abs(f_now - f_ini) < epsilon  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(f_now, ctrl_pre)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        SaveFile(f_now, ctrl_pre)
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy_Adam(Measurement, grape)
                    if  abs(f_now - f_ini) < epsilon  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile(f_list, ctrl_pre)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy(Measurement, grape)
                    if  abs(f_now - f_ini) < epsilon  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile(f_list, ctrl_pre)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            end
        end
    end
end

function line_search(grape, f_pre, uk, p, δF, c1, c2, ctrl_total)
    #backtrack line search with wolfe conditions
    ϵ = 1.0
    uk_new = uk + ϵ*p
    δF_new = gradient_QFI_bfgs(grape, uk_new)
    δF = reshape(reduce(vcat,δF), 1, ctrl_total)
    p1 = reshape(reduce(vcat,p), ctrl_total, 1)
    while -QFI_bfgs(grape, uk_new) >= f_pre+(c1*ϵ*(δF*p1)[1]) || (reshape(reduce(vcat,δF_new), 1, ctrl_total)*p1)[1] <= (δF*p1)[1]
        ϵ = 0.5*ϵ
        uk_new = uk + ϵ*p
        δF_new = gradient_QFI_bfgs(grape, uk_new)
    end
    return uk_new, δF_new, ϵ
end

function BFGS(grape, B, c1, c2, epsilon, max_episodes)
    ctrl_length = length(grape.control_coefficients)
    cnum = length(grape.control_coefficients[1])
    ctrl_total = ctrl_length*cnum
    f_ini = QFI_bfgs(grape, grape.control_coefficients)
    f_list = [f_ini]
    println("initial QFI is $(f_ini)")
    δF = gradient_QFI_bfgs(grape, grape.control_coefficients)
    episodes = 1
    while true
        p = -B.*δF
        #line_search
        # grape.control_coefficients, δF_new, ϵ = line_search(grape, -f_ini, grape.control_coefficients, p, δF, c1, c2, ctrl_total)
        # y = δF_new - δF
        # x = ϵ*p
        #bfgs algorithm
        grape.control_coefficients = uk + p
        δF_new = gradient_QFI_bfgs(grape, grape.control_coefficients)
        y = δF_new - δF
        x = ϵ*p
        #update B
        if reshape(reduce(vcat,y), 1, ctrl_total)[1]*reshape(reduce(vcat,x), ctrl_total, 1)[1] > 0.0
            for i in 1:ctrl_length
                sk = reshape(x[i],cnum, 1)
                sk_T = reshape(x[i],1, cnum)
                yk = reshape(y[i],cnum, 1)
                yk_T = reshape(y[i],1, cnum)
                B[i] = B[i] - B[i]*sk*sk_T*B[i]/(sk_T*B[i]*sk)[1]+yk*yk_T/(yk_T*sk)[1]
            end
        end
        f_now = QFI_bfgs(grape, grape.control_coefficients)
        if abs(f_now - f_ini) < epsilon  || episodes >= max_episodes
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final QFI is ", f_now)
            SaveFile(f_list, grape.control_coefficients)
            break
        end
        f_ini = f_now
        δF = δF_new 
        episodes += 1
        append!(f_list, f_now)
        println("current QFI is $(f_ini)")
    end
end

function autoGRAPE_BFGS(grape, save_file, epsilon, max_episodes, B, c1, c2, e)
    println("auto-GRAPE with BFGS:")
    println("quantum parameter estimation")
    episodes = 1
    ctrl_length = length(grape.control_coefficients)
    cnum = length(grape.control_coefficients[1])
    ctrl_total = ctrl_length*cnum
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        f_ini = QFI(grape)
        f_list = [f_ini]
        println("initial QFI is $(f_ini)")
        if save_file == true
            while true
                gradient_QFI!(grape)
                f_now = QFI(grape)
                if abs(f_now - f_ini) < e
                    δF = gradient_QFI_bfgs(grape, grape.control_coefficients)  #merge into gradient
                    p = -B.*δF
                    grape.control_coefficients, δF_new, ϵ = line_search(grape, -f_ini, grape.control_coefficients, p, δF, c1, c2, ctrl_total)
                    y = δF_new - δF
                    x = ϵ*p
                    #update B
                    if reshape(reduce(vcat,y), 1, ctrl_total)[1]*reshape(reduce(vcat,x), ctrl_total, 1)[1] > 0.0
                        for i in 1:ctrl_length
                            sk = reshape(x[i],cnum, 1)
                            sk_T = reshape(x[i],1, cnum)
                            yk = reshape(y[i],cnum, 1)
                            yk_T = reshape(y[i],1, cnum)
                            B[i] = B[i] - B[i]*sk*sk_T*B[i]/(sk_T*B[i]*sk)[1]+yk*yk_T/(yk_T*sk)[1]
                        end
                    end
                    f_now = QFI_bfgs(grape, grape.control_coefficients)
                    if  abs(f_now - f_ini) < epsilon  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(f_now, grape.control_coefficients)
                        break
                    else
                        f_ini = f_now
                        δF = δF_new 
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile(f_now, grape.control_coefficients)
                        print("(BFGS) current QFI is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                elseif episodes >= max_episodes
                    print("\e[2K")
                    println("Iteration over, data saved.")
                    println("Final QFI is ", f_now)
                    SaveFile(f_now, grape.control_coefficients)
                    break
                else
                    f_ini = f_now
                    episodes += 1
                    append!(f_list, f_now)
                    SaveFile(f_now, grape.control_coefficients)
                    print("(GRAPE) current QFI is ", f_now, " ($(episodes-1) episodes)    \r")
                end
            end      
        else
            while true
                gradient_QFI!(grape)
                f_now = QFI(grape)

                if abs(f_now - f_ini) < e
                    δF = gradient_QFI_bfgs(grape, grape.control_coefficients)  #merge into gradient
                    p = -B.*δF
                    grape.control_coefficients, δF_new, ϵ = line_search(grape, -f_ini, grape.control_coefficients, p, δF, c1, c2, ctrl_total)
                    y = δF_new - δF
                    x = ϵ*p
                    #update B
                    if reshape(reduce(vcat,y), 1, ctrl_total)[1]*reshape(reduce(vcat,x), ctrl_total, 1)[1] > 0.0
                        for i in 1:ctrl_length
                            sk = reshape(x[i],cnum, 1)
                            sk_T = reshape(x[i],1, cnum)
                            yk = reshape(y[i],cnum, 1)
                            yk_T = reshape(y[i],1, cnum)
                            B[i] = B[i] - B[i]*sk*sk_T*B[i]/(sk_T*B[i]*sk)[1]+yk*yk_T/(yk_T*sk)[1]
                        end
                    end
                    f_now = QFI_bfgs(grape, grape.control_coefficients)
                    if  abs(f_now - f_ini) < epsilon  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(f_now, grape.control_coefficients)
                        break
                    else
                        f_ini = f_now
                        δF = δF_new 
                        episodes += 1
                        append!(f_list, f_now)
                        print("current QFI is ", f_now, " ($(episodes-1) episodes2)    \r")
                    end
                elseif episodes >= max_episodes
                    print("\e[2K")
                    println("Iteration over, data saved.")
                    println("Final QFI is ", f_now)
                    SaveFile(f_now, grape.control_coefficients)
                    break
                else
                    f_ini = f_now
                    episodes += 1
                    append!(f_list, f_now)
                    print("current QFI is ", f_now, " ($(episodes-1) episodes1)    \r")
                end
            end
        end
    end
end
