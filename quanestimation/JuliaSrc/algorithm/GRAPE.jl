"""
	update!(opt::ControlOpt, alg::AbstractGRAPE, obj::QFIM_obj, dynamics, output)

Control optimization with GRAPE, where the objective is QFIM.

"""
function update!(opt::ControlOpt, alg::AbstractGRAPE, obj::QFIM_obj, dynamics, output)
    (; max_episode) = alg
    ctrl_length = length(dynamics.data.ctrl[1])
    ctrl_num = length(dynamics.data.Hc)
    
    dynamics_copy = set_ctrl(dynamics, [zeros(ctrl_length) for i in 1:ctrl_num])
    f_noctrl, f_comp = objective(obj, dynamics_copy)
    f_ini, f_comp = objective(obj, dynamics)
    set_f!(output, f_ini)
    set_buffer!(output, dynamics.data.ctrl)
    set_io!(output, f_noctrl, f_ini)
    show(opt, output, obj)
    for ei in 1:(max_episode-1)
        gradient_QFIM_analy(opt, alg, obj, dynamics)
        bound!(dynamics.data.ctrl, opt.ctrl_bound)
        f_out, f_now = objective(obj, dynamics)
        set_f!(output, f_out)
        set_buffer!(output, dynamics.data.ctrl)
        set_io!(output, f_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

"""
Control optimization with GRAPE, where the objective is CFIM.

"""
function update!(opt::ControlOpt, alg::AbstractGRAPE, obj::CFIM_obj, dynamics, output)
    (; max_episode) = alg
    ctrl_length = length(dynamics.data.ctrl[1])
    ctrl_num = length(dynamics.data.Hc)
    
    dynamics_copy = set_ctrl(dynamics, [zeros(ctrl_length) for i in 1:ctrl_num])
    f_noctrl, f_comp = objective(obj, dynamics_copy)
    f_ini, f_comp = objective(obj, dynamics)
    set_f!(output, f_ini)
    set_buffer!(output, dynamics.data.ctrl)
    set_io!(output, f_noctrl, f_ini)
    show(opt, output, obj)
    for ei in 1:(max_episode-1)
        gradient_CFIM_analy(opt, alg, obj, dynamics)
        bound!(dynamics.data.ctrl, opt.ctrl_bound)
        f_out, f_now = objective(obj, dynamics)
        set_f!(output, f_out)
        set_buffer!(output, dynamics.data.ctrl)
        set_io!(output, f_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

function dynamics_analy(dynamics, dim, tnum, para_num, ctrl_num)
    Δt = dynamics.data.tspan[2] - dynamics.data.tspan[1]
    H = Htot(dynamics.data.H0, dynamics.data.Hc, dynamics.data.ctrl)

    ρt = [Vector{ComplexF64}(undef, dim^2) for i in 1:tnum]
    ∂ρt_∂x = [[Vector{ComplexF64}(undef, dim^2) for para in 1:para_num] for i in 1:tnum]
    δρt_δV = [[] for i in 1:ctrl_num]
    ∂xδρt_δV = [[[] for i in 1:ctrl_num] for i in 1:para_num]
    ∂H_L = [Matrix{ComplexF64}(undef, dim^2,dim^2) for i in 1:para_num]
    Hc_L = [Matrix{ComplexF64}(undef, dim^2,dim^2) for i in 1:ctrl_num]

    ρt[1] = dynamics.data.ρ0 |> vec
    for cj in 1:ctrl_num
        Hc_L[cj] = liouville_commu(dynamics.data.Hc[cj])
        append!(δρt_δV[cj], [-im*Δt*Hc_L[cj]*ρt[1]])
    end

    for pj in 1:para_num
        ∂ρt_∂x[1][pj] = ρt[1] |> zero
        ∂H_L[pj] = liouville_commu(dynamics.data.dH[pj])
        for ci in 1:ctrl_num
            append!(∂xδρt_δV[pj][ci], [-im*Δt*Hc_L[ci]*∂ρt_∂x[1][pj]])
        end
    end

    for ti in 2:tnum
        exp_L = expL(H[ti-1], dynamics.data.decay_opt, dynamics.data.γ, Δt, ti)
        ρt[ti] = exp_L * ρt[ti-1]
        for pk in 1:para_num
            ∂ρt_∂x[ti][pk] = -im * Δt * ∂H_L[pk] * ρt[ti] + exp_L * ∂ρt_∂x[ti-1][pk]
        end
        
        for ck in 1:ctrl_num
            for tk in 1:(ti-1)
                δρt_δV_first = popfirst!(δρt_δV[ck])
                δρt_δV_tp = exp_L * δρt_δV_first
                append!(δρt_δV[ck], [δρt_δV_tp])
                for pk in 1:para_num
                    ∂xδρt_δV_first = popfirst!(∂xδρt_δV[pk][ck])
                    ∂xδρt_δV_tp = -im * Δt * ∂H_L[pk] * exp_L * δρt_δV_first + exp_L * ∂xδρt_δV_first
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

function gradient_QFIM_analy(opt, alg::GRAPE_Adam, obj, dynamics)
    dim = size(dynamics.data.ρ0)[1]
    tnum = length(dynamics.data.tspan)
    para_num = length(dynamics.data.dH)
    ctrl_num = length(dynamics.data.Hc)
    
    ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV = dynamics_analy(dynamics, dim, tnum, para_num, ctrl_num)

    Lx = SLD(ρt_T, ∂ρt_T; eps=obj.eps)
    F_T = QFIM_SLD(ρt_T, ∂ρt_T; eps=obj.eps)

    if para_num == 1
        cost_function = F_T[1]
        anti_commu = 2*Lx[1]*Lx[1]
        for cm in 1:ctrl_num
            mt, vt = 0.0, 0.0
            for tm in 1:(tnum-1)
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                ∂xδρt_T_δV = ∂xδρt_δV[1][cm][tm] |> vec2mat
                term1 = tr(∂xδρt_T_δV*Lx[1])
                term2 = tr(∂ρt_T_δV*anti_commu)
                δF = ((2*term1-0.5*term2) |> real)
                dynamics.data.ctrl[cm][tm], mt, vt = Adam(δF, tm, dynamics.data.ctrl[cm][tm], mt, vt, alg.epsilon, alg.beta1, alg.beta2, obj.eps)
            end
        end
    elseif para_num == 2
        coeff1 = real(det(F))
        coeff2 = obj.W[1,1]*F_T[2,2]+obj.W[2,2]*F_T[1,1]-obj.W[1,2]*F_T[2,1]-obj.W[2,1]*F_T[1,2]
        cost_function = (abs(det(F_T)) < obj.eps ? (1.0/obj.eps) : real(tr(obj.W*inv(F_T))))
        for cm in 1:ctrl_num
            mt, vt = 0.0, 0.0
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
                item2 = (obj.W[1,1]*δF_all[2][2]+obj.W[2,2]*δF_all[1][1]-obj.W[1,2]*δF_all[2][1]-obj.W[2,1]*δF_all[1][2])/coeff1
                δF = -(item1+item2)*cost_function^2
                dynamics.data.ctrl[cm][tm], mt, vt = Adam(δF, tm, dynamics.data.ctrl[cm][tm], mt, vt, alg.epsilon, alg.beta1, alg.beta2, obj.eps)
            end
        end
    else       
        cost_function = (abs(det(F_T)) < obj.eps ? (1.0/obj.eps) : real(tr(obj.W*inv(F_T))))
        coeff = [obj.W[para,para]/F_T[para,para] for para in 1:para_num] |>sum
        coeff = coeff^(-2)
        for cm in 1:ctrl_num
            mt, vt = 0.0, 0.0
            for tm in 1:(tnum-1)
                δF = 0.0
                for pm in 1:para_num
                    ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                    ∂xδρt_T_δV = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                    term1 = tr(∂xδρt_T_δV * Lx[pm])
                    anti_commu = 2 * Lx[pm] * Lx[pm]
                    term2 = tr(∂ρt_T_δV * anti_commu)
                    δF = δF + obj.W[pm,pm]*(1.0/F_T[pm,pm]/F_T[pm,pm])*((2*term1-0.5*term2) |> real)
                end
                δF = δF*coeff
                dynamics.data.ctrl[cm][tm], mt, vt = Adam(δF, tm, dynamics.data.ctrl[cm][tm], mt, vt, alg.epsilon, alg.beta1, alg.beta2, obj.eps)
            end
        end
    end
end

function gradient_QFIM_analy(opt, alg::GRAPE, obj, dynamics)
    dim = size(dynamics.data.ρ0)[1]
    tnum = length(dynamics.data.tspan)
    para_num = length(dynamics.data.dH)
    ctrl_num = length(dynamics.data.Hc)
    
    ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV = dynamics_analy(dynamics, dim, tnum, para_num, ctrl_num)

    Lx = SLD(ρt_T, ∂ρt_T; eps=obj.eps)
    F_T = QFIM_SLD(ρt_T, ∂ρt_T; eps=obj.eps)

    cost_function = F_T[1]
    
    if para_num == 1
        anti_commu = 2*Lx[1]*Lx[1]
        for cm in 1:ctrl_num
            for tm in 1:(tnum-1)
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                ∂xδρt_T_δV = ∂xδρt_δV[1][cm][tm] |> vec2mat
                term1 = tr(∂xδρt_T_δV*Lx[1])
                term2 = tr(∂ρt_T_δV*anti_commu)
                δF = ((2*term1-0.5*term2) |> real)
                dynamics.data.ctrl[cm][tm] = dynamics.data.ctrl[cm][tm] + alg.epsilon*δF
            end
        end
    elseif para_num == 2
        coeff1 = real(det(F))
        coeff2 = obj.W[1,1]*F_T[2,2]+obj.W[2,2]*F_T[1,1]-obj.W[1,2]*F_T[2,1]-obj.W[2,1]*F_T[1,2]
        cost_function = (abs(det(F_T)) < obj.eps ? (1.0/obj.eps) : real(tr(obj.W*inv(F_T))))
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
                item2 = (obj.W[1,1]*δF_all[2][2]+obj.W[2,2]*δF_all[1][1]-obj.W[1,2]*δF_all[2][1]-obj.W[2,1]*δF_all[1][2])/coeff1
                δF = -(item1+item2)*cost_function^2
                dynamics.data.ctrl[cm][tm] = dynamics.data.ctrl[cm][tm] + alg.epsilon*δF
            end
        end
    else
        cost_function = (abs(det(F_T)) < obj.eps ? (1.0/obj.eps) : real(tr(obj.W*inv(F_T))))
        coeff = [obj.W[para,para]/F_T[para,para] for para in 1:para_num] |>sum
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
                    δF = δF + obj.W[pm,pm]*(1.0/F_T[pm,pm]/F_T[pm,pm])*((2*term1-0.5*term2) |> real)
                end
                δF = δF*coeff
                dynamics.data.ctrl[cm][tm] = dynamics.data.ctrl[cm][tm] + alg.epsilon*δF
            end
        end
    end
end

function gradient_CFIM_analy_Adam(opt, alg, obj, dynamics)
    dim = size(dynamics.data.ρ0)[1]
    tnum = length(dynamics.data.tspan)
    para_num = length(dynamics.data.dH)
    ctrl_num = length(dynamics.data.Hc)
    
    ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV = dynamics_analy(dynamics, dim, tnum, para_num, ctrl_num)

    if para_num == 1
        F_T = CFIM(ρt_T, ∂ρt_T[1], obj.M; eps=obj.eps)
        cost_function = F_T
        L1_tidle = zeros(ComplexF64, dim, dim)
        L2_tidle = zeros(ComplexF64, dim, dim)

        for mi in 1:dim
            p = (tr(ρt_T*obj.M[mi]) |> real)
            dp = (tr(∂ρt_T[1]*obj.M[mi]) |> real)
            if p > obj.eps
                L1_tidle = L1_tidle + dp*obj.M[mi]/p
                L2_tidle = L2_tidle + dp*dp*obj.M[mi]/p^2
            end
        end

        for cm in 1:ctrl_num
            mt, vt = 0.0, 0.0
            for tm in 1:(tnum-1)
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                ∂xδρt_T_δV = ∂xδρt_δV[1][cm][tm] |> vec2mat
                term1 = tr(∂xδρt_T_δV*L1_tidle)
                term2 = tr(∂ρt_T_δV*L2_tidle)
                δF = ((2*term1-term2) |> real)
                dynamics.data.ctrl[cm][tm], mt, vt = Adam(δF, tm, dynamics.data.ctrl[cm][tm], mt, vt, alg.epsilon, alg.beta1, alg.beta2, obj.eps)
            end
        end
    elseif para_num == 2
        F_T = CFIM(ρt_T, ∂ρt_T, obj.M; eps=obj.eps)
        L1_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
        L2_tidle = [[zeros(ComplexF64, dim, dim) for i in 1:para_num] for j in 1:para_num]
    
        for para_i in 1:para_num
            for mi in 1:dim
                p = (tr(ρt_T*obj.M[mi]) |> real)
                dp = (tr(∂ρt_T[para_i]*obj.M[mi]) |> real)
                if p > obj.eps
                    L1_tidle[para_i] = L1_tidle[para_i] + dp*obj.M[mi]/p
                end
            end
        end
    
        for para_i in 1:para_num
            dp_a = (tr(∂ρt_T[para_i]*obj.M[mi]) |> real)
            for para_j in 1:para_num
                dp_b = (tr(∂ρt_T[para_j]*obj.M[mi]) |> real)
                for mi in 1:dim
                    p = (tr(ρt_T*obj.M[mi]) |> real)
                    if p > obj.eps
                        L2_tidle[para_i][para_j] = L2_tidle[para_i][para_j] + dp_a*dp_b*obj.M[mi]/p^2
                    end
                end
            end
        end
        coeff1 = real(det(F))
        coeff2 = obj.W[1,1]*F_T[2,2]+obj.W[2,2]*F_T[1,1]-obj.W[1,2]*F_T[2,1]-obj.W[2,1]*F_T[1,2]
        cost_function = (abs(det(F_T)) < obj.eps ? (1.0/obj.eps) : real(tr(obj.W*inv(F_T))))
        for cm in 1:ctrl_num
            mt, vt = 0.0, 0.0
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
                item2 = (obj.W[1,1]*δF_all[2][2]+obj.W[2,2]*δF_all[1][1]-obj.W[1,2]*δF_all[2][1]-obj.W[2,1]*δF_all[1][2])/coeff1
                δF = -(item1+item2)*cost_function^2
                dynamics.data.ctrl[cm][tm], mt, vt = Adam(δF, tm, dynamics.data.ctrl[cm][tm], mt, vt, alg.epsilon, alg.beta1, alg.beta2, obj.eps)
            end
        end
    else
        F_T = CFIM(ρt_T, ∂ρt_T, obj.M; eps=obj.eps)
        L1_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
        L2_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
    
        for para_i in 1:para_num
            for mi in 1:dim
                p = (tr(ρt_T*obj.M[mi]) |> real)
                dp = (tr(∂ρt_T[para_i]*obj.M[mi]) |> real)
                if p > obj.eps
                    L1_tidle[para_i] = L1_tidle[para_i] + dp*obj.M[mi]/p
                    L2_tidle[para_i] = L2_tidle[para_i] + dp*dp*obj.M[mi]/p^2
                end
            end
        end

        cost_function = (abs(det(F_T)) < obj.eps ? (1.0/obj.eps) : real(tr(obj.W*inv(F_T))))
        coeff = [obj.W[para,para]/F_T[para,para] for para in 1:para_num] |>sum
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
                    δF = δF + obj.W[pm,pm]*(1.0/F_T[pm,pm]/F_T[pm,pm])*((2*term1-term2) |> real)
                end
                δF = δF*coeff
                dynamics.data.ctrl[cm][tm], mt, vt = Adam(δF, tm, dynamics.data.ctrl[cm][tm], mt, vt, alg.epsilon, alg.beta1, alg.beta2, obj.eps)
            end
        end
    end
end

function gradient_CFIM_analy(opt, alg, obj, dynamics)
    dim = size(dynamics.data.ρ0)[1]
    tnum = length(dynamics.data.tspan)
    para_num = length(dynamics.data.dH)
    ctrl_num = length(dynamics.data.Hc)
    
    ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV = dynamics_analy(dynamics, dim, tnum, para_num, ctrl_num)

    if para_num == 1
        F_T = CFIM(ρt_T, ∂ρt_T[1], obj.M; eps=obj.eps)
        cost_function = F_T
        L1_tidle = zeros(ComplexF64, dim, dim)
        L2_tidle = zeros(ComplexF64, dim, dim)

        for mi in 1:dim
            p = (tr(ρt_T*obj.M[mi]) |> real)
            dp = (tr(∂ρt_T[1]*obj.M[mi]) |> real)
            if p > obj.eps
                L1_tidle = L1_tidle + dp*obj.M[mi]/p
                L2_tidle = L2_tidle + dp*dp*obj.M[mi]/p^2
            end
        end

        for cm in 1:ctrl_num
            for tm in 1:(tnum-1)
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                ∂xδρt_T_δV = ∂xδρt_δV[1][cm][tm] |> vec2mat
                term1 = tr(∂xδρt_T_δV*L1_tidle)
                term2 = tr(∂ρt_T_δV*L2_tidle)
                δF = ((2*term1-term2) |> real)
                dynamics.data.ctrl[cm][tm] = dynamics.data.ctrl[cm][tm] + alg.epsilon*δF
            end
        end
    elseif para_num == 2
        F_T = CFIM(ρt_T, ∂ρt_T, obj.M; eps=obj.eps)
        L1_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
        L2_tidle = [[zeros(ComplexF64, dim, dim) for i in 1:para_num] for j in 1:para_num]
    
        for para_i in 1:para_num
            for mi in 1:dim
                p = (tr(ρt_T*obj.M[mi]) |> real)
                dp = (tr(∂ρt_T[para_i]*obj.M[mi]) |> real)
                if p > obj.eps
                    L1_tidle[para_i] = L1_tidle[para_i] + dp*obj.M[mi]/p
                end
            end
        end
    
        for para_i in 1:para_num
            dp_a = (tr(∂ρt_T[para_i]*obj.M[mi]) |> real)
            for para_j in 1:para_num
                dp_b = (tr(∂ρt_T[para_j]*obj.M[mi]) |> real)
                for mi in 1:dim
                    p = (tr(ρt_T*obj.M[mi]) |> real)
                    if p > obj.eps
                        L2_tidle[para_i][para_j] = L2_tidle[para_i][para_j] + dp_a*dp_b*obj.M[mi]/p^2
                    end
                end
            end
        end
        coeff1 = real(det(F))
        coeff2 = obj.W[1,1]*F_T[2,2]+obj.W[2,2]*F_T[1,1]-obj.W[1,2]*F_T[2,1]-obj.W[2,1]*F_T[1,2]
        cost_function = (abs(det(F_T)) < obj.eps ? (1.0/obj.eps) : real(tr(obj.W*inv(F_T))))
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
                item2 = (obj.W[1,1]*δF_all[2][2]+obj.W[2,2]*δF_all[1][1]-obj.W[1,2]*δF_all[2][1]-obj.W[2,1]*δF_all[1][2])/coeff1
                δF = -(item1+item2)*cost_function^2
                dynamics.data.ctrl[cm][tm] = dynamics.data.ctrl[cm][tm] + alg.epsilon*δF
            end
        end
    else
        F_T = CFIM(ρt_T, ∂ρt_T, obj.M; eps=obj.eps)
        L1_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
        L2_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
    
        for para_i in 1:para_num
            for mi in 1:dim
                p = (tr(ρt_T*obj.M[mi]) |> real)
                dp = (tr(∂ρt_T[para_i]*obj.M[mi]) |> real)
                if p > obj.eps
                    L1_tidle[para_i] = L1_tidle[para_i] + dp*obj.M[mi]/p
                    L2_tidle[para_i] = L2_tidle[para_i] + dp*dp*obj.M[mi]/p^2
                end
            end
        end

        cost_function = (abs(det(F_T)) < obj.eps ? (1.0/obj.eps) : real(tr(obj.W*inv(F_T))))
        coeff = [obj.W[para,para]/F_T[para,para] for para in 1:para_num] |>sum
        coeff = coeff^(-2)
        for cm in 1:ctrl_num
            for tm in 1:(tnum-1)
                δF = 0.0
                for pm in 1:para_num
                    ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                    ∂xδρt_T_δV = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                    term1 = tr(∂xδρt_T_δV * L1_tidle[pm])
                    term2 = tr(∂ρt_T_δV * L2_tidle[pm])
                    δF = δF + obj.W[pm,pm]*(1.0/F_T[pm,pm]/F_T[pm,pm])*((2*term1-term2) |> real)
                end
                δF = δF*coeff
                dynamics.data.ctrl[cm][tm] = dynamics.data.ctrl[cm][tm] + alg.epsilon*δF
            end
        end
    end
end
