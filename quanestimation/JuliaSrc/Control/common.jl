function Adam(gt, t, para, m_t, v_t, ϵ, beta1, beta2, precision)
    t = t+1
    m_t = beta1*m_t + (1-beta1)*gt
    v_t = beta2*v_t + (1-beta2)*(gt*gt)
    m_cap = m_t/(1-(beta1^t))
    v_cap = v_t/(1-(beta2^t))
    para = para+(ϵ*m_cap)/(sqrt(v_cap)+precision)
    return para, m_t, v_t
end

function Adam!(system, δ)
    ctrl_length = length(system.control_coefficients[1])
    for ctrl in 1:length(δ)
        mt = system.mt
        vt = system.vt
        for ti in 1:ctrl_length
            system.control_coefficients[ctrl][ti], mt, vt = Adam(δ[ctrl][ti], ti, system.control_coefficients[ctrl][ti], mt, vt, system.ϵ, system.beta1, system.beta2, system.precision)
        end
    end
end

function ctrl_bound(system)
    ctrl_num = length(system.control_coefficients)
    ctrl_length = length(system.control_coefficients[1])
    for ck in 1:ctrl_num
        for tk in 1:ctrl_length
            system.control_coefficients[ck][tk] = (x-> (x|>abs) < system.ctrl_max ? x : system.ctrl_max)(system.control_coefficients[ck][tk])
        end 
    end
end