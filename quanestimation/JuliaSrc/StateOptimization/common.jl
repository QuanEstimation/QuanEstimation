function StateOpt_Adam(gt, t, para, m_t, v_t, ϵ, beta1, beta2, precision)
    t = t+1
    m_t = beta1*m_t + (1-beta1)*gt
    v_t = beta2*v_t + (1-beta2)*(gt*gt)
    m_cap = m_t/(1-(beta1^t))
    v_cap = v_t/(1-(beta2^t))
    para = para+(ϵ*m_cap)/(sqrt(v_cap)+precision)
    return para, m_t, v_t
end

function StateOpt_Adam!(system, δ)
    mt = system.mt
    vt = system.vt
    for ctrl in 1:length(δ)
        system.psi[ctrl], mt, vt = StateOpt_Adam(δ[ctrl], ctrl, system.psi[ctrl], mt, vt, system.ϵ, system.beta1, system.beta2, system.precision)
    end
end
