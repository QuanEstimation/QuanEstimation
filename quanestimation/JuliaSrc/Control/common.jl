function Adam(gt, t, para, m_t, v_t, alpha=0.01, beta1=0.90, beta2=0.99, epsilon=1e-8)
    t = t+1
    m_t = beta1*m_t + (1-beta1)*gt
    v_t = beta2*v_t + (1-beta2)*(gt*gt)
    m_cap = m_t/(1-(beta1^t))
    v_cap = v_t/(1-(beta2^t))
    para = para+(alpha*m_cap)/(sqrt(v_cap)+epsilon)
    return para, m_t, v_t
end

function Adam!(system, δ, mt_in=0.0, vt_in=0.0)
    ctrl_length = length(system.control_coefficients[1])
    for ctrl in 1:length(δ)
        mt = mt_in
        vt = vt_in
        for ti in 1:ctrl_length
            system.control_coefficients[ctrl][ti], mt, vt = Adam(δ[ctrl][ti], ti, system.control_coefficients[ctrl][ti], mt, vt)
        end
    end
end