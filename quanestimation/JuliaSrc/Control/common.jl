function Adam(gt, t, para, m_t, v_t, ϵ, beta1, beta2, accuracy=1e-8)
    t = t+1
    m_t = beta1*m_t + (1-beta1)*gt
    v_t = beta2*v_t + (1-beta2)*(gt*gt)
    m_cap = m_t/(1-(beta1^t))
    v_cap = v_t/(1-(beta2^t))
    para = para+(ϵ*m_cap)/(sqrt(v_cap)+accuracy)
    return para, m_t, v_t
end

function Adam!(system, δ)
    ctrl_length = length(system.control_coefficients[1])
    for ctrl in 1:length(δ)
        mt = system.mt
        vt = system.vt
        for ti in 1:ctrl_length
            system.control_coefficients[ctrl][ti], mt, vt = Adam(δ[ctrl][ti], ti, system.control_coefficients[ctrl][ti], mt, vt, system.ϵ, system.beta1, system.beta2, system.accuracy)
        end
    end
end

function bound!(A::Array, bound)
    for a in A
        if a|>abs >=bound
            a = 0. #bound
        end
    end
end

function bound!(control_coefficients::Vector{Vector{Float64}}, ctrl_bound)
    ctrl_num = length(control_coefficients)
    ctrl_length = length(control_coefficients[1])
    for ck in 1:ctrl_num
        for tk in 1:ctrl_length
            control_coefficients[ck][tk] = (x-> x < ctrl_bound[1] ? ctrl_bound[1] : x > ctrl_bound[2] ? ctrl_bound[2] : x)(control_coefficients[ck][tk])
        end 
    end
end

function bound!(control_coefficients::Vector{Float64}, ctrl_bound)
    ctrl_num = length(control_coefficients)
    for ck in 1:ctrl_num
        control_coefficients[ck] = (x-> x < ctrl_bound[1] ? ctrl_bound[1] : x > ctrl_bound[2] ? ctrl_bound[2] : x)(control_coefficients[ck])
    end
end

function SaveFile_ctrl(f_now::Float64, control)
    open("f.csv","a") do f
        writedlm(f, [f_now])
    end
    open("controls.csv","a") do g
        writedlm(g, control)
    end
end

function SaveFile_ctrl(f_now::Vector{Float64}, control)
    open("f.csv","w") do f
        writedlm(f, f_now)
    end
    open("controls.csv","w") do g
        writedlm(g, control)
    end
end

function SaveFile_ddpg(f_now::Float64, reward::Float64, control)
    open("f.csv","a") do f
        writedlm(f, [f_now])
    end
    open("total_reward.csv","a") do m
        writedlm(m, [reward])
    end
    open("controls.csv","a") do g
        writedlm(g, control)
    end
end

function SaveFile_ddpg(f_now::Vector{Float64}, reward::Vector{Float64}, control)
    open("f.csv","w") do f
        writedlm(f, f_now)
    end
    open("total_reward.csv","w") do m
        writedlm(m, reward)
    end
    open("controls.csv","w") do g
        writedlm(g, control)
    end
end