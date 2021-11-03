function StateOpt_Adam(gt, t, para, m_t, v_t, ϵ, beta1, beta2, precision)
    t = t+1
    m_t = beta1*m_t + (1-beta1)*gt
    v_t = beta2*v_t + (1-beta2)*(gt*gt)
    m_cap = m_t/(1-(beta1^t))
    v_cap = v_t/(1-(beta2^t))
    para = para+(ϵ*m_cap)/(sqrt(v_cap)+precision)
    return para, m_t, v_t
end

function StateOpt_Adam!(system, δ, ϵ, mt, vt, beta1, beta2, precision)
    for ctrl in 1:length(δ)
        system.psi[ctrl], mt, vt = StateOpt_Adam(δ[ctrl], ctrl, system.psi[ctrl], mt, vt, ϵ, beta1, beta2, precision)
    end
end

function SaveFile_ad(dim, f_now::Float64, control)
    open("f_ad_N$(dim-1).csv","a") do f
        writedlm(f, [f_now])
    end
    open("state_ad_N$(dim-1).csv","a") do g
        writedlm(g, control)
    end
end

function SaveFile_ad(dim, f_now::Vector{Float64}, control)
    open("f_ad_N$(dim-1).csv","w") do f
        writedlm(f, f_now)
    end
    open("state_ad_N$(dim-1).csv","w") do g
        writedlm(g, control)
    end
end

function SaveFile_pso(dim, f_now::Float64, control)
    open("f_pso_N$(dim-1).csv","a") do f
        writedlm(f, [f_now])
    end
    open("state_pso_N$(dim-1).csv","a") do g
        writedlm(g, control)
    end
end

function SaveFile_pso(dim, f_now::Vector{Float64}, control)
    open("f_pso_N$(dim-1).csv","w") do f
        writedlm(f, f_now)
    end
    open("state_pso_N$(dim-1).csv","w") do g
        writedlm(g, control)
    end
end

function SaveFile_de(dim, f_now::Float64, control)
    open("f_de_N$(dim-1).csv","a") do f
        writedlm(f, [f_now])
    end
    open("state_de_N$(dim-1).csv","a") do g
        writedlm(g, control)
    end
end

function SaveFile_de(dim, f_now::Vector{Float64}, control)
    open("f_de_N$(dim-1).csv","w") do f
        writedlm(f, f_now)
    end
    open("state_de_N$(dim-1).csv","w") do g
        writedlm(g, control)
    end
end

function SaveFile_nm(dim, f_now::Float64, control)
    open("f_nm_N$(dim-1).csv","a") do f
        writedlm(f, [f_now])
    end
    open("state_nm_N$(dim-1).csv","a") do g
        writedlm(g, control)
    end
end

function SaveFile_nm(dim, f_now::Vector{Float64}, control)
    open("f_nm_N$(dim-1).csv","w") do f
        writedlm(f, f_now)
    end
    open("state_nm_N$(dim-1).csv","w") do g
        writedlm(g, control)
    end
end