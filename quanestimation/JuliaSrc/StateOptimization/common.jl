mutable struct TimeIndepend_noiseless{T <: Complex,M <: Real}
    freeHamiltonian
    Hamiltonian_derivative::Vector{Matrix{T}}
    psi::Vector{T}
    tspan::Vector{M}
    W::Matrix{M}
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    TimeIndepend_noiseless(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, psi::Vector{T},
                 tspan::Vector{M}, W::Matrix{M},
                 ρ=Vector{Matrix{T}}(undef, 1), ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1),∂ρ_∂V=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = 
                 new{T,M}(freeHamiltonian, Hamiltonian_derivative, psi, tspan, W, ρ, ∂ρ_∂x) 
end

mutable struct TimeIndepend_noise{T <: Complex,M <: Real}
    freeHamiltonian
    Hamiltonian_derivative::Vector{Matrix{T}}
    psi::Vector{T}
    tspan::Vector{M}
    Decay_opt::Vector{Matrix{T}}
    γ::Vector{M}
    W::Matrix{M}
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    TimeIndepend_noise(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, psi::Vector{T},
                 tspan::Vector{M}, Decay_opt::Vector{Matrix{T}},γ::Vector{M}, W::Matrix{M}, 
                 ρ=Vector{Matrix{T}}(undef, 1), ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1),∂ρ_∂V=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = 
                 new{T,M}(freeHamiltonian, Hamiltonian_derivative, psi, tspan, Decay_opt, γ, W, ρ, ∂ρ_∂x) 
end

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

function SaveFile_state(f_now::Float64, control)
    open("f.csv","a") do f
        writedlm(f, [f_now])
    end
    open("states.csv","a") do g
        writedlm(g, [control])
    end
end

function SaveFile_state(f_now::Vector{Float64}, control)
    open("f.csv","w") do f
        writedlm(f, f_now)
    end
    open("states.csv","w") do g
        writedlm(g, [control])
    end
end

function SaveFile_state_ddpg(f_now::Float64, reward::Float64, control)
    open("f.csv","a") do f
        writedlm(f, [f_now])
    end
    open("total_reward.csv","w") do m
        writedlm(m, [reward])
    end
    open("states.csv","a") do g
        writedlm(g, [control])
    end
end

function SaveFile_state_ddpg(f_now::Vector{Float64}, reward::Vector{Float64}, control)
    open("f.csv","w") do f
        writedlm(f, f_now)
    end
    open("total_reward.csv","w") do m
        writedlm(m, reward)
    end
    open("states.csv","w") do g
        writedlm(g, [control])
    end
end