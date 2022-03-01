mutable struct PSO_Copt{T <: Complex,M <: Real} <: ControlSystem
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
    eps::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    PSO_Copt(freeHamiltonian, Hamiltonian_derivative::Vector{Matrix{T}}, ρ0::Matrix{T},tspan::Vector{M}, decay_opt::Vector{Matrix{T}},
        γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{M}}, ctrl_bound::Vector{M},
        W::Matrix{M}, eps::M, ρ=Vector{Matrix{T}}(undef, 1), ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef,1)) where {T<:Complex,M<:Real}= 
        new{T,M}(freeHamiltonian, Hamiltonian_derivative, ρ0, tspan, decay_opt, γ, control_Hamiltonian, control_coefficients,
                 ctrl_bound, W, eps, ρ, ∂ρ_∂x) 
end

function QFIM_PSO_Copt(pso::PSO_Copt{T}, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file) where {T<: Complex}
    sym = Symbol(QFIM)
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    M = [zeros(ComplexF64, size(pso.ρ0)[1], size(pso.ρ0)[1])]
    return info_PSO_Copt(M, pso, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file, sym, str1, str2, str3)
end

function CFIM_PSO_Copt(M, pso::PSO_Copt{T}, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file) where {T<: Complex}
    sym = Symbol(CFIM)
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_PSO_Copt(M, pso, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file, sym, str1, str2, str3)
end

function HCRB_PSO_Copt(pso::PSO_Copt{T}, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file) where {T<: Complex}
    sym = Symbol("HCRB")
    str1 = ""
    str2 = "HCRB"
    str3 = "HCRB"
    M = [zeros(ComplexF64, size(pso.ρ0)[1], size(pso.ρ0)[1])]
    if length(pso.Hamiltonian_derivative) == 1
        println("In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the objection function for control optimization.")
        return nothing
    else
        return info_PSO_Copt(M, pso, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file, sym, str1, str2, str3)
    end
end

function info_PSO_Copt(M, pso, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file, sym, str1, str2, str3) where {T<:Complex}
    println("$str1 parameter estimation")
    Random.seed!(seed)
    ctrl_length = length(pso.control_coefficients[1])
    ctrl_num = length(pso.control_Hamiltonian)
    particles = repeat(pso, particle_num)
    if pso.ctrl_bound[1] == -Inf || pso.ctrl_bound[2] == Inf
        velocity = 0.1*(2.0*rand(ctrl_num, ctrl_length, particle_num)-ones(ctrl_num, ctrl_length, particle_num))
    else
        a = pso.ctrl_bound[1]
        b = pso.ctrl_bound[2]
        velocity = 0.1*((b-a)*rand(ctrl_num, ctrl_length, particle_num)+a*ones(ctrl_num, ctrl_length, particle_num))
    end
    pbest = zeros(ctrl_num, ctrl_length, particle_num)
    gbest = zeros(ctrl_num, ctrl_length)
    p_fit = zeros(particle_num)
    f_noctrl = obj_func(Val{sym}(), pso, M, [zeros(ctrl_length) for i in 1:ctrl_num])
    f_ini = obj_func(Val{sym}(), pso, M)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    # initialize
    if length(ini_particle) > particle_num
        ini_particle = [ini_particle[i] for i in 1:particle_num]
    end
    for pj in 1:length(ini_particle)
        particles[pj].control_coefficients = [[ini_particle[pj][i,j] for j in 1:ctrl_length] for i in 1:ctrl_num]
    end
    if pso.ctrl_bound[1] == -Inf || pso.ctrl_bound[2] == Inf
        for pj in (length(ini_particle)+1):particle_num
            particles[pj].control_coefficients = [[2*rand()-1.0 for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
    else
        a = pso.ctrl_bound[1]
        b = pso.ctrl_bound[2]
        for pj in (length(ini_particle)+1):particle_num
            particles[pj].control_coefficients = [[(b-a)*rand()+a for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
    end

    fit = 0.0
    if length(pso.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: Particle Swarm Optimization (PSO)")
        println("non-controlled $str2 is $(1.0/f_noctrl)")
        println("initial $str2 is $(1.0/f_ini)")
        f_list = [1.0/f_ini]
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity = PSO_train_Copt(M, particles, p_fit, fit, max_episode, 
                            c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
                print("current $str2 is $(fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest, gbest, velocity = PSO_train_Copt(M, particles, p_fit, fit, max_episode, 
                        c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, gbest, velocity, sym)
            append!(f_list, fit)
            SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str2 is $(fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity = PSO_train_Copt(M, particles, p_fit, fit, max_episode, 
                            c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                print("current $str2 is $(fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest, gbest, velocity = PSO_train_Copt(M, particles, p_fit, fit, max_episode, 
                        c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, gbest, velocity, sym)
            append!(f_list, fit)
            SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str2 is $(fit)")
        end
        return fit, [gbest[k, :] for k in 1:ctrl_num]
    else
        println("multiparameter scenario")
        println("control algorithm: Particle Swarm Optimization (PSO)")
        println("non-controlled value of $str3 is $(f_noctrl)")
        println("initial value of $str3 is $(f_ini)")
        f_list = [f_ini]
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity = PSO_train_Copt(M, particles, p_fit, fit, max_episode, 
                            c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, 1.0/fit)
                SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
                print("current value of $str3 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest, gbest, velocity = PSO_train_Copt(M, particles, p_fit, fit, max_episode, 
                        c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, gbest, velocity, sym)
            append!(f_list, 1.0/fit)
            SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is $(1.0/fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity = PSO_train_Copt(M, particles, p_fit, fit, max_episode,
                                        c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, 1.0/fit)
                print("current value of $str3 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest, gbest, velocity = PSO_train_Copt(M, particles, p_fit, fit, max_episode, 
                                    c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, gbest, velocity, sym)
            append!(f_list, 1.0/fit)
            SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is $(1.0/fit)")
        end
        return 1/fit, [gbest[k, :] for k in 1:ctrl_num]
    end
   
end

function PSO_train_Copt(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, gbest, velocity, sym)
    for pj in 1:particle_num
        f_tp = obj_func(Val{sym}(), particles[pj], M)
        f_now = 1.0/f_tp
        if f_now > p_fit[pj]
            p_fit[pj] = f_now
            for di in 1:ctrl_num
                for ni in 1:ctrl_length
                    pbest[di,ni,pj] = particles[pj].control_coefficients[di][ni]
                end
            end
        end
    end

    for pj in 1:particle_num
        if p_fit[pj] > fit
            fit = p_fit[pj]
            for dj in 1:ctrl_num
                for nj in 1:ctrl_length
                    gbest[dj, nj] = pbest[dj,nj,pj]
                end
            end
        end 
    end  

    for pk in 1:particle_num
        control_coeff_pre = [zeros(ctrl_length) for i in 1:ctrl_num]
        for dk in 1:ctrl_num
            for ck in 1:ctrl_length
                control_coeff_pre[dk][ck] = particles[pk].control_coefficients[dk][ck]
                velocity[dk, ck, pk] = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                     + c2*rand()*(gbest[dk, ck] - particles[pk].control_coefficients[dk][ck])
                particles[pk].control_coefficients[dk][ck] += velocity[dk, ck, pk]
            end
        end

        for dm in 1:ctrl_num
            for cm in 1:ctrl_length
                particles[pk].control_coefficients[dm][cm] = (x-> x < particles[pk].ctrl_bound[1] ? particles[pk].ctrl_bound[1] : x > particles[pk].ctrl_bound[2] ? particles[pk].ctrl_bound[2] : x)(particles[pk].control_coefficients[dm][cm])
                velocity[dm, cm, pk] = particles[pk].control_coefficients[dm][cm] - control_coeff_pre[dm][cm]
            end
        end
    end
    return p_fit, fit, pbest, gbest, velocity
end
