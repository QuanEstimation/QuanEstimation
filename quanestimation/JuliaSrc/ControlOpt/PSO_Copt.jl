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
    accuracy::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    PSO_Copt(freeHamiltonian, Hamiltonian_derivative::Vector{Matrix{T}}, ρ0::Matrix{T},tspan::Vector{M}, decay_opt::Vector{Matrix{T}},
        γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{M}}, ctrl_bound::Vector{M},
        W::Matrix{M}, accuracy::M, ρ=Vector{Matrix{T}}(undef, 1), ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef,1)) where {T<:Complex,M<:Real}= 
        new{T,M}(freeHamiltonian, Hamiltonian_derivative, ρ0, tspan, decay_opt, γ, control_Hamiltonian, control_coefficients,
                 ctrl_bound, W, accuracy, ρ, ∂ρ_∂x) 
end

function QFIM_PSO_Copt(pso::PSO_Copt{T}, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file) where {T<:Complex}
    println("quantum parameter estimation")
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
    velocity_best = zeros(ctrl_num,ctrl_length)
    p_fit = zeros(particle_num)
    F_noctrl = QFIM(pso.freeHamiltonian, pso.Hamiltonian_derivative, pso.ρ0, pso.decay_opt, pso.γ, 
                    pso.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], pso.tspan, pso.accuracy)
    f_noctrl = real(tr(pso.W*pinv(F_noctrl)))
    F_ini = QFIM(pso)
    f_ini = real(tr(pso.W*pinv(F_ini)))

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
        println("non-controlled QFI is $(1.0/f_noctrl)")
        println("initial QFI is $(1.0/f_ini)")
        f_list = [1.0/f_ini]
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_QFIM(particles, p_fit, fit, max_episode, 
                                                                    c0, c1, c2, particle_num, ctrl_num, ctrl_length, 
                                                                    pbest, gbest, velocity_best, velocity)
                if ei%max_episode[2] == 0
                    pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
                print("current QFI is $(fit) ($ei episodes) \r")
            end
            p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_QFIM(particles, p_fit, fit, max_episode, 
                                                                c0, c1, c2, particle_num, ctrl_num, ctrl_length, 
                                                                pbest, gbest, velocity_best, velocity)
            append!(f_list, fit)
            SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final QFI is $(fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_QFIM(particles, p_fit, fit, max_episode, 
                                                                    c0, c1, c2, particle_num, ctrl_num, ctrl_length,
                                                                    pbest, gbest, velocity_best, velocity)
                if ei%max_episode[2] == 0
                    pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                print("current QFI is $(fit) ($ei episodes) \r")
            end
            p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_QFIM(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, ctrl_num, 
                                                                                  ctrl_length, pbest, gbest, velocity_best, velocity)
            append!(f_list, fit)
            SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final QFI is $(fit)")
        end
    else
        println("multiparameter scenario")
        println("control algorithm: Particle Swarm Optimization (PSO)")
        println("non-controlled value of Tr(WF^{-1}) is $(f_noctrl)")
        println("initial value of Tr(WF^{-1}) is $(f_ini)")
        f_list = [f_ini]
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_QFIM(particles, p_fit, fit, max_episode, 
                                                                    c0, c1, c2, particle_num, ctrl_num, ctrl_length, 
                                                                    pbest, gbest, velocity_best, velocity)
                if ei%max_episode[2] == 0
                    pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, 1.0/fit)
                SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
                print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
            end
            p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_QFIM(particles, p_fit, fit, max_episode, 
                                                                c0, c1, c2, particle_num, ctrl_num, ctrl_length,
                                                                pbest, gbest, velocity_best, velocity)
            append!(f_list, 1.0/fit)
            SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_QFIM(particles, p_fit, fit, max_episode,
                                                                    c0, c1, c2, particle_num, ctrl_num, ctrl_length,
                                                                    pbest, gbest, velocity_best, velocity)
                if ei%max_episode[2] == 0
                    pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, 1.0/fit)
                print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
            end
            p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_QFIM(particles, p_fit, fit, max_episode, 
                                                                c0, c1, c2, particle_num, ctrl_num, ctrl_length,
                                                                pbest, gbest, velocity_best, velocity)
            append!(f_list, 1.0/fit)
            SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
        end
    end
end

function CFIM_PSO_Copt(Measurement, pso::PSO_Copt{T}, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file) where {T<:Complex}
    println("classical parameter estimation")
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
    velocity_best = zeros(ctrl_num,ctrl_length)
    p_fit = zeros(particle_num)
    F_noctrl = CFIM(Measurement, pso.freeHamiltonian, pso.Hamiltonian_derivative, pso.ρ0, pso.decay_opt, pso.γ, 
                    pso.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], pso.tspan, pso.accuracy)
    f_noctrl = real(tr(pso.W*pinv(F_noctrl)))
    F_ini = CFIM(Measurement, pso)
    f_ini = real(tr(pso.W*pinv(F_ini)))

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
        println("non-controlled CFI is $(1.0/f_noctrl)") 
        println("initial CFI is $(1.0/f_ini)")
        
        f_list = [1.0/f_ini]
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_CFIM(Measurement, particles, p_fit, fit, max_episode, 
                                                                    c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, 
                                                                    gbest, velocity_best, velocity)
                if ei%max_episode[2] == 0
                    pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
                print("current CFI is $(fit) ($ei episodes) \r")
            end
            p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_CFIM(Measurement, particles, p_fit, fit, max_episode, 
                                                                c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest,
                                                                gbest, velocity_best, velocity)
            append!(f_list, fit)
            SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final CFI is $(fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_CFIM(Measurement, particles, p_fit, fit, max_episode, 
                                                                    c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, 
                                                                    gbest, velocity_best, velocity)
                if ei%max_episode[2] == 0
                    pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                print("current CFI is $(fit) ($ei episodes) \r")
            end
            p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_CFIM(Measurement, particles, p_fit, fit, max_episode, 
                                                                c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, 
                                                                gbest, velocity_best, velocity)
            append!(f_list, fit)
            SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final CFI is $(fit)")
        end
    else
        println("multiparameter scenario")
        println("control algorithm: Particle Swarm Optimization (PSO)")
        println("non-controlled value of Tr(WI^{-1}) is $(f_noctrl)")
        println("initial value of Tr(WI^{-1}) is $(f_ini)")
        
        f_list = [f_ini]
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_CFIM(Measurement, particles, p_fit, fit, max_episode,
                                                                    c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, 
                                                                    gbest, velocity_best, velocity)
                if ei%max_episode[2] == 0
                    pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, 1.0/fit)
                SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
                print("current value of Tr(WI^{-1}) is $(1.0/fit) ($ei episodes) \r")
            end
            p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_CFIM(Measurement, particles, p_fit, fit, max_episode, 
                                                                c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, 
                                                                gbest, velocity_best, velocity)
            append!(f_list, 1.0/fit)
            SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of Tr(WI^{-1}) is $(1.0/fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_CFIM(Measurement, particles, p_fit, fit, max_episode, 
                                                                    c0, c1, c2, particle_num, ctrl_num,  ctrl_length, pbest, 
                                                                    gbest, velocity_best, velocity)
                if ei%max_episode[2] == 0
                    pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, 1.0/fit)
                print("current value of Tr(WI^{-1}) is $(1.0/fit) ($ei episodes) \r")
            end
            p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_CFIM(Measurement, particles, p_fit, fit, max_episode, 
                                                                c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, 
                                                                gbest, velocity_best, velocity)
            append!(f_list, 1.0/fit)
            SaveFile_ctrl(f_list, [gbest[k, :] for k in 1:ctrl_num])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of Tr(WI^{-1}) is $(1.0/fit)")
        end
    end
end

function PSO_train_QFIM(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, gbest, velocity_best, velocity)
    @inbounds for pj in 1:particle_num
        f_now = 1.0/real(tr(particles[pj].W*pinv(QFIM(particles[pj]))))
        if f_now > p_fit[pj]
            p_fit[pj] = f_now
            for di in 1:ctrl_num
                for ni in 1:ctrl_length
                    pbest[di,ni,pj] = particles[pj].control_coefficients[di][ni]
                end
            end
        end
    end
    @inbounds for pj in 1:particle_num
        if p_fit[pj] > fit
            fit = p_fit[pj]
            for dj in 1:ctrl_num
                @inbounds for nj in 1:ctrl_length
                    gbest[dj, nj] = particles[pj].control_coefficients[dj][nj]
                    velocity_best[dj, nj] = velocity[dj, nj, pj]
                end
            end
        end
    end  
    @inbounds for pk in 1:particle_num
        control_coeff_pre = [zeros(ctrl_length) for i in 1:ctrl_num]
        for dk in 1:ctrl_num
            @inbounds for ck in 1:ctrl_length
                control_coeff_pre[dk][ck] = particles[pk].control_coefficients[dk][ck]
                velocity[dk, ck, pk] = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                     + c2*rand()*(gbest[dk, ck] - particles[pk].control_coefficients[dk][ck])
                particles[pk].control_coefficients[dk][ck] += velocity[dk, ck, pk]
            end
        end

        for dm in 1:ctrl_num
            @inbounds for cm in 1:ctrl_length
                particles[pk].control_coefficients[dm][cm] = (x-> x < particles[pk].ctrl_bound[1] ? particles[pk].ctrl_bound[1] : x > particles[pk].ctrl_bound[2] ? particles[pk].ctrl_bound[2] : x)(particles[pk].control_coefficients[dm][cm])
                velocity[dm, cm, pk] = particles[pk].control_coefficients[dm][cm] - control_coeff_pre[dm][cm]
            end
        end
    end
    return p_fit, fit, pbest, gbest, velocity_best, velocity
end

function PSO_train_CFIM(Measurement, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, gbest, velocity_best, velocity)
    @inbounds for pj in 1:particle_num
        f_now = 1.0/real(tr(particles[pj].W*pinv(CFIM(Measurement, particles[pj]))))
        if f_now > p_fit[pj]
            p_fit[pj] = f_now
            for di in 1:ctrl_num
                for ni in 1:ctrl_length
                    pbest[di,ni,pj] = particles[pj].control_coefficients[di][ni]
                end
            end
        end
    end
    @inbounds for pj in 1:particle_num
        if p_fit[pj] > fit
            fit = p_fit[pj]
            for dj in 1:ctrl_num
                @inbounds for nj in 1:ctrl_length
                    gbest[dj, nj] = particles[pj].control_coefficients[dj][nj]
                    velocity_best[dj, nj] = velocity[dj, nj, pj]
                end
            end
        end
    end  
    @inbounds for pk in 1:particle_num
        control_coeff_pre = [zeros(ctrl_length) for i in 1:ctrl_num]
        for dk in 1:ctrl_num
            @inbounds for ck in 1:ctrl_length
                control_coeff_pre[dk][ck] = particles[pk].control_coefficients[dk][ck]

                velocity[dk, ck, pk] = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                     + c2*rand()*(gbest[dk, ck] - particles[pk].control_coefficients[dk][ck])
                particles[pk].control_coefficients[dk][ck] += velocity[dk, ck, pk]
            end
        end

        for dm in 1:ctrl_num
            @inbounds for cm in 1:ctrl_length
                particles[pk].control_coefficients[dm][cm] = (x-> x < particles[pk].ctrl_bound[1] ? particles[pk].ctrl_bound[1] : x > particles[pk].ctrl_bound[2] ? particles[pk].ctrl_bound[2] : x)(particles[pk].control_coefficients[dm][cm])
                velocity[dm, cm, pk] = particles[pk].control_coefficients[dm][cm] - control_coeff_pre[dm][cm]
            end
        end
    end
    return p_fit, fit, pbest, gbest, velocity_best, velocity
end
