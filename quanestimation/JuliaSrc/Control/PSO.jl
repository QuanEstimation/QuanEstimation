mutable struct PSO{T <: Complex,M <: Real} <: ControlSystem
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
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    PSO(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, ρ_initial::Matrix{T},
             times::Vector{M}, Liouville_operator::Vector{Matrix{T}},γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}},
             control_coefficients::Vector{Vector{M}}, ctrl_bound::Vector{M}, W::Matrix{M}, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = new{T,M}(freeHamiltonian, 
                Hamiltonian_derivative, ρ_initial, times, Liouville_operator, γ, control_Hamiltonian, control_coefficients, ctrl_bound, W, ρ, ∂ρ_∂x) 
end

function PSO_QFI(pso::PSO{T}, max_episodes, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("quantum parameter estimation")
    println("single parameter scenario")
    println("control algorithm: Particle Swarm Optimization (PSO)")
    Random.seed!(sd)
    ctrl_length = length(pso.control_coefficients[1])
    ctrl_num = length(pso.control_Hamiltonian)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ctrl_num, ctrl_length, particle_num)
    pbest = zeros(ctrl_num, ctrl_length, particle_num)
    gbest = zeros(ctrl_num, ctrl_length)
    velocity_best = zeros(ctrl_num,ctrl_length)
    p_fit = zeros(particle_num)
    qfi_noctrl = QFI(pso.freeHamiltonian, pso.Hamiltonian_derivative[1], pso.ρ_initial, pso.Liouville_operator, pso.γ, 
                    pso.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], pso.times)
    qfi_ini = QFI(pso.freeHamiltonian, pso.Hamiltonian_derivative[1], pso.ρ_initial, pso.Liouville_operator, pso.γ, 
                    pso.control_Hamiltonian, pso.control_coefficients, pso.times)
    f_list = [qfi_ini]
    println("non-controlled QFI is $(qfi_noctrl)")
    println("initial QFI is $(qfi_ini)")

    if typeof(max_episodes) == Int
        max_episodes = [max_episodes, max_episodes]
    end

    # initialize
    for pj in 1:length(ini_particle)
        particles[pj].control_coefficients = [[ini_particle[pj][i,j] for j in 1:ctrl_length] for i in 1:ctrl_num]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        particles[pj].control_coefficients = [[rand() for j in 1:ctrl_length] for i in 1:ctrl_num]
    end

    fit = 0.0
    if save_file==true
        for ei in 1:(max_episodes[1]-1)
            p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_QFIM(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, 
                                                                         ctrl_length, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            SaveFile(f_list, [gbest[k, :] for k in 1:ctrl_num])
            print("current QFI is $fit ($ei episodes) \r")
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_QFIM(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, 
                                                                         ctrl_length, pbest, gbest, velocity_best, velocity)
        append!(f_list, fit)
        SaveFile(f_list, [gbest[k, :] for k in 1:ctrl_num])
        print("\e[2K")
        println("Iteration over, data saved.")    
        println("Final QFI is $fit")
    else
        for ei in 1:(max_episodes[1]-1)
            p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_QFIM(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, 
                                                                              ctrl_length, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            print("current QFI is $fit ($ei episodes) \r")
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_QFIM(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, 
                                                                              ctrl_length, pbest, gbest, velocity_best, velocity)
        append!(f_list, fit)
        SaveFile(f_list, [gbest[k, :] for k in 1:ctrl_num])
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final QFI is $fit")
    end
    return nothing
end

function PSO_CFI(M, pso::PSO{T}, max_episodes, particle_num, ini_particle, c0, c1, c2,v0, sd, save_file) where {T<: Complex}
    println("quantum parameter estimation")
    println("single parameter scenario")
    println("control algorithm: Particle Swarm Optimization (PSO)")
    Random.seed!(sd)
    ctrl_length = length(pso.control_coefficients[1])
    ctrl_num = length(pso.control_Hamiltonian)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ctrl_num, ctrl_length, particle_num)
    pbest = zeros(ctrl_num, ctrl_length, particle_num)
    gbest = zeros(ctrl_num, ctrl_length)
    velocity_best = zeros(ctrl_num,ctrl_length)
    p_fit = zeros(particle_num)
    cfi_noctrl = CFI(M, pso.freeHamiltonian, pso.Hamiltonian_derivative[1], pso.ρ_initial, pso.Liouville_operator, pso.γ, 
                  pso.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], pso.times)
    cfi_ini = CFI(M, pso.freeHamiltonian, pso.Hamiltonian_derivative[1], pso.ρ_initial, pso.Liouville_operator, pso.γ, 
                  pso.control_Hamiltonian, pso.control_coefficients, pso.times)
    f_list = [cfi_ini]
    println("non-controlled CFI is $(cfi_noctrl)") 
    println("initial CFI is $(cfi_ini)")    
    
    if typeof(max_episodes) == Int
        max_episodes = [max_episodes, max_episodes]
    end

    # initialize
    for pj in 1:length(ini_particle)
        particles[pj].control_coefficients = [[ini_particle[pj][i,j] for j in 1:ctrl_length] for i in 1:ctrl_num]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        particles[pj].control_coefficients = [[rand() for j in 1:ctrl_length] for i in 1:ctrl_num]
    end

    fit = 0.0
    if save_file==true
        for ei in 1:(max_episodes[1]-1)
            p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_CFIM(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, 
                                                                              ctrl_length, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            SaveFile(f_list, [gbest[k, :] for k in 1:ctrl_num])
            print("current CFI is $fit ($ei episodes) \r")
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_CFIM(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, 
                                                                              ctrl_length, pbest, gbest, velocity_best, velocity)
        append!(f_list, fit)
        SaveFile(f_list, [gbest[k, :] for k in 1:ctrl_num])
        print("\e[2K")
        println("Iteration over, data saved.")    
        println("Final CFI is $fit")
    else
        for ei in 1:(max_episodes[1]-1)
            p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_CFIM(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, 
                                                                              ctrl_length, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            print("current CFI is $fit ($ei episodes) \r")
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_CFIM(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, 
                                                                              ctrl_length, pbest, gbest, velocity_best, velocity)
        append!(f_list, fit)
        SaveFile(f_list, [gbest[k, :] for k in 1:ctrl_num])
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final CFI is $fit")
    end
    return nothing
end

function PSO_QFIM(pso::PSO{T}, max_episodes, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("quantum parameter estimation")
    println("multiparameter scenario")
    println("control algorithm: Particle Swarm Optimization (PSO)")
    Random.seed!(sd)
    ctrl_length = length(pso.control_coefficients[1])
    ctrl_num = length(pso.control_Hamiltonian)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ctrl_num, ctrl_length, particle_num)
    pbest = zeros(ctrl_num, ctrl_length, particle_num)
    gbest = zeros(ctrl_num, ctrl_length)
    velocity_best = zeros(ctrl_num,ctrl_length)
    p_fit = zeros(particle_num)
    F_noctrl = QFIM(pso.freeHamiltonian, pso.Hamiltonian_derivative, pso.ρ_initial, pso.Liouville_operator, pso.γ, 
                    pso.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], pso.times)
    qfi_noctrl = real(tr(pso.W*pinv(F_noctrl)))
    F_ini = QFIM(pso.freeHamiltonian, pso.Hamiltonian_derivative, pso.ρ_initial, pso.Liouville_operator, pso.γ, 
                    pso.control_Hamiltonian, pso.control_coefficients, pso.times)
    qfi_ini = real(tr(pso.W*pinv(F_ini)))
    f_list = [qfi_ini]
    println("non-controlled value of Tr(WF^{-1}) is $(qfi_noctrl)")
    println("initial value of Tr(WF^{-1}) is $(qfi_ini)")
    
    if typeof(max_episodes) == Int
        max_episodes = [max_episodes, max_episodes]
    end

    # initialize
    for pj in 1:length(ini_particle)
        particles[pj].control_coefficients = [[ini_particle[pj][i,j] for j in 1:ctrl_length] for i in 1:ctrl_num]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        particles[pj].control_coefficients = [[rand() for j in 1:ctrl_length] for i in 1:ctrl_num]
    end

    fit = 0.0
    if save_file==true
        for ei in 1:(max_episodes[1]-1)
            p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_QFIM(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, 
                                                                              ctrl_length, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, 1.0/fit)
            SaveFile(f_list, [gbest[k, :] for k in 1:ctrl_num])
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_QFIM(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, 
                                                                              ctrl_length, pbest, gbest, velocity_best, velocity)
        append!(f_list, 1.0/fit)
        SaveFile(f_list, [gbest[k, :] for k in 1:ctrl_num])
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    else
        for ei in 1:(max_episodes[1]-1)
            p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_QFIM(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, 
                                                                              ctrl_length, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, 1.0/fit)
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_QFIM(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, 
                                                                              ctrl_length, pbest, gbest, velocity_best, velocity)
        append!(f_list, 1.0/fit)
        SaveFile(f_list, [gbest[k, :] for k in 1:ctrl_num])
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    end
    return nothing
end

function PSO_CFIM(M, pso::PSO{T}, max_episodes, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("quantum parameter estimation")
    println("multiparameter scenario")
    println("control algorithm: Particle Swarm Optimization (PSO)")
    Random.seed!(sd)
    ctrl_length = length(pso.control_coefficients[1])
    ctrl_num = length(pso.control_Hamiltonian)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ctrl_num, ctrl_length, particle_num)
    pbest = zeros(ctrl_num, ctrl_length, particle_num)
    gbest = zeros(ctrl_num, ctrl_length)
    velocity_best = zeros(ctrl_num,ctrl_length)
    p_fit = zeros(particle_num)
    F_noctrl = CFIM(M, pso.freeHamiltonian, pso.Hamiltonian_derivative, pso.ρ_initial, pso.Liouville_operator, pso.γ, 
                    pso.control_Hamiltonian, [zeros(ctrl_length) for i in 1:ctrl_num], pso.times)
    cfi_noctrl = real(tr(pso.W*pinv(F_noctrl)))
    F_ini = CFIM(M, pso.freeHamiltonian, pso.Hamiltonian_derivative, pso.ρ_initial, pso.Liouville_operator, pso.γ, 
                    pso.control_Hamiltonian, pso.control_coefficients, pso.times)
    cfi_ini = real(tr(pso.W*pinv(F_ini)))
    f_list = [cfi_ini]
    println("non-controlled value of Tr(WF^{-1}) is $(cfi_noctrl)")
    println("initial value of Tr(WF^{-1}) is $(cfi_ini)")

    if typeof(max_episodes) == Int
        max_episodes = [max_episodes, max_episodes]
    end
    
    # initialize
    for pj in 1:length(ini_particle)
        particles[pj].control_coefficients = [[ini_particle[pj][i,j] for j in 1:ctrl_length] for i in 1:ctrl_num]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        particles[pj].control_coefficients = [[rand() for j in 1:ctrl_length] for i in 1:ctrl_num]
    end

    fit = 0.0
    if save_file==true
        for ei in 1:(max_episodes[1]-1)
            p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_CFIM(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, 
                                                                              ctrl_length, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, 1.0/fit)
            SaveFile(f_list, [gbest[k, :] for k in 1:ctrl_num])
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_CFIM(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, 
                                                                              ctrl_length, pbest, gbest, velocity_best, velocity)
        append!(f_list, 1.0/fit)
        SaveFile(f_list, [gbest[k, :] for k in 1:ctrl_num])
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    else
        for ei in 1:(max_episodes[1]-1)
            p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_CFIM(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, 
                                                                              ctrl_length, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, 1.0/fit)
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = PSO_train_CFIM(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, 
                                                                              ctrl_length, pbest, gbest, velocity_best, velocity)
        append!(f_list, 1.0/fit)
        SaveFile(f_list, [gbest[k, :] for k in 1:ctrl_num])
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    end
    return nothing
end

function PSO_train_QFIM(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, gbest, velocity_best, velocity)
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
                    gbest[dj, nj] =  particles[pj].control_coefficients[dj][nj]
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

                velocity[dk, ck, pk]  = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
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

function PSO_train_CFIM(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, ctrl_num, ctrl_length, pbest, gbest, velocity_best, velocity)
    @inbounds for pj in 1:particle_num
        f_now = 1.0/real(tr(pinv(particles[pj].W*CFIM(M, particles[pj]))))
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
                    gbest[dj, nj] =  particles[pj].control_coefficients[dj][nj]
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

                velocity[dk, ck, pk]  = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
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
