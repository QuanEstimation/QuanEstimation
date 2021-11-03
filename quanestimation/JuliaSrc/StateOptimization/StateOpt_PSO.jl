############# time-independent Hamiltonian (noiseless) ################
mutable struct TimeIndepend_noiseless{T <: Complex,M <: Real}
    freeHamiltonian::Matrix{T}
    Hamiltonian_derivative::Vector{Matrix{T}}
    psi::Vector{T}
    times::Vector{M}
    W::Matrix{M}
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    TimeIndepend_noiseless(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, psi::Vector{T},
             times::Vector{M}, W::Matrix{M}, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = new{T,M}(freeHamiltonian, 
                Hamiltonian_derivative, psi, times, W, ρ, ∂ρ_∂x) 
end

function PSO_QFI(pso::TimeIndepend_noiseless{T}, max_episodes, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("state optimization")
    println("single parameter scenario")
    println("search algorithm: Particle Swarm Optimization (PSO)")
    Random.seed!(sd)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
    velocity_best = zeros(ComplexF64, dim)
    p_fit = zeros(particle_num)

    if typeof(max_episodes) == Int
        max_episodes = [max_episodes, max_episodes]
    end
    # initialize
    for pj in 1:length(ini_particle)
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    qfi_ini = QFIM_TimeIndepend(pso.freeHamiltonian, pso.Hamiltonian_derivative[1], pso.psi, pso.times)
    f_list = [qfi_ini]
    println("initial QFI is $(qfi_ini)")      
    fit = 0.0
    if save_file==true
        for ei in 1:(max_episodes[1]-1)
            #### train ####
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_QFIM_noiseless(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            SaveFile_pso(dim, f_list, gbest)
            print("current QFI is $fit ($ei episodes) \r")
            
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = train_QFIM_noiseless(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
        append!(f_list, fit)
        SaveFile_pso(dim, f_list, gbest)
        print("\e[2K")    
        println("Iteration over, data saved.")
        println("Final QFI is $fit")
    else
        for ei in 1:(max_episodes[1]-1)
            #### train ####
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_QFIM_noiseless(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            print("current QFI is $fit ($ei episodes) \r")
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = train_QFIM_noiseless(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
        append!(f_list, fit)
        SaveFile_pso(dim, f_list, gbest)
        print("\e[2K") 
        println("Iteration over, data saved.") 
        println("Final QFI is $fit")
    end
    return nothing
end

function PSO_QFIM(pso::TimeIndepend_noiseless{T}, max_episodes, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("state optimization")
    println("multiparameter scenario")
    println("search algorithm: Particle Swarm Optimization (PSO)")
    Random.seed!(sd)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
    velocity_best = zeros(ComplexF64, dim)
    p_fit = zeros(particle_num)

    if typeof(max_episodes) == Int
        max_episodes = [max_episodes, max_episodes]
    end

    # initialize
    for pj in 1:length(ini_particle)
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    F = QFIM_TimeIndepend(pso.freeHamiltonian, pso.Hamiltonian_derivative, pso.psi, pso.times)
    qfi_ini = real(tr(pso.W*pinv(F)))
    f_list = [qfi_ini]
    println("initial value of Tr(WF^{-1}) is $(qfi_ini)")       
    fit = 0.0
    if save_file==true
        for ei in 1:(max_episodes[1]-1)
            #### train ####
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_QFIM_noiseless(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, (1.0/fit))
            SaveFile_pso(dim, f_list, gbest)
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
            
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = train_QFIM_noiseless(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
        append!(f_list, (1.0/fit))
        SaveFile_pso(dim, f_list, gbest)
        print("\e[2K")    
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    else
        for ei in 1:(max_episodes[1]-1)
            #### train ####
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_QFIM_noiseless(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, (1.0/fit))
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = train_QFIM_noiseless(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
        append!(f_list, (1.0/fit))
        SaveFile_pso(dim, f_list, gbest)
        print("\e[2K") 
        println("Iteration over, data saved.") 
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    end
    return nothing
end

function PSO_CFI(M, pso::TimeIndepend_noiseless{T}, max_episodes, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("state optimization")
    println("single parameter scenario")
    println("search algorithm: Particle Swarm Optimization (PSO)")
    Random.seed!(sd)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
    velocity_best = zeros(ComplexF64, dim)
    p_fit = zeros(particle_num)

    if typeof(max_episodes) == Int
        max_episodes = [max_episodes, max_episodes]
    end

    # initialize
    for pj in 1:length(ini_particle)
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    f_ini = CFIM_TimeIndepend(M, pso.freeHamiltonian, pso.Hamiltonian_derivative[1], pso.psi, pso.times)
    f_list = [f_ini]
    println("initial CFI is $(f_ini)")       
    fit = 0.0
    if save_file==true
        for ei in 1:(max_episodes[1]-1)
            #### train ####
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM_noiseless(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            SaveFile_pso(dim, f_list, gbest)
            print("current CFI is $fit ($ei episodes) \r")
            
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM_noiseless(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
        append!(f_list, fit)
        SaveFile_pso(dim, f_list, gbest)
        print("\e[2K")    
        println("Iteration over, data saved.")
        println("Final CFI is $fit")
    else
        for ei in 1:(max_episodes[1]-1)
            #### train ####
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM_noiseless(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            print("current CFI is $fit ($ei episodes) \r")
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM_noiseless(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
        append!(f_list, fit) 
        SaveFile_pso(dim, f_list, gbest)
        print("\e[2K") 
        println("Iteration over, data saved.") 
        println("Final CFI is $fit")
    end
    return nothing
end

function PSO_CFIM(M, pso::TimeIndepend_noiseless{T}, max_episodes, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("state optimization")
    println("multiparameter scenario")
    println("search algorithm: Particle Swarm Optimization (PSO)")
    Random.seed!(sd)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
    velocity_best = zeros(ComplexF64, dim)
    p_fit = zeros(particle_num)

    if typeof(max_episodes) == Int
        max_episodes = [max_episodes, max_episodes]
    end

    # initialize
    for pj in 1:length(ini_particle)
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    F = CFIM_TimeIndepend(M, pso.freeHamiltonian, pso.Hamiltonian_derivative, pso.psi, pso.times)
    f_ini = real(tr(pso.W*pinv(F)))
    f_list = [f_ini]
    println("initial value of Tr(WF^{-1}) is $(f_ini)")       
    fit = 0.0
    if save_file==true
        for ei in 1:(max_episodes[1]-1)
            #### train ####
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM_noiseless(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, (1.0/fit))
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
            SaveFile_pso(dim, f_list, gbest)
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM_noiseless(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
        append!(f_list, (1.0/fit))
        print("\e[2K")    
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    else
        for ei in 1:(max_episodes[1]-1)
            #### train ####
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM_noiseless(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, (1.0/fit))
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM_noiseless(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
        append!(f_list, (1.0/fit))
        SaveFile_pso(dim, f_list, gbest)
        print("\e[2K") 
        println("Iteration over, data saved.") 
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    end
    return nothing
end

function train_QFIM_noiseless(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, dim, pbest, gbest, velocity_best, velocity)
    for pj in 1:particle_num
        F_tp = QFIM_TimeIndepend(particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative, particles[pj].psi, particles[pj].times)
        f_now = 1.0/real(tr(particles[pj].W*pinv(F_tp)))
        if f_now > p_fit[pj]
            p_fit[pj] = f_now
            for di in 1:dim
                pbest[di,pj] = particles[pj].psi[di]
            end
        end
    end

    for pj in 1:particle_num
        if p_fit[pj] > fit
            fit = p_fit[pj]
            for dj in 1:dim
                gbest[dj] = particles[pj].psi[dj]
                velocity_best[dj] = velocity[dj, pj]
            end
        end
    end  

    for pk in 1:particle_num
        psi_pre = zeros(ComplexF64, dim)
        for dk in 1:dim
            psi_pre[dk] = particles[pk].psi[dk]
            velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
            particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
        end

        particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

        for dm in 1:dim
            velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
        end
    end
    return p_fit, fit, pbest, gbest, velocity_best, velocity
end

function train_CFIM_noiseless(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, dim, pbest, gbest, velocity_best, velocity)
    for pj in 1:particle_num
        F_tp = CFIM_TimeIndepend(M, particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative, particles[pj].psi, particles[pj].times)
        f_now = 1.0/real(tr(particles[pj].W*pinv(F_tp)))
        if f_now > p_fit[pj]
            p_fit[pj] = f_now
            for di in 1:dim
                pbest[di,pj] = particles[pj].psi[di]
            end
        end
    end

    for pj in 1:particle_num
        if p_fit[pj] > fit
            fit = p_fit[pj]
            for dj in 1:dim
                gbest[dj] = particles[pj].psi[dj]
                velocity_best[dj] = velocity[dj, pj]
            end
        end
    end  

    for pk in 1:particle_num
        psi_pre = zeros(ComplexF64, dim)
        for dk in 1:dim
            psi_pre[dk] = particles[pk].psi[dk]
            velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
            particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
        end

        particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

        for dm in 1:dim
            velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
        end
    end
    return p_fit, fit, pbest, gbest, velocity_best, velocity
end

############# time-independent Hamiltonian (noise) ################
mutable struct TimeIndepend_noise{T <: Complex,M <: Real}
    freeHamiltonian::Matrix{T}
    Hamiltonian_derivative::Vector{Matrix{T}}
    psi::Vector{T}
    times::Vector{M}
    Liouville_operator::Vector{Matrix{T}}
    γ::Vector{M}
    W::Matrix{M}
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    TimeIndepend_noise(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, psi::Vector{T},
             times::Vector{M}, Liouville_operator::Vector{Matrix{T}}, γ::Vector{M}, W::Matrix{M}, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = new{T,M}(freeHamiltonian, 
                Hamiltonian_derivative, psi, times, Liouville_operator, γ, W, ρ, ∂ρ_∂x) 
end

function PSO_QFI(pso::TimeIndepend_noise{T}, max_episodes, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("state optimization")
    println("single parameter scenario")
    println("search algorithm: Particle Swarm Optimization (PSO)")
    Random.seed!(sd)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
    velocity_best = zeros(ComplexF64, dim)
    p_fit = zeros(particle_num)

    if typeof(max_episodes) == Int
        max_episodes = [max_episodes, max_episodes]
    end
    # initialize
    for pj in 1:length(ini_particle)
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    qfi_ini = QFIM_TimeIndepend(pso.freeHamiltonian, pso.Hamiltonian_derivative[1], pso.psi*(pso.psi)', pso.Liouville_operator, pso.γ, pso.times)
    f_list = [qfi_ini]
    println("initial QFI is $(qfi_ini)")      
    fit = 0.0
    if save_file==true
        for ei in 1:(max_episodes[1]-1)
            #### train ####
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_QFIM_noise(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            SaveFile_pso(dim, f_list, gbest)
            print("current QFI is $fit ($ei episodes) \r")
            
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = train_QFIM_noise(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
        append!(f_list, fit)
        SaveFile_pso(dim, f_list, gbest)
        print("\e[2K")    
        println("Iteration over, data saved.")
        println("Final QFI is $fit")
    else
        for ei in 1:(max_episodes[1]-1)
            #### train ####
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_QFIM_noise(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            print("current QFI is $fit ($ei episodes) \r")
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = train_QFIM_noise(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
        append!(f_list, fit)
        SaveFile_pso(dim, f_list, gbest)
        print("\e[2K") 
        println("Iteration over, data saved.") 
        println("Final QFI is $fit")
    end
    return nothing
end

function PSO_QFIM(pso::TimeIndepend_noise{T}, max_episodes, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("state optimization")
    println("multiparameter scenario")
    println("search algorithm: Particle Swarm Optimization (PSO)")
    Random.seed!(sd)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
    velocity_best = zeros(ComplexF64, dim)
    p_fit = zeros(particle_num)

    if typeof(max_episodes) == Int
        max_episodes = [max_episodes, max_episodes]
    end

    # initialize
    for pj in 1:length(ini_particle)
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    F = QFIM_TimeIndepend(pso.freeHamiltonian, pso.Hamiltonian_derivative, pso.psi*(pso.psi)', pso.Liouville_operator, pso.γ, pso.times)
    qfi_ini = real(tr(pso.W*pinv(F)))
    f_list = [qfi_ini]
    println("initial value of Tr(WF^{-1}) is $(qfi_ini)")       
    fit = 0.0
    if save_file==true
        for ei in 1:(max_episodes[1]-1)
            #### train ####
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_QFIM_noise(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, (1.0/fit))
            SaveFile_pso(dim, f_list, gbest)
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
            
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = train_QFIM_noise(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
        append!(f_list, (1.0/fit))
        SaveFile_pso(dim, f_list, gbest)
        print("\e[2K")    
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    else
        for ei in 1:(max_episodes[1]-1)
            #### train ####
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_QFIM_noise(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, (1.0/fit))
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = train_QFIM_noise(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
        append!(f_list, (1.0/fit))
        SaveFile_pso(dim, f_list, gbest)
        print("\e[2K") 
        println("Iteration over, data saved.") 
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    end
    return nothing
end

function PSO_CFI(M, pso::TimeIndepend_noise{T}, max_episodes, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("state optimization")
    println("single parameter scenario")
    println("search algorithm: Particle Swarm Optimization (PSO)")
    Random.seed!(sd)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
    velocity_best = zeros(ComplexF64, dim)
    p_fit = zeros(particle_num)

    if typeof(max_episodes) == Int
        max_episodes = [max_episodes, max_episodes]
    end

    # initialize
    for pj in 1:length(ini_particle)
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    f_ini = CFIM_TimeIndepend(M, pso.freeHamiltonian, pso.Hamiltonian_derivative[1], pso.psi*(pso.psi)', pso.Liouville_operator, pso.γ, pso.times)
    f_list = [f_ini]
    println("initial CFI is $(f_ini)")       
    fit = 0.0
    if save_file==true
        for ei in 1:(max_episodes[1]-1)
            #### train ####
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM_noise(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            SaveFile_pso(dim, f_list, gbest)
            print("current CFI is $fit ($ei episodes) \r")
            
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM_noise(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
        append!(f_list, fit)
        SaveFile_pso(dim, f_list, gbest)
        print("\e[2K")    
        println("Iteration over, data saved.")
        println("Final CFI is $fit")
    else
        for ei in 1:(max_episodes[1]-1)
            #### train ####
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM_noise(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            print("current CFI is $fit ($ei episodes) \r")
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM_noise(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
        append!(f_list, fit)
        SaveFile_pso(dim, f_list, gbest)
        print("\e[2K") 
        println("Iteration over, data saved.") 
        println("Final CFI is $fit")
    end
    return nothing
end

function PSO_CFIM(M, pso::TimeIndepend_noise{T}, max_episodes, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("state optimization")
    println("multiparameter scenario")
    println("search algorithm: Particle Swarm Optimization (PSO)")
    Random.seed!(sd)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
    velocity_best = zeros(ComplexF64, dim)
    p_fit = zeros(particle_num)

    if typeof(max_episodes) == Int
        max_episodes = [max_episodes, max_episodes]
    end
    
    # initialize
    for pj in 1:length(ini_particle)
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    F = CFIM_TimeIndepend(M, pso.freeHamiltonian, pso.Hamiltonian_derivative, pso.psi*(pso.psi)', pso.Liouville_operator, pso.γ, pso.times)
    f_ini = real(tr(pso.W*pinv(F)))
    f_list = [f_ini]
    println("initial value of Tr(WF^{-1}) is $(f_ini)")       
    fit = 0.0
    if save_file==true
        for ei in 1:(max_episodes[1]-1)
            #### train ####
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM_noise(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, (1.0/fit))
            SaveFile_pso(dim, f_list, gbest)
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
            
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM_noise(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
        append!(f_list, (1.0/fit))
        SaveFile_pso(dim, f_list, gbest)
        print("\e[2K")    
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    else
        for ei in 1:(max_episodes[1]-1)
            #### train ####
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM_noise(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, (1.0/fit))
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
        end
        p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM_noise(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity_best, velocity)
        append!(f_list, (1.0/fit))
        SaveFile_pso(dim, f_list, gbest)
        print("\e[2K") 
        println("Iteration over, data saved.") 
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    end
    return nothing
end

function train_QFIM_noise(particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, dim, pbest, gbest, velocity_best, velocity)
    for pj in 1:particle_num
        rho = particles[pj].psi*(particles[pj].psi)'
        F_tp = QFIM_TimeIndepend(particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative, rho, particles[pj].Liouville_operator, particles[pj].γ, particles[pj].times)
        f_now = 1.0/real(tr(particles[pj].W*pinv(F_tp)))
        if f_now > p_fit[pj]
            p_fit[pj] = f_now
            for di in 1:dim
                pbest[di,pj] = particles[pj].psi[di]
            end
        end
    end

    for pj in 1:particle_num
        if p_fit[pj] > fit
            fit = p_fit[pj]
            for dj in 1:dim
                gbest[dj] = particles[pj].psi[dj]
                velocity_best[dj] = velocity[dj, pj]
            end
        end
    end  

    for pk in 1:particle_num
        psi_pre = zeros(ComplexF64, dim)
        for dk in 1:dim
            psi_pre[dk] = particles[pk].psi[dk]
            velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
            particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
        end

        particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

        for dm in 1:dim
            velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
        end
    end
    return p_fit, fit, pbest, gbest, velocity_best, velocity
end

function train_CFIM_noise(M, particles, p_fit, fit, max_episodes, c0, c1, c2, particle_num, dim, pbest, gbest, velocity_best, velocity)
    for pj in 1:particle_num
        rho = particles[pj].psi*(particles[pj].psi)'
        F_tp = CFIM_TimeIndepend(M, particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative, rho, particles[pj].Liouville_operator, particles[pj].γ, particles[pj].times)
        f_now = 1.0/real(tr(particles[pj].W*pinv(F_tp)))
        if f_now > p_fit[pj]
            p_fit[pj] = f_now
            for di in 1:dim
                pbest[di,pj] = particles[pj].psi[di]
            end
        end
    end

    for pj in 1:particle_num
        if p_fit[pj] > fit
            fit = p_fit[pj]
            for dj in 1:dim
                gbest[dj] = particles[pj].psi[dj]
                velocity_best[dj] = velocity[dj, pj]
            end
        end
    end  

    for pk in 1:particle_num
        psi_pre = zeros(ComplexF64, dim)
        for dk in 1:dim
            psi_pre[dk] = particles[pk].psi[dk]
            velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
            particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
        end

        particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

        for dm in 1:dim
            velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
        end
    end
    return p_fit, fit, pbest, gbest, velocity_best, velocity
end
