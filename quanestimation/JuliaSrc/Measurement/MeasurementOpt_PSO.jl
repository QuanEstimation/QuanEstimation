function PSO_CFIM(pso::MeasurementOpt{T}, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file) where {T<:Complex}
    println("measurement optimization")
    Random.seed!(seed)
    dim = size(pso.ρ0)[1]
    M_num = length(pso.Measurement)
    particles = repeat(pso, particle_num)
    velocity = 0.1*rand(ComplexF64, M_num, dim, particle_num)
    pbest = zeros(ComplexF64, M_num, dim, particle_num)
    gbest = zeros(ComplexF64, M_num, dim)
    velocity_best = zeros(ComplexF64, M_num, dim)
    p_fit = zeros(particle_num)
    fit = 0.0

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    # initialize 
    if length(ini_particle) > particle_num
        ini_particle = [ini_particle[i] for i in 1:particle_num]
    end 
    for pj in 1:length(ini_particle)
        particles[pj].Measurement = [[ini_particle[pj][i,j] for j in 1:dim] for i in 1:M_num]
    end
    for pj in (length(ini_particle)+1):particle_num
        M_tp = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
        for mi in 1:M_num
            r_ini = 2*rand(dim)-ones(dim)
            r = r_ini/norm(r_ini)
            phi = 2*pi*rand(dim)
            M_tp[mi] = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
        end
        particles[pj].Measurement = [[M_tp[i][j] for j in 1:dim] for i in 1:M_num]
        # orthogonality and normalization 
        particles[pj].Measurement = GramSchmidt(particles[pj].Measurement)
    end

    p_fit = [0.0 for i in 1:particle_num] 
    for pj in 1:particle_num
        Measurement = [particles[pj].Measurement[i]*(particles[pj].Measurement[i])' for i in 1:M_num]
        F_tp = CFIM(Measurement, pso.freeHamiltonian, pso.Hamiltonian_derivative, pso.ρ0, pso.decay_opt, pso.γ, pso.tspan, pso.accuracy)
        p_fit[pj] = 1.0/real(tr(pso.W*pinv(F_tp)))
    end

    f_ini= p_fit[1]
    F_opt = QFIM(pso.freeHamiltonian, pso.Hamiltonian_derivative, pso.ρ0, pso.decay_opt, pso.γ, pso.tspan, pso.accuracy)
    f_opt= real(tr(pso.W*pinv(F_opt)))

    if length(pso.Hamiltonian_derivative) == 1
        f_list = [f_ini]

        println("single parameter scenario")
        println("search algorithm: Particle Swarm Optimization (PSO)")
        println("initial CFI is $(f_ini)")
        println("QFI is $(1.0/f_opt)")
        
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                               M_num, dim, pbest, gbest, velocity_best, velocity)
                if ei%max_episode[2] == 0
                    pso.Measurement = [gbest[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                SaveFile_meas(f_list, gbest)
                print("current CFI is $fit ($ei episodes) \r")
            end
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                           M_num, dim, pbest, gbest, velocity_best, velocity)
            append!(f_list, fit)
            SaveFile_meas(f_list, gbest)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final CFI is $fit")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                               M_num, dim, pbest, gbest, velocity_best, velocity)
                if ei%max_episode[2] == 0
                    pso.Measurement = [gbest[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                print("current CFI is $fit ($ei episodes) \r")
                
            end
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                           M_num, dim, pbest, gbest, velocity_best, velocity)
            append!(f_list, fit)
            SaveFile_meas(f_list, gbest)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final CFI is $fit")
        end
    else
        f_list = [1.0/f_ini]
        println("multiparameter scenario")
        println("search algorithm: Particle Swarm Optimization (PSO)")
        println("initial value of Tr(WI^{-1}) is $(1.0/f_ini)")
        println("Tr(WF^{-1}) is $(f_opt)")

        if save_file == true
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                               M_num, dim, pbest, gbest, velocity_best, velocity)
                if ei%max_episode[2] == 0
                    pso.Measurement = [gbest[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, 1.0/fit)
                SaveFile_meas(f_list, gbest)
                print("current value of Tr(WI^{-1}) is $(1.0/fit) ($ei episodes) \r")
            end
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                           M_num, dim, pbest, gbest, velocity_best, velocity)
            append!(f_list, 1.0/fit)
            SaveFile_meas(f_list, gbest)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final value of Tr(WI^{-1}) is $(1.0/fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                           M_num, dim, pbest, gbest, velocity_best, velocity)
                if ei%max_episode[2] == 0
                    pso.Measurement = [gbest[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, 1.0/fit)
                print("current value of Tr(WI^{-1}) is $fit ($ei episodes) \r")
                
            end
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_CFIM(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                           M_num, dim, pbest, gbest, velocity_best, velocity)
            append!(f_list, 1.0/fit)
            SaveFile_meas(f_list, gbest)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final value of Tr(WI^{-1}) is $(1.0/fit)")
        end
    end
end

function train_CFIM(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity_best, velocity)
    for pj in 1:particle_num
        Measurement = [particles[pj].Measurement[i]*(particles[pj].Measurement[i])' for i in 1:M_num]
        F_tp = CFIM(Measurement, particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative, particles[pj].ρ0, 
                    particles[pj].decay_opt, particles[pj].γ, particles[pj].tspan, particles[pj].accuracy)
        f_now = 1.0/real(tr(particles[pj].W*pinv(F_tp)))

        if f_now > p_fit[pj]
            p_fit[pj] = f_now
            for di in 1:M_num
                for ni in 1:dim
                    pbest[di,ni,pj] = particles[pj].Measurement[di][ni]
                end
            end
        end

        for pj in 1:particle_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                for dj in 1:M_num
                    for nj in 1:dim
                        gbest[dj, nj] = particles[pj].Measurement[dj][nj]
                        velocity_best[dj, nj] = velocity[dj, nj, pj]
                    end
                end
            end
        end  

        for pk in 1:particle_num
            meas_pre = [zeros(ComplexF64, dim) for i in 1:M_num]
            for dk in 1:M_num
                for ck in 1:dim
                    meas_pre[dk][ck] = particles[pk].Measurement[dk][ck]
    
                    velocity[dk, ck, pk] = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].Measurement[dk][ck]) 
                                           + c2*rand()*(gbest[dk, ck] - particles[pk].Measurement[dk][ck])
                    particles[pk].Measurement[dk][ck] += velocity[dk, ck, pk]
                end
            end
            particles[pk].Measurement = GramSchmidt(particles[pk].Measurement)

            for dm in 1:M_num
                for cm in 1:dim
                    velocity[dm, cm, pk] = particles[pk].Measurement[dm][cm] - meas_pre[dm][cm]
                end
            end
        end
    end
    return p_fit, fit, pbest, gbest, velocity_best, velocity
end
