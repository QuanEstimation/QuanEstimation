################ projection measurement ###############
function CFIM_PSO_Mopt(pso::projection_Mopt{T}, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file) where {T<:Complex}
    sym = Symbol("CFIM_noctrl")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_PSO_projection(pso, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file, sym, str1, str2)
end

function info_PSO_projection(pso, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file, sym, str1, str2) where {T<:Complex}
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
        particles[pj].Measurement = gramschmidt(particles[pj].Measurement)
    end

    p_fit = [0.0 for i in 1:particle_num] 
    for pj in 1:particle_num
        Measurement = [particles[pj].Measurement[i]*(particles[pj].Measurement[i])' for i in 1:M_num]
        F_tp = obj_func(Val{sym}(), pso, Measurement)
        p_fit[pj] = 1.0/real(tr(pso.W*pinv(F_tp)))
    end

    f_ini= p_fit[1]
    F_opt = obj_func(Val{:QFIM_noctrl}(), pso, pso.Measurement)
    f_opt= 1.0/real(tr(pso.W*pinv(F_opt)))

    if length(pso.Hamiltonian_derivative) == 1
        f_list = [f_ini]

        println("single parameter scenario")
        println("search algorithm: Particle Swarm Optimization (PSO)")
        println("initial $str1 is $(f_ini)")
        println("QFI is $(f_opt)")
        
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, pbest, gbest, velocity_best, velocity = train_projection(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                               M_num, dim, pbest, gbest, velocity_best, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.Measurement = [gbest[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                Measurement = [gbest[i]*gbest[i]' for i in 1:M_num]
                append!(f_list, fit)
                SaveFile_meas(f_list, Measurement)
                print("current $str1 is $fit ($ei episodes) \r")
            end
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_projection(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                           M_num, dim, pbest, gbest, velocity_best, velocity, sym)
            Measurement = [gbest[i]*gbest[i]' for i in 1:M_num]
            append!(f_list, fit)
            SaveFile_meas(f_list, Measurement)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final $str1 is $fit")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity_best, velocity = train_projection(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                               M_num, dim, pbest, gbest, velocity_best, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.Measurement = [gbest[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                print("current $str1 is $fit ($ei episodes) \r")
                
            end
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_projection(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                           M_num, dim, pbest, gbest, velocity_best, velocity, sym)
            Measurement = [gbest[i]*gbest[i]' for i in 1:M_num]
            append!(f_list, fit)
            SaveFile_meas(f_list, Measurement)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final $str1 is $fit")
        end
    else
        f_list = [1.0/f_ini]
        println("multiparameter scenario")
        println("search algorithm: Particle Swarm Optimization (PSO)")
        println("initial value of $str2 is $(1.0/f_ini)")
        println("Tr(WF^{-1}) is $(1.0/f_opt)")

        if save_file == true
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, pbest, gbest, velocity_best, velocity = train_projection(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                               M_num, dim, pbest, gbest, velocity_best, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.Measurement = [gbest[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                Measurement = [gbest[i]*gbest[i]' for i in 1:M_num]
                append!(f_list, 1.0/fit)
                SaveFile_meas(f_list, Measurement)
                print("current value of $str2 is $(1.0/fit) ($ei episodes) \r")
            end
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_projection(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                           M_num, dim, pbest, gbest, velocity_best, velocity, sym)
            Measurement = [gbest[i]*gbest[i]' for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_meas(f_list, Measurement)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final value of $str2 is $(1.0/fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity_best, velocity = train_projection(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                           M_num, dim, pbest, gbest, velocity_best, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.Measurement = [gbest[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, 1.0/fit)
                print("current value of $str2 is $fit ($ei episodes) \r")
                
            end
            p_fit, fit, pbest, gbest, velocity_best, velocity = train_projection(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                           M_num, dim, pbest, gbest, velocity_best, velocity, sym)
            Measurement = [gbest[i]*gbest[i]' for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_meas(f_list, Measurement)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final value of $str2 is $(1.0/fit)")
        end
    end
end

function train_projection(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity_best, velocity, sym)
    for pj in 1:particle_num
        Measurement = [particles[pj].Measurement[i]*(particles[pj].Measurement[i])' for i in 1:M_num]
        F_tp = obj_func(Val{sym}(), particles[pj], Measurement)
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
            particles[pk].Measurement = gramschmidt(particles[pk].Measurement)

            for dm in 1:M_num
                for cm in 1:dim
                    velocity[dm, cm, pk] = particles[pk].Measurement[dm][cm] - meas_pre[dm][cm]
                end
            end
        end
    end
    return p_fit, fit, pbest, gbest, velocity_best, velocity
end

################## update the coefficients according to the given basis ############
function CFIM_PSO_Mopt(pso::givenpovm_Mopt{T}, max_episode, particle_num, c0, c1, c2, seed, save_file) where {T<:Complex}
    sym = Symbol("CFIM_noctrl")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_PSO_givenpovm(pso, max_episode, particle_num, c0, c1, c2, seed, save_file, sym, str1, str2)
end

function info_PSO_givenpovm(pso::givenpovm_Mopt{T}, max_episode, particle_num, c0, c1, c2, seed, save_file, sym, str1, str2) where {T<:Complex}
    println("measurement optimization")
    Random.seed!(seed)
    dim = size(pso.ρ0)[1]
    POVM_basis = pso.povm_basis
    M_num = pso.M_num
    particles = repeat(pso, particle_num)
    velocity = 0.1*rand(Float64, M_num, dim^2, particle_num)
    pbest = zeros(Float64, M_num, dim^2, particle_num)
    gbest = zeros(Float64, M_num, dim^2)
    velocity_best = zeros(Float64, M_num, dim^2)
    p_fit = zeros(particle_num)
    fit = 0.0

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    # initialize 
    coeff = [[zeros(dim^2) for i in 1:M_num] for j in 1:particle_num]
    for pj in 1:particle_num
        coeff[pj] = generate_coeff(M_num, dim)
    end

    p_fit = [0.0 for i in 1:particle_num] 
    for pj in 1:particle_num
        Measurement = [sum([coeff[pj][i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
        F_tp = obj_func(Val{sym}(), pso, Measurement)
        p_fit[pj] = 1.0/real(tr(pso.W*pinv(F_tp)))
    end

    f_ini= p_fit[1]
    F_opt = obj_func(Val{:QFIM_noctrl}(), pso, POVM_basis)
    f_opt= 1.0/real(tr(pso.W*pinv(F_opt)))

    F_povm = obj_func(Val{sym}(), pso, POVM_basis)
    f_povm= 1.0/real(tr(pso.W*pinv(F_povm)))

    if length(pso.Hamiltonian_derivative) == 1
        f_list = [f_ini]

        println("single parameter scenario")
        println("search algorithm: Particle Swarm Optimization (PSO)")
        println("initial $str1 is $(f_ini)")
        println("CFI under the given POVMs is $(f_povm)")
        println("QFI is $(f_opt)")
        
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, coeff, pbest, gbest, velocity_best, velocity = train_givenpovm(particles, p_fit, fit, coeff, POVM_basis, max_episode, c0, c1, c2, particle_num, 
                                                                               M_num, dim, pbest, gbest, velocity_best, velocity, sym)
                if ei%max_episode[2] == 0
                    particles = repeat(pso, particle_num)
                    for i in 1:particle_num
                        coeff[i] = [gbest[k, :] for k in 1:M_num]
                    end
                end
                Measurement = [sum([gbest[i,j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
                append!(f_list, fit)
                SaveFile_meas(f_list, Measurement)
                print("current $str1 is $fit ($ei episodes) \r")
            end
            p_fit, fit, coeff, pbest, gbest, velocity_best, velocity = train_givenpovm(particles, p_fit, fit, coeff, POVM_basis, 
                                             max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity_best, velocity, sym)
            Measurement = [sum([gbest[i,j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
            append!(f_list, fit)
            SaveFile_meas(f_list, Measurement)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final $str1 is $fit")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, coeff, pbest, gbest, velocity_best, velocity = train_givenpovm(particles, p_fit, fit, coeff, POVM_basis, 
                                                 max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity_best, velocity, sym)
                if ei%max_episode[2] == 0
                    particles = repeat(pso, particle_num)
                    for i in 1:particle_num
                        coeff[i] = [gbest[k, :] for k in 1:M_num]
                    end
                end
                append!(f_list, fit)
                print("current $str1 is $fit ($ei episodes) \r")
                
            end
            p_fit, fit, coeff, pbest, gbest, velocity_best, velocity = train_givenpovm(particles, p_fit, fit, coeff, POVM_basis, 
                                              max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity_best, velocity, sym)
            Measurement = [sum([gbest[i,j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
            append!(f_list, fit)
            SaveFile_meas(f_list, Measurement)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final $str1 is $fit")
        end
    else
        f_list = [1.0/f_ini]
        println("multiparameter scenario")
        println("search algorithm: Particle Swarm Optimization (PSO)")
        println("initial value of $str2 is $(1.0/f_ini)")
        println("Tr(WI^{-1}) under the given POVMs is $(1.0/f_povm)")
        println("Tr(WF^{-1}) is $(1.0/f_opt)")

        if save_file == true
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, coeff, pbest, gbest, velocity_best, velocity = train_givenpovm(particles, p_fit, fit, coeff, POVM_basis, 
                                                 max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity_best, velocity, sym)
                if ei%max_episode[2] == 0
                    particles = repeat(pso, particle_num)
                    for i in 1:particle_num
                        coeff[i] = [gbest[k, :] for k in 1:M_num]
                    end
                end
                Measurement = [sum([gbest[i,j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
                append!(f_list, 1.0/fit)
                SaveFile_meas(f_list, Measurement)
                print("current value of $str2 is $(1.0/fit) ($ei episodes) \r")
            end
            p_fit, fit, coeff, pbest, gbest, velocity_best, velocity = train_givenpovm(particles, p_fit, fit, coeff, POVM_basis, 
                                              max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity_best, velocity, sym)
            Measurement = [sum([gbest[i,j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_meas(f_list, Measurement)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final value of $str2 is $(1.0/fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, coeff, pbest, gbest, velocity_best, velocity = train_givenpovm(particles, p_fit, fit, coeff, POVM_basis, 
                                                 max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity_best, velocity, sym)
                if ei%max_episode[2] == 0
                    particles = repeat(pso, particle_num)
                    for i in 1:particle_num
                        coeff[i] = [gbest[k, :] for k in 1:M_num]
                    end
                end
                append!(f_list, 1.0/fit)
                print("current value of $str2 is $fit ($ei episodes) \r")
                
            end
            p_fit, fit, coeff, pbest, gbest, velocity_best, velocity = train_givenpovm(particles, p_fit, fit, coeff, POVM_basis, 
                                             max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity_best, velocity, sym)
            Measurement = [sum([gbest[i,j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_meas(f_list, Measurement)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final value of $str2 is $(1.0/fit)")
        end
    end
end

function train_givenpovm(particles, p_fit, fit, coeff, POVM_basis, max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity_best, velocity, sym)
    for pj in 1:particle_num
        Measurement = [sum([coeff[pj][i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
        F_tp = obj_func(Val{sym}(), particles[pj], Measurement)
        f_now = 1.0/real(tr(particles[pj].W*pinv(F_tp)))

        if f_now > p_fit[pj]
            p_fit[pj] = f_now
            for di in 1:M_num
                for ni in 1:dim^2
                    pbest[di,ni,pj] = coeff[pj][di][ni]
                end
            end
        end

        for pj in 1:particle_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                for dj in 1:M_num
                    for nj in 1:dim^2
                        gbest[dj, nj] = coeff[pj][dj][nj]
                        velocity_best[dj, nj] = velocity[dj, nj, pj]
                    end
                end
            end
        end  

        for pk in 1:particle_num
            meas_pre = [zeros(Float64, dim^2) for i in 1:M_num]
            for dk in 1:M_num
                for ck in 1:dim^2
                    meas_pre[dk][ck] = coeff[pk][dk][ck]
    
                    velocity[dk, ck, pk] = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - coeff[pk][dk][ck]) 
                                           + c2*rand()*(gbest[dk, ck] - coeff[pk][dk][ck])
                    coeff[pk][dk][ck] += velocity[dk, ck, pk]
                end
            end
            bound!(coeff[pk])
            for i in 1:M_num
                for j in 1:dim^2
                    coeff[pk][i][j] = coeff[pk][i][j]/sum([coeff[pk][m][j] for m in 1:M_num])
                end
            end
            Measurement = [sum([coeff[pk][i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]

            for dm in 1:M_num
                for cm in 1:dim^2
                    velocity[dm, cm, pk] = coeff[pk][dm][cm] - meas_pre[dm][cm]
                end
            end
        end
    end
    return p_fit, fit, coeff, pbest, gbest, velocity_best, velocity
end

function generate_coeff(M_num, dim)
    coeff_tp = [rand(dim^2) for i in 1:M_num]
    vec_tp = ones(dim^2)
    for i in 2:(M_num-1)
        vec_tp -= [coeff_tp[i-1][m] for m in 1:dim^2]
        coeff_tp[i] = [coeff_tp[i][n]*vec_tp[n] for n in 1:dim^2]
    end
    coeff_tp[end] = [1.0-sum([coeff_tp[i][j] for i in 1:(M_num-1)]) for j in 1:dim^2]
    return coeff_tp
end
