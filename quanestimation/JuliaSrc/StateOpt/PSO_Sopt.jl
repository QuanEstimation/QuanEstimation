############# time-independent Hamiltonian (noiseless) ################
function QFIM_PSO_Sopt(pso::TimeIndepend_noiseless{T}, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file) where {T<:Complex}
    sym = Symbol("QFIM_TimeIndepend_noiseless")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    M = [zeros(ComplexF64, size(pso.psi)[1], size(pso.psi)[1])]
    return info_PSO_noiseless(M, pso, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file, sym, str1, str2, str3)
end

function CFIM_PSO_Sopt(M, pso::TimeIndepend_noiseless{T}, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file) where {T<:Complex}
    sym = Symbol("CFIM_TimeIndepend_noiseless")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_PSO_noiseless(M, pso, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file, sym, str1, str2, str3)
end

function HCRB_PSO_Sopt(pso::TimeIndepend_noiseless{T}, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file) where {T<:Complex}
    sym = Symbol("HCRB_TimeIndepend_noiseless")
    str1 = ""
    str2 = "HCRB"
    str3 = "HCRB"
    M = [zeros(ComplexF64, size(pso.psi)[1], size(pso.psi)[1])]
    if length(pso.Hamiltonian_derivative) == 1
        println("In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the objection function for state optimization.")
        return nothing
    else
        return info_PSO_noiseless(M, pso, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file, sym, str1, str2, str3)
    end
end

function info_PSO_noiseless(M, pso, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file, sym, str1, str2, str3) where {T<: Complex}
    println("$str1 state optimization")
    Random.seed!(seed)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*rand(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
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
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end
    for pj in (length(ini_particle)+1):particle_num
        r_ini = 2*rand(dim)-ones(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    qfi_ini = obj_func(Val{sym}(), pso, M)

    if length(pso.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Particle Swarm Optimization (PSO)")
        println("initial $str2 is $(1.0/qfi_ini)")      
        f_list = [1.0/qfi_ini]
        if save_file==true
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, pbest, gbest, velocity = train_noiseless_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.psi = [gbest[i] for i in 1:dim]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                SaveFile_state(f_list, gbest)
                print("current $str2 is $fit ($ei episodes)    \r")
                
            end
            p_fit, fit, pbest, gbest, velocity = train_noiseless_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                                dim, pbest, gbest, velocity, sym)
            append!(f_list, fit)
            SaveFile_state(f_list, gbest)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final $str2 is $fit")
        else
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, pbest, gbest, velocity = train_noiseless_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.psi = [gbest[i] for i in 1:dim]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                print("current $str2 is $fit ($ei episodes)    \r")
            end
            p_fit, fit, pbest, gbest, velocity = train_noiseless_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                                dim, pbest, gbest, velocity, sym)
            append!(f_list, fit)
            SaveFile_state(f_list, gbest)
            print("\e[2K") 
            println("Iteration over, data saved.") 
            println("Final $str2 is $fit")
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Particle Swarm Optimization (PSO)")
        println("initial value of $str3 is $(qfi_ini)")       
        f_list = [qfi_ini]
        if save_file==true
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, pbest, gbest, velocity = train_noiseless_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.psi = [gbest[i] for i in 1:dim]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, (1.0/fit))
                SaveFile_state(f_list, gbest)
                print("current value of $str3 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest, gbest, velocity = train_noiseless_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                                dim, pbest, gbest, velocity, sym)
            append!(f_list, (1.0/fit))
            SaveFile_state(f_list, gbest)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is $(1.0/fit)")
        else
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, pbest, gbest, velocity = train_noiseless_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                                    dim, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.psi = [gbest[i] for i in 1:dim]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, (1.0/fit))
                print("current value of $str3 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest, gbest, velocity = train_noiseless_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                                dim, pbest, gbest, velocity, sym)
            append!(f_list, (1.0/fit))
            SaveFile_state(f_list, gbest)
            print("\e[2K") 
            println("Iteration over, data saved.") 
            println("Final value of $str3 is $(1.0/fit)")
        end
    end
end

function train_noiseless_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, pbest, gbest, velocity, sym)
    for pj in 1:particle_num
        f_now = obj_func(Val{sym}(), particles[pj], M) 
        f_now = 1.0/f_now
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
                gbest[dj] = pbest[dj,pj]
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
    return p_fit, fit, pbest, gbest, velocity
end


############# time-independent Hamiltonian (noise) ################
function QFIM_PSO_Sopt(pso::TimeIndepend_noise{T}, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file) where {T<:Complex}
    sym = Symbol("QFIM_TimeIndepend_noise")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    M = [zeros(ComplexF64, size(pso.psi)[1], size(pso.psi)[1])]
    return info_PSO_noise(M, pso, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file, sym, str1, str2, str3)
end

function CFIM_PSO_Sopt(M, pso::TimeIndepend_noise{T}, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file) where {T<:Complex}
    sym = Symbol("CFIM_TimeIndepend_noise")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_PSO_noise(M, pso, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file, sym, str1, str2, str3)
end

function HCRB_PSO_Sopt(pso::TimeIndepend_noise{T}, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file) where {T<:Complex}
    sym = Symbol("HCRB_TimeIndepend_noise")
    str1 = ""
    str2 = "HCRB"
    str3 = "HCRB"
    M = [zeros(ComplexF64, size(pso.psi)[1], size(pso.psi)[1])]
    if length(pso.Hamiltonian_derivative) == 1
        println("In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the objection function for state optimization.")
        return nothing
    else
        return info_PSO_noise(M, pso, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file, sym, str1, str2, str3)
    end
end

function info_PSO_noise(M, pso, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file, sym, str1, str2, str3) where {T<:Complex}
    println("$str1 state optimization")
    Random.seed!(seed)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*rand(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
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
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end
    for pj in (length(ini_particle)+1):particle_num
        r_ini = 2*rand(dim)-ones(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    qfi_ini = obj_func(Val{sym}(), pso, M)

    if length(pso.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Particle Swarm Optimization (PSO)")
        println("initial $str2 is $(1.0/qfi_ini)")
        f_list = [1.0/qfi_ini]
        if save_file==true
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, pbest, gbest, velocity = train_noise_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                                dim, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.psi = [gbest[i] for i in 1:dim]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                SaveFile_state(f_list, gbest)
                print("current $str2 is $fit ($ei episodes)    \r")

            end
            p_fit, fit, pbest, gbest, velocity = train_noise_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                            dim, pbest, gbest, velocity, sym)
            append!(f_list, fit)
            SaveFile_state(f_list, gbest)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str2 is $fit")
        else
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, pbest, gbest, velocity = train_noise_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                                dim, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.psi = [gbest[i] for i in 1:dim]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                print("current $str2 is $fit ($ei episodes)    \r")
            end
            p_fit, fit, pbest, gbest, velocity = train_noise_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                            dim, pbest, gbest, velocity, sym)
            append!(f_list, fit)
            SaveFile_state(f_list, gbest)
            print("\e[2K") 
            println("Iteration over, data saved.") 
            println("Final $str2 is $fit")
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Particle Swarm Optimization (PSO)")
        println("initial value of $str3 is $(qfi_ini)")
        f_list = [qfi_ini]
        if save_file==true
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, pbest, gbest, velocity = train_noise_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                                dim, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.psi = [gbest[i] for i in 1:dim]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, (1.0/fit))
                SaveFile_state(f_list, gbest)
                print("current value of $str3 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest, gbest, velocity = train_noise_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                            dim, pbest, gbest, velocity, sym)
            append!(f_list, (1.0/fit))
            SaveFile_state(f_list, gbest)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is $(1.0/fit)")
        else
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, pbest, gbest, velocity = train_noise_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                                dim, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.psi = [gbest[i] for i in 1:dim]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, (1.0/fit))
                print("current value of $str3 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest, gbest, velocity = train_noise_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                            dim, pbest, gbest, velocity, sym)
            append!(f_list, (1.0/fit))
            SaveFile_state(f_list, gbest)
            print("\e[2K") 
            println("Iteration over, data saved.") 
            println("Final value of $str3 is $(1.0/fit)")
        end
    end
end

function train_noise_PSO(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, pbest, gbest, velocity, sym)
    for pj in 1:particle_num
        f_now = obj_func(Val{sym}(), particles[pj], M)
        f_now = 1.0/f_now
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
                gbest[dj] = pbest[dj,pj]
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
    return p_fit, fit, pbest, gbest, velocity
end
