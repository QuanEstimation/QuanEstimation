
function SM_PSO_Compopt(pso::SM_Compopt_Kraus{T}, max_episode, particle_num, psi0, measurement0, c0, c1, c2, seed, save_file) where {T<: Complex}
    sym = Symbol("CFIM_SMopt_Kraus")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    M = [zeros(ComplexF64, size(pso.psi)[1], size(pso.psi)[1])]
    return info_PSO_SMopt_Kraus(M, pso, max_episode, particle_num, psi0, measurement0, c0, c1, c2, seed, save_file, sym, str1, str2, str3)
end

function info_PSO_SMopt_Kraus(M, pso, max_episode, particle_num, psi0, measurement0, c0, c1, c2, seed, save_file, sym, str1, str2, str3) where {T<:Complex}
    println("comprehensive optimization")
    Random.seed!(seed)
    dim = length(pso.psi)
    M_num = length(pso.C)
    particles = repeat(pso, particle_num)

    velocity_state = 0.1.*rand(ComplexF64, dim, particle_num)
    pbest_state = zeros(ComplexF64, dim, particle_num)
    gbest_state = zeros(ComplexF64, dim)

    velocity_meas = 0.1*rand(ComplexF64, M_num, dim, particle_num)
    pbest_meas = zeros(ComplexF64, M_num, dim, particle_num)
    gbest_meas = zeros(ComplexF64, M_num, dim)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    # initialize
    if length(psi0) > particle_num
        psi0 = [psi0[i] for i in 1:particle_num]
    end
    if length(measurement0) > particle_num
        measurement0 = [measurement0[i] for i in 1:particle_num]
    end 

    for pj in (length(psi0)+1):particle_num
        r_ini = 2*rand(dim)-ones(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    for pj in 1:length(measurement0)
        particles[pj].C = [[measurement0[pj][i,j] for j in 1:dim] for i in 1:M_num]
    end
    for pj in (length(measurement0)+1):particle_num
        M_tp = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
        for mi in 1:M_num
            r_ini = 2*rand(dim)-ones(dim)
            r = r_ini/norm(r_ini)
            phi = 2*pi*rand(dim)
            M_tp[mi] = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
        end
        particles[pj].C = [[M_tp[i][j] for j in 1:dim] for i in 1:M_num]
        # orthogonality and normalization 
        particles[pj].C = gramschmidt(particles[pj].C)
    end
    
    p_fit = [0.0 for i in 1:particle_num] 
    for pj in 1:particle_num
        M = [particles[pj].C[i]*(particles[pj].C[i])' for i in 1:M_num]
        p_fit[pj] = 1.0/obj_func(Val{sym}(), pso, particles[pj].psi, M)
    end

    f_ini= p_fit[1]

    fit = 0.0
    if length(pso.dK) == 1
        println("single parameter scenario")
        println("control algorithm: Particle Swarm Optimization (PSO)")
        println("initial $str2 is $(1.0/f_ini)")
        f_list = [1.0/f_ini]
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_state, velocity_meas = train_PSO_SMopt(M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, 
                pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_state, velocity_meas, sym)
                if ei%max_episode[2] == 0
                    pso.psi = [gbest_state[i] for i in 1:dim]
                    pso.C = [gbest_meas[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
                append!(f_list, fit)
                SaveFile_SM(f_list, gbest_state, M)
                print("current $str2 is $(fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_state, velocity_meas = train_PSO_SMopt(M, 
            particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, 
            pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_state, velocity_meas, sym)         
            M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
            append!(f_list, fit)
            SaveFile_SM(f_list, gbest_state, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str2 is $(fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_state, velocity_meas = train_PSO_SMopt(M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num,
                pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_state, velocity_meas, sym)               
                if ei%max_episode[2] == 0
                    pso.psi = [gbest_state[i] for i in 1:dim]
                    pso.C = [gbest_meas[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                print("current $str2 is $(fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_state, velocity_meas = train_PSO_SMopt(M, 
            particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num,
            pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_state, velocity_meas, sym)         
            M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
            append!(f_list, fit)
            SaveFile_SM(f_list, gbest_state, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str2 is $(fit)")
        end
    else
        println("multiparameter scenario")
        println("control algorithm: Particle Swarm Optimization (PSO)")
        println("initial value of $str3 is $(f_ini)")
        f_list = [f_ini]
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_state, velocity_meas = train_PSO_SMopt(M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num,
                pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_state, velocity_meas, sym)               
                if ei%max_episode[2] == 0
                    pso.psi = [gbest_state[i] for i in 1:dim]
                    pso.C = [gbest_meas[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
                append!(f_list, 1.0/fit)
                SaveFile_SM(f_list, gbest_state, M)
                print("current value of $str3 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_state, velocity_meas = train_PSO_SMopt(M, 
            particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num,
            pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_state, velocity_meas, sym)           
            M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_SM(f_list, gbest_state, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is $(1.0/fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_state, velocity_meas = train_PSO_SMopt(M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num,
                pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_state, velocity_meas, sym)               
                if ei%max_episode[2] == 0
                    pso.psi = [gbest_state[i] for i in 1:dim]
                    pso.C = [gbest_meas[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, 1.0/fit)
                print("current value of $str3 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_state, velocity_meas = train_PSO_SMopt(M, 
            particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num,
            pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_state, velocity_meas, sym)            
            M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_SM(f_list, gbest_state, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is $(1.0/fit)")
        end
    end
end
