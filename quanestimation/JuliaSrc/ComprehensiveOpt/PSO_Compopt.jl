################ state and control optimization ###############
function PSO_Compopt_SCopt(pso::Compopt_SCopt{T}, max_episode, particle_num, psi0, ctrl0, c0, c1, c2, seed, save_file) where {T<: Complex}
    sym = Symbol("QFIM_SCopt")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    M = [zeros(ComplexF64, size(pso.psi)[1], size(pso.psi)[1])]
    return info_PSO_SCopt(M, pso, max_episode, particle_num, psi0, ctrl0, c0, c1, c2, seed, save_file, sym, str1, str2, str3)
end

function PSO_Compopt_SCopt(M, pso::Compopt_SCopt{T}, max_episode, particle_num, psi0, ctrl0, c0, c1, c2, seed, save_file) where {T<: Complex}
    sym = Symbol("CFIM_SCopt")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_PSO_SCopt(M, pso, max_episode, particle_num, psi0, ctrl0, c0, c1, c2, seed, save_file, sym, str1, str2, str3)
end

function info_PSO_SCopt(M, pso, max_episode, particle_num, psi0, ctrl0, c0, c1, c2, seed, save_file, sym, str1, str2, str3) where {T<:Complex}
    println("comprehensive optimization")
    Random.seed!(seed)
    dim = length(pso.psi)
    ctrl_length = length(pso.control_coefficients[1])
    ctrl_num = length(pso.control_Hamiltonian)
    particles = repeat(pso, particle_num)

    velocity_state = 0.1.*rand(ComplexF64, dim, particle_num)
    pbest_state = zeros(ComplexF64, dim, particle_num)
    gbest_state = zeros(ComplexF64, dim)
    velocity_best_state = zeros(ComplexF64, dim)

    if pso.ctrl_bound[1] == -Inf || pso.ctrl_bound[2] == Inf
        velocity_ctrl = 0.1*(2.0*rand(ctrl_num, ctrl_length, particle_num)-ones(ctrl_num, ctrl_length, particle_num))
    else
        a = pso.ctrl_bound[1]
        b = pso.ctrl_bound[2]
        velocity_ctrl = 0.5*((b-a)*rand(ctrl_num, ctrl_length, particle_num)+a*ones(ctrl_num, ctrl_length, particle_num))
    end
    pbest_ctrl = zeros(ctrl_num, ctrl_length, particle_num)
    gbest_ctrl = zeros(ctrl_num, ctrl_length)
    velocity_best_ctrl = zeros(ctrl_num,ctrl_length)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    # initialize
    if length(psi0) > particle_num
        psi0 = [psi0[i] for i in 1:particle_num]
    end
    if length(ctrl0) > particle_num
        ctrl0 = [ctrl0[i] for i in 1:particle_num]
    end

    for pj in (length(psi0)+1):particle_num
        r_ini = 2*rand(dim)-ones(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    for pj in 1:length(ctrl0)
        particles[pj].control_coefficients = [[ctrl0[pj][i,j] for j in 1:ctrl_length] for i in 1:ctrl_num]
    end
    if pso.ctrl_bound[1] == -Inf || pso.ctrl_bound[2] == Inf
        for pj in (length(ctrl0)+1):particle_num
            particles[pj].control_coefficients = [[2*rand()-1.0 for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
    else
        a = pso.ctrl_bound[1]
        b = pso.ctrl_bound[2]
        for pj in (length(ctrl0)+1):particle_num
            particles[pj].control_coefficients = [[(b-a)*rand()+a for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
    end

    p_fit = [0.0 for i in 1:particle_num] 
    for pj in 1:particle_num
        p_fit[pj] = obj_func(Val{sym}(), pso, M, particles[pj].psi, particles[pj].control_coefficients)
    end

    f_ini= p_fit[1]

    fit = 0.0
    if length(pso.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: Particle Swarm Optimization (PSO)")
        println("initial $str2 is $(1.0/f_ini)")
        f_list = [1.0/f_ini]
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl = train_PSO_SCopt(M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, ctrl_num, ctrl_length, 
                pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl, sym)
                if ei%max_episode[2] == 0
                    pso.psi = [gbest_state[i] for i in 1:dim]
                    pso.control_coefficients = [gbest_ctrl[k, :] for k in 1:ctrl_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                SaveFile_SC(f_list, gbest_state, [gbest_ctrl[k, :] for k in 1:ctrl_num])
                print("current $str2 is $(fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl = train_PSO_SCopt(M, 
            particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, ctrl_num, ctrl_length, 
            pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl, sym)         
            append!(f_list, fit)
            SaveFile_SC(f_list, gbest_state, [gbest_ctrl[k, :] for k in 1:ctrl_num])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str2 is $(fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl = train_PSO_SCopt(M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, ctrl_num, ctrl_length, 
                pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl, sym)               
                if ei%max_episode[2] == 0
                    pso.psi = [gbest_state[i] for i in 1:dim]
                    pso.control_coefficients = [gbest_ctrl[k, :] for k in 1:ctrl_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                print("current $str2 is $(fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl = train_PSO_SCopt(M, 
            particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, ctrl_num, ctrl_length, 
            pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl, sym)         
            append!(f_list, fit)
            SaveFile_SC(f_list, gbest_state, [gbest_ctrl[k, :] for k in 1:ctrl_num])
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
                p_fit, fit, pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl = train_PSO_SCopt(M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, ctrl_num, ctrl_length, 
                pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl, sym)               
                if ei%max_episode[2] == 0
                    pso.psi = [gbest_state[i] for i in 1:dim]
                    pso.control_coefficients = [gbest_ctrl[k, :] for k in 1:ctrl_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, 1.0/fit)
                SaveFile_SC(f_list, gbest_state, [gbest_ctrl[k, :] for k in 1:ctrl_num])
                print("current value of $str3 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl = train_PSO_SCopt(M, 
            particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, ctrl_num, ctrl_length, 
            pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl, sym)           
            append!(f_list, 1.0/fit)
            SaveFile_SC(f_list, gbest_state, [gbest_ctrl[k, :] for k in 1:ctrl_num])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is $(1.0/fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl = train_PSO_SCopt(M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, ctrl_num, ctrl_length, 
                pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl, sym)               
                if ei%max_episode[2] == 0
                    pso.psi = [gbest_state[i] for i in 1:dim]
                    pso.control_coefficients = [gbest_ctrl[k, :] for k in 1:ctrl_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, 1.0/fit)
                print("current value of $str3 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl = train_PSO_SCopt(M, 
            particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, ctrl_num, ctrl_length, 
            pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl, sym)            
            append!(f_list, 1.0/fit)
            SaveFile_SC(f_list, gbest_state, [gbest_ctrl[k, :] for k in 1:ctrl_num])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is $(1.0/fit)")
        end
    end
end

function train_PSO_SCopt(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, ctrl_num, ctrl_length, pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl, sym)
    for pj in 1:particle_num
        f_tp = obj_func(Val{sym}(), particles[pj], M, particles[pj].psi, particles[pj].control_coefficients)
        f_now = 1.0/f_tp
        if f_now > p_fit[pj]
            p_fit[pj] = f_now
            for di in 1:dim
                pbest_state[di,pj] = particles[pj].psi[di]
            end

            for di in 1:ctrl_num
                for ni in 1:ctrl_length
                    pbest_ctrl[di,ni,pj] = particles[pj].control_coefficients[di][ni]
                end
            end
        end
    end

    for pj in 1:particle_num
        if p_fit[pj] > fit
            fit = p_fit[pj]
            for dj in 1:dim
                gbest_state[dj] = particles[pj].psi[dj]
                velocity_best_state[dj] = velocity_state[dj, pj]
            end

            for dj in 1:ctrl_num
                for nj in 1:ctrl_length
                    gbest_ctrl[dj, nj] = particles[pj].control_coefficients[dj][nj]
                    velocity_best_ctrl[dj, nj] = velocity_ctrl[dj, nj, pj]
                end
            end
        end
    end  

    for pk in 1:particle_num
        psi_pre = zeros(ComplexF64, dim)
        for dk in 1:dim
            psi_pre[dk] = particles[pk].psi[dk]
            velocity_state[dk, pk] = c0*velocity_state[dk, pk] + c1*rand()*(pbest_state[dk, pk] - particles[pk].psi[dk]) + 
                                     c2*rand()*(gbest_state[dk] - particles[pk].psi[dk])
            particles[pk].psi[dk] = particles[pk].psi[dk] + velocity_state[dk, pk]
        end
        particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)
        for dm in 1:dim
            velocity_state[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
        end

        control_coeff_pre = [zeros(ctrl_length) for i in 1:ctrl_num]
        for dk in 1:ctrl_num
            for ck in 1:ctrl_length
                control_coeff_pre[dk][ck] = particles[pk].control_coefficients[dk][ck]
                velocity_ctrl[dk, ck, pk] = c0*velocity_ctrl[dk, ck, pk] + c1*rand()*(pbest_ctrl[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                     + c2*rand()*(gbest_ctrl[dk, ck] - particles[pk].control_coefficients[dk][ck])
                particles[pk].control_coefficients[dk][ck] += velocity_ctrl[dk, ck, pk]
            end
        end

        for dm in 1:ctrl_num
            for cm in 1:ctrl_length
                particles[pk].control_coefficients[dm][cm] = (x-> x < particles[pk].ctrl_bound[1] ? particles[pk].ctrl_bound[1] : x > particles[pk].ctrl_bound[2] ? particles[pk].ctrl_bound[2] : x)(particles[pk].control_coefficients[dm][cm])
                velocity_ctrl[dm, cm, pk] = particles[pk].control_coefficients[dm][cm] - control_coeff_pre[dm][cm]
            end
        end
    end
    return p_fit, fit, pbest_state, pbest_ctrl, gbest_state, gbest_ctrl, velocity_best_state, velocity_best_ctrl, velocity_state, velocity_ctrl
end


################ state and measurement optimization ###############
function PSO_Compopt_SMopt(pso::Compopt_SMopt{T}, max_episode, particle_num, psi0, measurement0, c0, c1, c2, seed, save_file) where {T<: Complex}
    sym = Symbol("CFIM_SMopt")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    M = [zeros(ComplexF64, size(pso.psi)[1], size(pso.psi)[1])]
    return info_PSO_SMopt(M, pso, max_episode, particle_num, psi0, measurement0, c0, c1, c2, seed, save_file, sym, str1, str2, str3)
end

function info_PSO_SMopt(M, pso, max_episode, particle_num, psi0, measurement0, c0, c1, c2, seed, save_file, sym, str1, str2, str3) where {T<:Complex}
    println("comprehensive optimization")
    Random.seed!(seed)
    dim = length(pso.psi)
    M_num = length(pso.C)
    particles = repeat(pso, particle_num)

    velocity_state = 0.1.*rand(ComplexF64, dim, particle_num)
    pbest_state = zeros(ComplexF64, dim, particle_num)
    gbest_state = zeros(ComplexF64, dim)
    velocity_best_state = zeros(ComplexF64, dim)

    velocity_meas = 0.1*rand(ComplexF64, M_num, dim, particle_num)
    pbest_meas = zeros(ComplexF64, M_num, dim, particle_num)
    gbest_meas = zeros(ComplexF64, M_num, dim)
    velocity_best_meas = zeros(ComplexF64, M_num, dim)

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
    if length(pso.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: Particle Swarm Optimization (PSO)")
        println("initial $str2 is $(1.0/f_ini)")
        f_list = [1.0/f_ini]
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas = train_PSO_SMopt(M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, 
                pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas, sym)
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
            p_fit, fit, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas = train_PSO_SMopt(M, 
            particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, 
            pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas, sym)         
            M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
            append!(f_list, fit)
            SaveFile_SM(f_list, gbest_state, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str2 is $(fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas = train_PSO_SMopt(M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num,
                pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas, sym)               
                if ei%max_episode[2] == 0
                    pso.psi = [gbest_state[i] for i in 1:dim]
                    pso.C = [gbest_meas[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                print("current $str2 is $(fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas = train_PSO_SMopt(M, 
            particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num,
            pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas, sym)         
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
                p_fit, fit, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas = train_PSO_SMopt(M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num,
                pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas, sym)               
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
            p_fit, fit, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas = train_PSO_SMopt(M, 
            particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num,
            pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas, sym)           
            M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_SM(f_list, gbest_state, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is $(1.0/fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas = train_PSO_SMopt(M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num,
                pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas, sym)               
                if ei%max_episode[2] == 0
                    pso.psi = [gbest_state[i] for i in 1:dim]
                    pso.C = [gbest_meas[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, 1.0/fit)
                print("current value of $str3 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas = train_PSO_SMopt(M, 
            particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num,
            pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas, sym)            
            M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_SM(f_list, gbest_state, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is $(1.0/fit)")
        end
    end
end

function train_PSO_SMopt(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas, sym)
    for pj in 1:particle_num
        M = [particles[pj].C[i]*(particles[pj].C[i])' for i in 1:M_num]
        f_tp = obj_func(Val{sym}(), particles[pj], particles[pj].psi, M)
        f_now = 1.0/f_tp
        if f_now > p_fit[pj]
            p_fit[pj] = f_now
            for di in 1:dim
                pbest_state[di,pj] = particles[pj].psi[di]
            end

            for di in 1:M_num
                for ni in 1:dim
                    pbest_meas[di,ni,pj] = particles[pj].C[di][ni]
                end
            end
        end
    end

    for pj in 1:particle_num
        if p_fit[pj] > fit
            fit = p_fit[pj]
            for dj in 1:dim
                gbest_state[dj] = particles[pj].psi[dj]
                velocity_best_state[dj] = velocity_state[dj, pj]
            end

            for dj in 1:M_num
                for nj in 1:dim
                    gbest_meas[dj, nj] = particles[pj].C[dj][nj]
                    velocity_best_meas[dj, nj] = velocity_meas[dj, nj, pj]
                end
            end
        end
    end  

    for pk in 1:particle_num
        psi_pre = zeros(ComplexF64, dim)
        for dk in 1:dim
            psi_pre[dk] = particles[pk].psi[dk]
            velocity_state[dk, pk] = c0*velocity_state[dk, pk] + c1*rand()*(pbest_state[dk, pk] - particles[pk].psi[dk]) + 
                                     c2*rand()*(gbest_state[dk] - particles[pk].psi[dk])
            particles[pk].psi[dk] = particles[pk].psi[dk] + velocity_state[dk, pk]
        end
        particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)
        for dm in 1:dim
            velocity_state[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
        end

        meas_pre = [zeros(ComplexF64, dim) for i in 1:M_num]
        for dk in 1:M_num
            for ck in 1:dim
                meas_pre[dk][ck] = particles[pk].C[dk][ck]
    
                velocity_meas[dk, ck, pk] = c0*velocity_meas[dk, ck, pk] + c1*rand()*(pbest_meas[dk, ck, pk] - particles[pk].C[dk][ck]) 
                                           + c2*rand()*(gbest_meas[dk, ck] - particles[pk].C[dk][ck])
                particles[pk].C[dk][ck] += velocity_meas[dk, ck, pk]
            end
        end
        particles[pk].C = gramschmidt(particles[pk].C)

        for dm in 1:M_num
            for cm in 1:dim
                velocity_meas[dm, cm, pk] = particles[pk].C[dm][cm] - meas_pre[dm][cm]
            end
        end
    end
    return p_fit, fit, pbest_state, pbest_meas, gbest_state, gbest_meas, velocity_best_state, velocity_best_meas, velocity_state, velocity_meas
end


################ control and measurement optimization ###############
function PSO_Compopt_CMopt(rho0, pso::Compopt_CMopt{T}, max_episode, particle_num, psi0, ctrl0, measurement0, c0, c1, c2, seed, save_file) where {T<: Complex}
    sym = Symbol("CFIM_CMopt")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    M = [zeros(ComplexF64, size(rho0)[1], size(rho0)[1])]
    return info_PSO_CMopt(rho0, M, pso, max_episode, particle_num, psi0, ctrl0, measurement0, c0, c1, c2, seed, save_file, sym, str1, str2, str3)
end

function info_PSO_CMopt(rho0, M, pso, max_episode, particle_num, psi0, ctrl0, measurement0, c0, c1, c2, seed, save_file, sym, str1, str2, str3) where {T<:Complex}
    println("comprehensive optimization")
    Random.seed!(seed)
    dim = size(rho0)[1]
    M_num = length(pso.C)
    ctrl_length = length(pso.control_coefficients[1])
    ctrl_num = length(pso.control_Hamiltonian)
    particles = repeat(pso, particle_num)

    velocity_meas = 0.1*rand(ComplexF64, M_num, dim, particle_num)
    pbest_meas = zeros(ComplexF64, M_num, dim, particle_num)
    gbest_meas = zeros(ComplexF64, M_num, dim)
    velocity_best_meas = zeros(ComplexF64, M_num, dim)

    if pso.ctrl_bound[1] == -Inf || pso.ctrl_bound[2] == Inf
        velocity_ctrl = 0.1*(2.0*rand(ctrl_num, ctrl_length, particle_num)-ones(ctrl_num, ctrl_length, particle_num))
    else
        a = pso.ctrl_bound[1]
        b = pso.ctrl_bound[2]
        velocity_ctrl = 0.1*((b-a)*rand(ctrl_num, ctrl_length, particle_num)+a*ones(ctrl_num, ctrl_length, particle_num))
    end
    pbest_ctrl = zeros(ctrl_num, ctrl_length, particle_num)
    gbest_ctrl = zeros(ctrl_num, ctrl_length)
    velocity_best_ctrl = zeros(ctrl_num,ctrl_length)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    # initialize
    if length(ctrl0) > particle_num
        ctrl0 = [ctrl0[i] for i in 1:particle_num]
    end
    if length(measurement0) > particle_num
        measurement0 = [measurement0[i] for i in 1:particle_num]
    end 

    for pj in 1:length(ctrl0)
        particles[pj].control_coefficients = [[ctrl0[pj][i,j] for j in 1:ctrl_length] for i in 1:ctrl_num]
    end
    if pso.ctrl_bound[1] == -Inf || pso.ctrl_bound[2] == Inf
        for pj in (length(ctrl0)+1):particle_num
            particles[pj].control_coefficients = [[2*rand()-1.0 for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
    else
        a = pso.ctrl_bound[1]
        b = pso.ctrl_bound[2]
        for pj in (length(ctrl0)+1):particle_num
            particles[pj].control_coefficients = [[(b-a)*rand()+a for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
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
        p_fit[pj] = 1.0/obj_func(Val{sym}(), pso, M, rho0, particles[pj].control_coefficients)
    end

    f_ini= p_fit[1]

    fit = 0.0
    if length(pso.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: Particle Swarm Optimization (PSO)")
        println("initial $str2 is $(1.0/f_ini)")
        f_list = [1.0/f_ini]
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas = train_PSO_CMopt(rho0, M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length,
                pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas, sym)
                if ei%max_episode[2] == 0
                    pso.control_coefficients = [gbest_ctrl[k, :] for k in 1:ctrl_num]
                    pso.C = [gbest_meas[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
                append!(f_list, fit)
                SaveFile_CM(f_list, [gbest_ctrl[k, :] for k in 1:ctrl_num], M)
                print("current $str2 is $(fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas = train_PSO_CMopt(rho0, M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length,
                pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas, sym)
            M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
            append!(f_list, fit)
            SaveFile_CM(f_list, [gbest_ctrl[k, :] for k in 1:ctrl_num], M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str2 is $(fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas = train_PSO_CMopt(rho0, M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length,
                pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas, sym)
                if ei%max_episode[2] == 0
                    pso.control_coefficients = [gbest_ctrl[k, :] for k in 1:ctrl_num]
                    pso.C = [gbest_meas[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                print("current $str2 is $(fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas = train_PSO_CMopt(rho0, M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length,
                pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas, sym)
            M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
            append!(f_list, fit)
            SaveFile_CM(f_list, [gbest_ctrl[k, :] for k in 1:ctrl_num], M)
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
                p_fit, fit, pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas = train_PSO_CMopt(rho0, M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length,
                pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas, sym)
                if ei%max_episode[2] == 0
                    pso.control_coefficients = [gbest_ctrl[k, :] for k in 1:ctrl_num]
                    pso.C = [gbest_meas[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
                append!(f_list, 1.0/fit)
                SaveFile_CM(f_list, [gbest_ctrl[k, :] for k in 1:ctrl_num], M)
                print("current value of $str3 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas = train_PSO_CMopt(rho0, M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length,
                pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas, sym)
            M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_CM(f_list, [gbest_ctrl[k, :] for k in 1:ctrl_num], M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is $(1.0/fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas = train_PSO_CMopt(rho0, M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length,
                pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas, sym)
                if ei%max_episode[2] == 0
                    pso.control_coefficients = [gbest_ctrl[k, :] for k in 1:ctrl_num]
                    pso.C = [gbest_meas[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, 1.0/fit)
                print("current value of $str3 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas = train_PSO_CMopt(rho0, M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length,
                pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas, sym)
            M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_CM(f_list, [gbest_ctrl[k, :] for k in 1:ctrl_num], M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is $(1.0/fit)")
        end
    end
end

function train_PSO_CMopt(rho0, M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length, pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas, sym)
    for pj in 1:particle_num
        M = [particles[pj].C[i]*(particles[pj].C[i])' for i in 1:M_num]
        f_tp = obj_func(Val{sym}(), particles[pj], M, rho0, particles[pj].control_coefficients)
        f_now = 1.0/f_tp
        if f_now > p_fit[pj]
            p_fit[pj] = f_now
            for di in 1:ctrl_num
                for ni in 1:ctrl_length
                    pbest_ctrl[di,ni,pj] = particles[pj].control_coefficients[di][ni]
                end
            end
            for di in 1:M_num
                for ni in 1:dim
                    pbest_meas[di,ni,pj] = particles[pj].C[di][ni]
                end
            end
        end
    end

    for pj in 1:particle_num
        if p_fit[pj] > fit
            fit = p_fit[pj]
            for dj in 1:ctrl_num
                for nj in 1:ctrl_length
                    gbest_ctrl[dj, nj] = particles[pj].control_coefficients[dj][nj]
                    velocity_best_ctrl[dj, nj] = velocity_ctrl[dj, nj, pj]
                end
            end
            for dj in 1:M_num
                for nj in 1:dim
                    gbest_meas[dj, nj] = particles[pj].C[dj][nj]
                    velocity_best_meas[dj, nj] = velocity_meas[dj, nj, pj]
                end
            end
        end
    end  

    for pk in 1:particle_num
        control_coeff_pre = [zeros(ctrl_length) for i in 1:ctrl_num]
        for dk in 1:ctrl_num
            for ck in 1:ctrl_length
                control_coeff_pre[dk][ck] = particles[pk].control_coefficients[dk][ck]
                velocity_ctrl[dk, ck, pk] = c0*velocity_ctrl[dk, ck, pk] + c1*rand()*(pbest_ctrl[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                     + c2*rand()*(gbest_ctrl[dk, ck] - particles[pk].control_coefficients[dk][ck])
                particles[pk].control_coefficients[dk][ck] += velocity_ctrl[dk, ck, pk]
            end
        end

        for dm in 1:ctrl_num
            for cm in 1:ctrl_length
                particles[pk].control_coefficients[dm][cm] = (x-> x < particles[pk].ctrl_bound[1] ? particles[pk].ctrl_bound[1] : x > particles[pk].ctrl_bound[2] ? particles[pk].ctrl_bound[2] : x)(particles[pk].control_coefficients[dm][cm])
                velocity_ctrl[dm, cm, pk] = particles[pk].control_coefficients[dm][cm] - control_coeff_pre[dm][cm]
            end
        end

        meas_pre = [zeros(ComplexF64, dim) for i in 1:M_num]
        for dk in 1:M_num
            for ck in 1:dim
                meas_pre[dk][ck] = particles[pk].C[dk][ck]
    
                velocity_meas[dk, ck, pk] = c0*velocity_meas[dk, ck, pk] + c1*rand()*(pbest_meas[dk, ck, pk] - particles[pk].C[dk][ck]) 
                                           + c2*rand()*(gbest_meas[dk, ck] - particles[pk].C[dk][ck])
                particles[pk].C[dk][ck] += velocity_meas[dk, ck, pk]
            end
        end
        particles[pk].C = gramschmidt(particles[pk].C)

        for dm in 1:M_num
            for cm in 1:dim
                velocity_meas[dm, cm, pk] = particles[pk].C[dm][cm] - meas_pre[dm][cm]
            end
        end
    end
    return p_fit, fit, pbest_ctrl, pbest_meas, gbest_ctrl, gbest_meas, velocity_best_ctrl, velocity_best_meas, velocity_ctrl, velocity_meas
end


################ state, control and measurement optimization ###############
function PSO_Compopt_SCMopt(pso::Compopt_SCMopt{T}, max_episode, particle_num, psi0, ctrl0, measurement0, c0, c1, c2, seed, save_file) where {T<: Complex}
    sym = Symbol("CFIM_SCopt")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    M = [zeros(ComplexF64, size(pso.psi)[1], size(pso.psi)[1])]
    return info_PSO_SCMopt(M, pso, max_episode, particle_num, psi0, ctrl0, measurement0, c0, c1, c2, seed, save_file, sym, str1, str2, str3)
end

function info_PSO_SCMopt(M, pso, max_episode, particle_num, psi0, ctrl0, measurement0, c0, c1, c2, seed, save_file, sym, str1, str2, str3) where {T<:Complex}
    println("comprehensive optimization")
    Random.seed!(seed)
    dim = length(pso.psi)
    M_num = length(pso.C)
    ctrl_length = length(pso.control_coefficients[1])
    ctrl_num = length(pso.control_Hamiltonian)
    particles = repeat(pso, particle_num)

    velocity_state = 0.1.*rand(ComplexF64, dim, particle_num)
    pbest_state = zeros(ComplexF64, dim, particle_num)
    gbest_state = zeros(ComplexF64, dim)
    velocity_best_state = zeros(ComplexF64, dim)

    velocity_meas = 0.1*rand(ComplexF64, M_num, dim, particle_num)
    pbest_meas = zeros(ComplexF64, M_num, dim, particle_num)
    gbest_meas = zeros(ComplexF64, M_num, dim)
    velocity_best_meas = zeros(ComplexF64, M_num, dim)

    if pso.ctrl_bound[1] == -Inf || pso.ctrl_bound[2] == Inf
        velocity_ctrl = 0.1*(2.0*rand(ctrl_num, ctrl_length, particle_num)-ones(ctrl_num, ctrl_length, particle_num))
    else
        a = pso.ctrl_bound[1]
        b = pso.ctrl_bound[2]
        velocity_ctrl = 0.1*((b-a)*rand(ctrl_num, ctrl_length, particle_num)+a*ones(ctrl_num, ctrl_length, particle_num))
    end
    pbest_ctrl = zeros(ctrl_num, ctrl_length, particle_num)
    gbest_ctrl = zeros(ctrl_num, ctrl_length)
    velocity_best_ctrl = zeros(ctrl_num,ctrl_length)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    # initialize
    if length(psi0) > particle_num
        psi0 = [psi0[i] for i in 1:particle_num]
    end
    if length(ctrl0) > particle_num
        ctrl0 = [ctrl0[i] for i in 1:particle_num]
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

    for pj in 1:length(ctrl0)
        particles[pj].control_coefficients = [[ctrl0[pj][i,j] for j in 1:ctrl_length] for i in 1:ctrl_num]
    end
    if pso.ctrl_bound[1] == -Inf || pso.ctrl_bound[2] == Inf
        for pj in (length(ctrl0)+1):particle_num
            particles[pj].control_coefficients = [[2*rand()-1.0 for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
    else
        a = pso.ctrl_bound[1]
        b = pso.ctrl_bound[2]
        for pj in (length(ctrl0)+1):particle_num
            particles[pj].control_coefficients = [[(b-a)*rand()+a for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
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
        p_fit[pj] = 1.0/obj_func(Val{sym}(), pso, M, particles[pj].psi, particles[pj].control_coefficients)
    end

    f_ini= p_fit[1]

    fit = 0.0
    if length(pso.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: Particle Swarm Optimization (PSO)")
        println("initial $str2 is $(1.0/f_ini)")
        f_list = [1.0/f_ini]
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas = train_PSO_SCMopt(M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length,
                pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas, sym)
                if ei%max_episode[2] == 0
                    pso.psi = [gbest_state[i] for i in 1:dim]
                    pso.control_coefficients = [gbest_ctrl[k, :] for k in 1:ctrl_num]
                    pso.C = [gbest_meas[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
                append!(f_list, fit)
                SaveFile_SCM(f_list, gbest_state, [gbest_ctrl[k, :] for k in 1:ctrl_num], M)
                print("current $str2 is $(fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas = train_PSO_SCMopt(M, 
            particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length,
            pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas, sym)
            M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
            append!(f_list, fit)
            SaveFile_SCM(f_list, gbest_state, [gbest_ctrl[k, :] for k in 1:ctrl_num], M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str2 is $(fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas = train_PSO_SCMopt(M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length,
                pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas, sym)
                if ei%max_episode[2] == 0
                    pso.psi = [gbest_state[i] for i in 1:dim]
                    pso.control_coefficients = [gbest_ctrl[k, :] for k in 1:ctrl_num]
                    pso.C = [gbest_meas[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                print("current $str2 is $(fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas = train_PSO_SCMopt(M, 
            particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length,
            pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas, sym)
            M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
            append!(f_list, fit)
            SaveFile_SCM(f_list, gbest_state, [gbest_ctrl[k, :] for k in 1:ctrl_num], M)
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
                p_fit, fit, pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas = train_PSO_SCMopt(M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length,
                pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas, sym)
                if ei%max_episode[2] == 0
                    pso.psi = [gbest_state[i] for i in 1:dim]
                    pso.control_coefficients = [gbest_ctrl[k, :] for k in 1:ctrl_num]
                    pso.C = [gbest_meas[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
                append!(f_list, 1.0/fit)
                SaveFile_SCM(f_list, gbest_state, [gbest_ctrl[k, :] for k in 1:ctrl_num], M)
                print("current value of $str3 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas = train_PSO_SCMopt(M, 
            particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length,
            pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas, sym)
            M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_SCM(f_list, gbest_state, [gbest_ctrl[k, :] for k in 1:ctrl_num], M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is $(1.0/fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas = train_PSO_SCMopt(M, 
                particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length,
                pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas, sym)
                if ei%max_episode[2] == 0
                    pso.psi = [gbest_state[i] for i in 1:dim]
                    pso.control_coefficients = [gbest_ctrl[k, :] for k in 1:ctrl_num]
                    pso.C = [gbest_meas[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, 1.0/fit)
                print("current value of $str3 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas = train_PSO_SCMopt(M, 
            particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length,
            pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas, sym)
            M = [gbest_meas[i]*gbest_meas[i]' for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_SCM(f_list, gbest_state, [gbest_ctrl[k, :] for k in 1:ctrl_num], M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is $(1.0/fit)")
        end
    end
end

function train_PSO_SCMopt(M, particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, dim, M_num, ctrl_num, ctrl_length, pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas, sym)
    for pj in 1:particle_num
        M = [particles[pj].C[i]*(particles[pj].C[i])' for i in 1:M_num]
        f_tp = obj_func(Val{sym}(), particles[pj], M, particles[pj].psi, particles[pj].control_coefficients)
        f_now = 1.0/f_tp
        if f_now > p_fit[pj]
            p_fit[pj] = f_now
            for di in 1:dim
                pbest_state[di,pj] = particles[pj].psi[di]
            end
            for di in 1:ctrl_num
                for ni in 1:ctrl_length
                    pbest_ctrl[di,ni,pj] = particles[pj].control_coefficients[di][ni]
                end
            end
            for di in 1:M_num
                for ni in 1:dim
                    pbest_meas[di,ni,pj] = particles[pj].C[di][ni]
                end
            end
        end
    end

    for pj in 1:particle_num
        if p_fit[pj] > fit
            fit = p_fit[pj]
            for dj in 1:dim
                gbest_state[dj] = particles[pj].psi[dj]
                velocity_best_state[dj] = velocity_state[dj, pj]
            end
            for dj in 1:ctrl_num
                for nj in 1:ctrl_length
                    gbest_ctrl[dj, nj] = particles[pj].control_coefficients[dj][nj]
                    velocity_best_ctrl[dj, nj] = velocity_ctrl[dj, nj, pj]
                end
            end
            for dj in 1:M_num
                for nj in 1:dim
                    gbest_meas[dj, nj] = particles[pj].C[dj][nj]
                    velocity_best_meas[dj, nj] = velocity_meas[dj, nj, pj]
                end
            end
        end
    end  

    for pk in 1:particle_num
        psi_pre = zeros(ComplexF64, dim)
        for dk in 1:dim
            psi_pre[dk] = particles[pk].psi[dk]
            velocity_state[dk, pk] = c0*velocity_state[dk, pk] + c1*rand()*(pbest_state[dk, pk] - particles[pk].psi[dk]) + 
                                     c2*rand()*(gbest_state[dk] - particles[pk].psi[dk])
            particles[pk].psi[dk] = particles[pk].psi[dk] + velocity_state[dk, pk]
        end
        particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)
        for dm in 1:dim
            velocity_state[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
        end

        control_coeff_pre = [zeros(ctrl_length) for i in 1:ctrl_num]
        for dk in 1:ctrl_num
            for ck in 1:ctrl_length
                control_coeff_pre[dk][ck] = particles[pk].control_coefficients[dk][ck]
                velocity_ctrl[dk, ck, pk] = c0*velocity_ctrl[dk, ck, pk] + c1*rand()*(pbest_ctrl[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                     + c2*rand()*(gbest_ctrl[dk, ck] - particles[pk].control_coefficients[dk][ck])
                particles[pk].control_coefficients[dk][ck] += velocity_ctrl[dk, ck, pk]
            end
        end

        for dm in 1:ctrl_num
            for cm in 1:ctrl_length
                particles[pk].control_coefficients[dm][cm] = (x-> x < particles[pk].ctrl_bound[1] ? particles[pk].ctrl_bound[1] : x > particles[pk].ctrl_bound[2] ? particles[pk].ctrl_bound[2] : x)(particles[pk].control_coefficients[dm][cm])
                velocity_ctrl[dm, cm, pk] = particles[pk].control_coefficients[dm][cm] - control_coeff_pre[dm][cm]
            end
        end

        meas_pre = [zeros(ComplexF64, dim) for i in 1:M_num]
        for dk in 1:M_num
            for ck in 1:dim
                meas_pre[dk][ck] = particles[pk].C[dk][ck]
    
                velocity_meas[dk, ck, pk] = c0*velocity_meas[dk, ck, pk] + c1*rand()*(pbest_meas[dk, ck, pk] - particles[pk].C[dk][ck]) 
                                           + c2*rand()*(gbest_meas[dk, ck] - particles[pk].C[dk][ck])
                particles[pk].C[dk][ck] += velocity_meas[dk, ck, pk]
            end
        end
        particles[pk].C = gramschmidt(particles[pk].C)

        for dm in 1:M_num
            for cm in 1:dim
                velocity_meas[dm, cm, pk] = particles[pk].C[dm][cm] - meas_pre[dm][cm]
            end
        end
    end
    return p_fit, fit, pbest_state, pbest_ctrl, pbest_meas, gbest_state, gbest_ctrl, gbest_meas, velocity_best_state, velocity_best_ctrl, velocity_best_meas, velocity_state, velocity_ctrl, velocity_meas
end
