
function CFIM_PSO_Mopt(pso::projection_Mopt_Kraus{T}, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file) where {T<:Complex}
    sym = Symbol("CFIM_noctrl_Kraus")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_PSO_projection_Kraus(pso, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file, sym, str1, str2)
end

function CFIM_PSO_Mopt(pso::LinearComb_Mopt_Kraus{T}, max_episode, particle_num, c0, c1, c2, seed, save_file) where {T<:Complex}
    sym = Symbol("CFIM_noctrl_Kraus")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_PSO_LinearComb_Kraus(pso, max_episode, particle_num, c0, c1, c2, seed, save_file, sym, str1, str2)
end


function CFIM_PSO_Mopt(pso::RotateBasis_Mopt_Kraus{T}, max_episode, particle_num, c0, c1, c2, seed, save_file) where {T<:Complex}
    sym = Symbol("CFIM_noctrl_Kraus")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_PSO_RotateBasis_Kraus(pso, max_episode, particle_num, c0, c1, c2, seed, save_file, sym, str1, str2)
end

function info_PSO_projection_Kraus(pso, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file, sym, str1, str2) where {T<:Complex}
    println("measurement optimization")
    Random.seed!(seed)
    dim = size(pso.ρ0)[1] 
    M_num = length(pso.C)
    particles = repeat(pso, particle_num)
    velocity = 0.1*rand(ComplexF64, M_num, dim, particle_num)
    pbest = zeros(ComplexF64, M_num, dim, particle_num)
    gbest = zeros(ComplexF64, M_num, dim)
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
        particles[pj].C = [[ini_particle[pj][i,j] for j in 1:dim] for i in 1:M_num]
    end
    for pj in (length(ini_particle)+1):particle_num
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
        p_fit[pj] = 1.0/obj_func(Val{sym}(), pso, M)
    end

    f_ini= p_fit[1]
    f_opt = obj_func(Val{:QFIM_noctrl_Kraus}(), pso, pso.C)
    f_opt= 1.0/f_opt

    if length(pso.dK) == 1
        f_list = [f_ini]

        println("single parameter scenario")
        println("search algorithm: Particle Swarm Optimization (PSO)")
        println("initial $str1 is $(f_ini)")
        println("QFI is $(f_opt)")
        
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, pbest, gbest, velocity = train_projection(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                               M_num, dim, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.C = [gbest[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                M = [gbest[i]*gbest[i]' for i in 1:M_num]
                append!(f_list, fit)
                SaveFile_meas(f_list, M)
                print("current $str1 is $fit ($ei episodes)    \r")
            end
            p_fit, fit, pbest, gbest, velocity = train_projection(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                           M_num, dim, pbest, gbest, velocity, sym)
            M = [gbest[i]*gbest[i]' for i in 1:M_num]
            append!(f_list, fit)
            SaveFile_meas(f_list, M)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final $str1 is $fit")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity = train_projection(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                               M_num, dim, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.C = [gbest[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, fit)
                print("current $str1 is $fit ($ei episodes)    \r")
                
            end
            p_fit, fit, pbest, gbest, velocity = train_projection(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                           M_num, dim, pbest, gbest, velocity, sym)
            M = [gbest[i]*gbest[i]' for i in 1:M_num]
            append!(f_list, fit)
            SaveFile_meas(f_list, M)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final $str1 is $fit")
        end
    else
        f_list = [1.0/f_ini]
        println("multiparameter scenario")
        println("search algorithm: Particle Swarm Optimization (PSO)")
        println("initial value of $str2 is $(1.0/f_ini)")
        println("tr(WF^{-1}) is $(1.0/f_opt)")

        if save_file == true
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, pbest, gbest, velocity = train_projection(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                               M_num, dim, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.C = [gbest[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                M = [gbest[i]*gbest[i]' for i in 1:M_num]
                append!(f_list, 1.0/fit)
                SaveFile_meas(f_list, M)
                print("current value of $str2 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, pbest, gbest, velocity = train_projection(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                           M_num, dim, pbest, gbest, velocity, sym)
            M = [gbest[i]*gbest[i]' for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_meas(f_list, M)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final value of $str2 is $(1.0/fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, pbest, gbest, velocity = train_projection(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                           M_num, dim, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    pso.C = [gbest[k, :] for k in 1:M_num]
                    particles = repeat(pso, particle_num)
                end
                append!(f_list, 1.0/fit)
                print("current value of $str2 is $(1.0/fit) ($ei episodes)    \r")
                
            end
            p_fit, fit, pbest, gbest, velocity = train_projection(particles, p_fit, fit, max_episode, c0, c1, c2, particle_num, 
                                                                           M_num, dim, pbest, gbest, velocity, sym)
            M = [gbest[i]*gbest[i]' for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_meas(f_list, M)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final value of $str2 is $(1.0/fit)")
        end
    end
end

function info_PSO_LinearComb_Kraus(pso, max_episode, particle_num, c0, c1, c2, seed, save_file, sym, str1, str2) where {T<:Complex}
    println("measurement optimization")
    Random.seed!(seed)
    dim = size(pso.ρ0)[1]
    POVM_basis = pso.povm_basis
    M_num = pso.M_num
    basis_num = length(POVM_basis)
    particles = repeat(pso, particle_num)
    velocity = 0.1*rand(Float64, M_num, basis_num, particle_num)
    pbest = zeros(Float64, M_num, basis_num, particle_num)
    gbest = zeros(Float64, M_num, basis_num)
    p_fit = zeros(particle_num)
    fit = 0.0

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    # initialize 
    B_all = [[zeros(basis_num) for i in 1:M_num] for j in 1:particle_num]
    for pj in 1:particle_num
        B_all[pj] = [rand(basis_num) for i in 1:M_num]
        B_all[pj] = bound_LC_coeff(B_all[pj])
    end

    p_fit = [0.0 for i in 1:particle_num] 
    for pj in 1:particle_num
        M = [sum([B_all[pj][i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
        p_fit[pj] = 1.0/obj_func(Val{sym}(), pso, M)
    end

    f_ini = p_fit[1]
    f_opt = obj_func(Val{:QFIM_noctrl_Kraus}(), pso, POVM_basis)
    f_opt = 1.0/f_opt

    f_povm = obj_func(Val{sym}(), pso, POVM_basis)
    f_povm = 1.0/f_povm

    if length(pso.dK) == 1
        f_list = [f_ini]

        println("single parameter scenario")
        println("search algorithm: Particle Swarm Optimization (PSO)")
        println("initial $str1 is $(f_ini)")
        println("CFI under the given POVMs is $(f_povm)")
        println("QFI is $(f_opt)")
        
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, B_all, pbest, gbest, velocity = train_LinearComb(particles, p_fit, fit, B_all, POVM_basis, max_episode, c0, c1, c2, particle_num, 
                                                                               M_num, basis_num, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    particles = repeat(pso, particle_num)
                    for i in 1:particle_num
                        B_all[i] = [gbest[k, :] for k in 1:M_num]
                    end
                end
                M = [sum([gbest[i,j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
                append!(f_list, fit)
                SaveFile_meas(f_list, M)
                print("current $str1 is $fit ($ei episodes)    \r")
            end
            p_fit, fit, B_all, pbest, gbest, velocity = train_LinearComb(particles, p_fit, fit, B_all, POVM_basis, 
                                    max_episode, c0, c1, c2, particle_num, M_num, basis_num, pbest, gbest, velocity, sym)
            M = [sum([gbest[i,j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
            append!(f_list, fit)
            SaveFile_meas(f_list, M)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final $str1 is $fit")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, B_all, pbest, gbest, velocity = train_LinearComb(particles, p_fit, fit, B_all, POVM_basis, 
                                    max_episode, c0, c1, c2, particle_num, M_num, basis_num, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    particles = repeat(pso, particle_num)
                    for i in 1:particle_num
                        B_all[i] = [gbest[k, :] for k in 1:M_num]
                    end
                end
                append!(f_list, fit)
                print("current $str1 is $fit ($ei episodes)    \r")
                
            end
            p_fit, fit, B_all, pbest, gbest, velocity = train_LinearComb(particles, p_fit, fit, B_all, POVM_basis, 
                                max_episode, c0, c1, c2, particle_num, M_num, basis_num, pbest, gbest, velocity, sym)
            M = [sum([gbest[i,j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
            append!(f_list, fit)
            SaveFile_meas(f_list, M)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final $str1 is $fit")
        end
    else
        f_list = [1.0/f_ini]
        println("multiparameter scenario")
        println("search algorithm: Particle Swarm Optimization (PSO)")
        println("initial value of $str2 is $(1.0/f_ini)")
        println("tr(WI^{-1}) under the given POVMs is $(1.0/f_povm)")
        println("tr(WF^{-1}) is $(1.0/f_opt)")

        if save_file == true
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, B_all, pbest, gbest, velocity = train_LinearComb(particles, p_fit, fit, B_all, POVM_basis, 
                                max_episode, c0, c1, c2, particle_num, M_num, basis_num, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    particles = repeat(pso, particle_num)
                    for i in 1:particle_num
                        B_all[i] = [gbest[k, :] for k in 1:M_num]
                    end
                end
                M = [sum([gbest[i,j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
                append!(f_list, 1.0/fit)
                SaveFile_meas(f_list, M)
                print("current value of $str2 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, B_all, pbest, gbest, velocity = train_LinearComb(particles, p_fit, fit, B_all, POVM_basis, 
                                max_episode, c0, c1, c2, particle_num, M_num, basis_num, pbest, gbest, velocity, sym)
            M = [sum([gbest[i,j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_meas(f_list, M)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final value of $str2 is $(1.0/fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, B_all, pbest, gbest, velocity = train_LinearComb(particles, p_fit, fit, B_all, POVM_basis, 
                                    max_episode, c0, c1, c2, particle_num, M_num, basis_num, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    particles = repeat(pso, particle_num)
                    for i in 1:particle_num
                        B_all[i] = [gbest[k, :] for k in 1:M_num]
                    end
                end
                append!(f_list, 1.0/fit)
                print("current value of $str2 is $(1.0/fit) ($ei episodes)    \r")
                
            end
            p_fit, fit, B_all, pbest, gbest, velocity = train_LinearComb(particles, p_fit, fit, B_all, POVM_basis, 
                                max_episode, c0, c1, c2, particle_num, M_num, basis_num, pbest, gbest, velocity, sym)
            M = [sum([gbest[i,j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_meas(f_list, M)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final value of $str2 is $(1.0/fit)")
        end
    end
end

function info_PSO_RotateBasis_Kraus(pso, max_episode, particle_num, c0, c1, c2, seed, save_file, sym, str1, str2) where {T<:Complex}
    println("measurement optimization")
    Random.seed!(seed)
    dim = size(pso.ρ0)[1]
    suN = suN_generator(dim)
    Lambda = [Matrix{ComplexF64}(I,dim,dim)]
    append!(Lambda, [suN[i] for i in 1:length(suN)])

    POVM_basis = pso.povm_basis
    M_num = length(POVM_basis)
    particles = repeat(pso, particle_num)
    velocity = 0.1*rand(Float64, dim^2, particle_num)
    pbest = zeros(Float64, dim^2, particle_num)
    gbest = zeros(Float64, dim^2)
    s_all = [zeros(dim*dim) for i in 1:particle_num]
    p_fit = zeros(particle_num)
    fit = 0.0

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    # initialize 
    p_fit = [0.0 for i in 1:particle_num] 
    for pj in 1:particle_num
        # generate a rotation matrix randomly
        s_all[pj] = rand(dim*dim)
        U = rotation_matrix(s_all[pj], Lambda)
        M = [U*POVM_basis[i]*U' for i in 1:M_num]
        p_fit[pj] = 1.0/obj_func(Val{sym}(), pso, M)
    end

    f_ini = p_fit[1]
    f_opt = obj_func(Val{:QFIM_noctrl_Kraus}(), pso, POVM_basis)
    f_opt = 1.0/f_opt

    f_povm = obj_func(Val{sym}(), pso, POVM_basis)
    f_povm = 1.0/f_povm

    if length(pso.dK) == 1
        f_list = [f_ini]

        println("single parameter scenario")
        println("search algorithm: Particle Swarm Optimization (PSO)")
        println("initial $str1 is $(f_ini)")
        println("CFI under the given POVMs is $(f_povm)")
        println("QFI is $(f_opt)")
        
        if save_file == true
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, s_all, pbest, gbest, velocity = train_RotateBasis(particles, s_all, Lambda, p_fit, fit, POVM_basis, 
                                            max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    particles = repeat(pso, particle_num)
                    for pj in 1:particle_num
                        s_all[pj] = [gbest[i] for i in 1:dim^2]
                    end
                end
                U = rotation_matrix(gbest, Lambda)
                M = [U*POVM_basis[i]*U' for i in 1:M_num]
                append!(f_list, fit)
                SaveFile_meas(f_list, M)
                print("current $str1 is $fit ($ei episodes)    \r")
            end
            p_fit, fit, s_all, pbest, gbest, velocity = train_RotateBasis(particles, s_all, Lambda, p_fit, fit, POVM_basis, 
                                            max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity, sym)
            U = rotation_matrix(gbest, Lambda)
            M = [U*POVM_basis[i]*U' for i in 1:M_num]
            append!(f_list, fit)
            SaveFile_meas(f_list, M)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final $str1 is $fit")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, s_all, pbest, gbest, velocity = train_RotateBasis(particles, s_all, Lambda, p_fit, fit, POVM_basis, 
                                            max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    particles = repeat(pso, particle_num)
                    for pj in 1:particle_num
                        s_all[pj] = [gbest[i] for i in 1:dim^2]
                    end
                end
                append!(f_list, fit)
                print("current $str1 is $fit ($ei episodes)    \r")
                
            end
            p_fit, fit, s_all, pbest, gbest, velocity = train_RotateBasis(particles, s_all, Lambda, p_fit, fit, POVM_basis, 
                                            max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity, sym)
            U = rotation_matrix(gbest, Lambda)
            M = [U*POVM_basis[i]*U' for i in 1:M_num]
            append!(f_list, fit)
            SaveFile_meas(f_list, M)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final $str1 is $fit")
        end
    else
        f_list = [1.0/f_ini]
        println("multiparameter scenario")
        println("search algorithm: Particle Swarm Optimization (PSO)")
        println("initial value of $str2 is $(1.0/f_ini)")
        println("tr(WI^{-1}) under the given POVMs is $(1.0/f_povm)")
        println("tr(WF^{-1}) is $(1.0/f_opt)")

        if save_file == true
            for ei in 1:(max_episode[1]-1)
                #### train ####
                p_fit, fit, s_all, pbest, gbest, velocity = train_RotateBasis(particles, s_all, Lambda, p_fit, fit, POVM_basis, 
                                                 max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    particles = repeat(pso, particle_num)
                    for pj in 1:particle_num
                        s_all[pj] = [gbest[i] for i in 1:dim^2]
                    end
                end
                U = rotation_matrix(gbest, Lambda)
                M = [U*POVM_basis[i]*U' for i in 1:M_num]
                append!(f_list, 1.0/fit)
                SaveFile_meas(f_list, M)
                print("current value of $str2 is $(1.0/fit) ($ei episodes)    \r")
            end
            p_fit, fit, s_all, pbest, gbest, velocity = train_RotateBasis(particles, s_all, Lambda, p_fit, fit, POVM_basis, 
                                              max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity, sym)
            U = rotation_matrix(gbest, Lambda)
            M = [U*POVM_basis[i]*U' for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_meas(f_list, M)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final value of $str2 is $(1.0/fit)")
        else
            for ei in 1:(max_episode[1]-1)
                p_fit, fit, s_all, pbest, gbest, velocity = train_RotateBasis(particles, s_all, Lambda, p_fit, fit, POVM_basis, 
                                                 max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity, sym)
                if ei%max_episode[2] == 0
                    particles = repeat(pso, particle_num)
                    for pj in 1:particle_num
                        s_all[pj] = [gbest[i] for i in 1:dim^2]
                    end
                end
                append!(f_list, 1.0/fit)
                print("current value of $str2 is $(1.0/fit) ($ei episodes)    \r")
                
            end
            p_fit, fit, s_all, pbest, gbest, velocity = train_RotateBasis(particles, s_all, Lambda, p_fit, fit, POVM_basis, 
                                             max_episode, c0, c1, c2, particle_num, M_num, dim, pbest, gbest, velocity, sym)
            U = rotation_matrix(gbest, Lambda)
            M = [U*POVM_basis[i]*U' for i in 1:M_num]
            append!(f_list, 1.0/fit)
            SaveFile_meas(f_list, M)
            print("\e[2K")    
            println("Iteration over, data saved.")
            println("Final value of $str2 is $(1.0/fit)")
        end
    end
end