############# time-independent Hamiltonian (noiseless) ################
function DE_QFIM(DE::TimeIndepend_noiseless{T}, popsize, ini_population, c, cr, seed, max_episode, save_file) where {T<: Complex}
    println("state optimization")
    Random.seed!(seed)
    dim = length(DE.psi)
    p_num = popsize
    populations = repeat(DE, p_num)
    # initialize
    if length(ini_population) > popsize
        ini_population = [ini_population[i] for i in 1:popsize]
    end 
    for pj in 1:length(ini_population)
        populations[pj].psi = [ini_population[pj][i] for i in 1:dim]
    end
    for pj in (length(ini_population)+1):p_num
        r_ini = 2*rand(dim)-ones(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        populations[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    p_fit = [0.0 for i in 1:p_num] 
    for pj in 1:p_num
        F_tp = QFIM_TimeIndepend(DE.freeHamiltonian, DE.Hamiltonian_derivative, populations[pj].psi, DE.tspan, DE.accuracy)
        p_fit[pj] = 1.0/real(tr(DE.W*pinv(F_tp)))
    end

    F = QFIM_TimeIndepend(DE.freeHamiltonian, DE.Hamiltonian_derivative, DE.psi, DE.tspan, DE.accuracy)
    f_ini= real(tr(DE.W*pinv(F)))

    if length(DE.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial QFI is $(1.0/f_ini)")
    
        f_list = [1.0/f_ini]
        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_state(f_list, populations[indx].psi)
            for i in 1:(max_episode-1)
                p_fit = train_QFIM_noiseless(populations, c, cr, p_num, dim, p_fit)
                indx = findmax(p_fit)[2]
                append!(f_list, maximum(p_fit))
                SaveFile_state(f_list, populations[indx].psi)
                print("current QFI is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_QFIM_noiseless(populations, c, cr, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final QFI is ", maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_QFIM_noiseless(populations, c, cr, p_num, dim, p_fit)
                append!(f_list, maximum(p_fit))
                print("current QFI is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_QFIM_noiseless(populations, c, cr, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final QFI is ", maximum(p_fit))
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial value of Tr(WF^{-1}) is $(f_ini)")

        f_list = [f_ini]
        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_state(f_list, populations[indx].psi)
            for i in 1:(max_episode-1)
                p_fit = train_QFIM_noiseless(populations, c, cr, p_num, dim, p_fit)
                indx = findmax(p_fit)[2]
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_state(f_list, populations[indx].psi)
                print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_QFIM_noiseless(populations, c, cr, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_QFIM_noiseless(populations, c, cr, p_num, dim, p_fit)
                append!(f_list, 1.0/maximum(p_fit))
                print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_QFIM_noiseless(populations, c, cr, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
        end
    end
end

function DE_CFIM(Measurement, DE::TimeIndepend_noiseless{T}, popsize, ini_population, c, cr, seed, max_episode, save_file) where {T<: Complex}
    println("state optimization")
    Random.seed!(seed)
    dim = length(DE.psi)

    p_num = popsize
    populations = repeat(DE, p_num)
    # initialize 
    if length(ini_population) > popsize
        ini_population = [ini_population[i] for i in 1:popsize]
    end 
    for pj in 1:length(ini_population)
        populations[pj].psi = [ini_population[pj][i] for i in 1:dim]
    end
    for pj in (length(ini_population)+1):p_num
        r_ini = 2*rand(dim)-ones(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        populations[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    p_fit = [0.0 for i in 1:p_num] 
    for pj in 1:p_num
        F_tp = CFIM_TimeIndepend(Measurement, DE.freeHamiltonian, DE.Hamiltonian_derivative, populations[pj].psi, DE.tspan, DE.accuracy)
        p_fit[pj] = 1.0/real(tr(DE.W*pinv(F_tp)))
    end

    F = CFIM_TimeIndepend(Measurement, DE.freeHamiltonian, DE.Hamiltonian_derivative, DE.psi, DE.tspan, DE.accuracy)
    f_ini= real(tr(DE.W*pinv(F)))

    if length(DE.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial CFI is $(1.0/f_ini)")

        f_list = [1.0/f_ini]
        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_state(f_list, populations[indx].psi)
            for i in 1:(max_episode-1)
                p_fit = train_CFIM_noiseless(Measurement, populations, c, cr, p_num, dim, p_fit)
                indx = findmax(p_fit)[2]
                append!(f_list, maximum(p_fit))
                SaveFile_state(f_list, populations[indx].psi)
                print("current CFI is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_CFIM_noiseless(Measurement, populations, c, cr, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final CFI is ", maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_CFIM_noiseless(Measurement, populations, c, cr, p_num, dim, p_fit)
                append!(f_list, maximum(p_fit))
                print("current CFI is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_CFIM_noiseless(Measurement, populations, c, cr, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final CFI is ", maximum(p_fit))
        end
    else  
        println("multiparameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial value of Tr(WF^{-1}) is $(f_ini)")

        f_list = [f_ini]
        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_state(f_list, populations[indx].psi)
            for i in 1:(max_episode-1)
                p_fit = train_CFIM_noiseless(Measurement, populations, c, cr, p_num, dim, p_fit)
                indx = findmax(p_fit)[2]
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_state(f_list, populations[indx].psi)
                print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_CFIM_noiseless(Measurement, populations, c, cr, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_CFIM_noiseless(Measurement, populations, c, cr, p_num, dim, p_fit)
                append!(f_list, 1.0/maximum(p_fit))
                print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_CFIM_noiseless(Measurement, populations, c, cr, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
        end
    end
end

function train_QFIM_noiseless(populations, c, cr, p_num, dim, p_fit)
    f_mean = p_fit |> mean
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace=false)
        ctrl_mut = zeros(ComplexF64, dim)
        for ci in 1:dim
            ctrl_mut[ci] = populations[mut_num[1]].psi[ci]+c*(populations[mut_num[2]].psi[ci]-populations[mut_num[3]].psi[ci])
        end
        #crossover
        # if p_fit[pj] > f_mean
        #     cr = c0 + (c1-c0)*(p_fit[pj]-minimum(p_fit))/(maximum(p_fit)-minimum(p_fit))
        # else
        #     cr = c0
        # end
        ctrl_cross = zeros(ComplexF64, dim)
        cross_int = sample(1:dim, 1, replace=false)
        for cj in 1:dim
            rand_num = rand()
            if rand_num <= cr
                ctrl_cross[cj] = ctrl_mut[cj]
            else
                ctrl_cross[cj] = populations[pj].psi[cj]
            end
            ctrl_cross[cross_int] = ctrl_mut[cross_int]
        end
        psi_cross = ctrl_cross/norm(ctrl_cross)

        #selection
        F_tp = QFIM_TimeIndepend(populations[pj].freeHamiltonian, populations[pj].Hamiltonian_derivative, psi_cross, 
                                 populations[pj].tspan, populations[pj].accuracy)
        f_cross = 1.0/real(tr(populations[pj].W*pinv(F_tp)))
        if f_cross > p_fit[pj]
            p_fit[pj] = f_cross
            for ck in 1:dim
                populations[pj].psi[ck] = psi_cross[ck]
            end
        end
    end
    return p_fit
end

function train_CFIM_noiseless(Measurement, populations, c, cr, p_num, dim, p_fit)
    f_mean = p_fit |> mean
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace=false)
        ctrl_mut = zeros(ComplexF64, dim)
        for ci in 1:dim
            ctrl_mut[ci] = populations[mut_num[1]].psi[ci]+c*(populations[mut_num[2]].psi[ci]-populations[mut_num[3]].psi[ci])
        end
        #crossover
        # if p_fit[pj] > f_mean
        #     cr = c0 + (c1-c0)*(p_fit[pj]-minimum(p_fit))/(maximum(p_fit)-minimum(p_fit))
        # else
        #     cr = c0
        # end
        ctrl_cross = zeros(ComplexF64, dim)
        cross_int = sample(1:dim, 1, replace=false)
        for cj in 1:dim
            rand_num = rand()
            if rand_num <= cr
                ctrl_cross[cj] = ctrl_mut[cj]
            else
                ctrl_cross[cj] = populations[pj].psi[cj]
            end
            ctrl_cross[cross_int] = ctrl_mut[cross_int]
        end
        psi_cross = ctrl_cross/norm(ctrl_cross)

        #selection
        F_tp = CFIM_TimeIndepend(Measurement, populations[pj].freeHamiltonian, populations[pj].Hamiltonian_derivative, psi_cross, 
                                 populations[pj].tspan, populations[pj].accuracy)
        f_cross = 1.0/real(tr(populations[pj].W*pinv(F_tp)))
        if f_cross > p_fit[pj]
            p_fit[pj] = f_cross
            for ck in 1:dim
                populations[pj].psi[ck] = psi_cross[ck]
            end
        end
    end
    return p_fit
end


############# time-independent Hamiltonian (noise) ################
function DE_QFIM(DE::TimeIndepend_noise{T}, popsize, ini_population, c, cr, seed, max_episode, save_file) where {T<: Complex}
    println("state optimization")
    Random.seed!(seed)
    dim = length(DE.psi)

    p_num = popsize
    populations = repeat(DE, p_num)
    # initialize 
    if length(ini_population) > popsize
        ini_population = [ini_population[i] for i in 1:popsize]
    end 
    for pj in 1:length(ini_population)
        populations[pj].psi = [ini_population[pj][i] for i in 1:dim]
    end
    for pj in (length(ini_population)+1):p_num
        r_ini = 2*rand(dim)-ones(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        populations[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    p_fit = [0.0 for i in 1:p_num] 
    for pj in 1:p_num
        rho = populations[pj].psi*(populations[pj].psi)'
        F_tp = QFIM_TimeIndepend(DE.freeHamiltonian, DE.Hamiltonian_derivative, rho, DE.decay_opt, DE.γ, DE.tspan, DE.accuracy)
        p_fit[pj] = 1.0/real(tr(DE.W*pinv(F_tp)))
    end

    F = QFIM_TimeIndepend(DE.freeHamiltonian, DE.Hamiltonian_derivative, DE.psi*(DE.psi)', DE.decay_opt, DE.γ, DE.tspan, DE.accuracy)
    f_ini= real(tr(DE.W*pinv(F)))
    f_list = [f_ini]

    if length(DE.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial QFI is $(f_ini)")
    
        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_state(f_list, populations[indx].psi)
            for i in 1:(max_episode-1)
                p_fit = train_QFIM_noise(populations, c, cr, p_num, dim, p_fit)
                indx = findmax(p_fit)[2]
                append!(f_list, maximum(p_fit))
                SaveFile_state(f_list, populations[indx].psi)
                print("current QFI is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_QFIM_noise(populations, c, cr, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final QFI is ", maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_QFIM_noise(populations, c, cr, p_num, dim, p_fit)
                append!(f_list, maximum(p_fit))
                print("current QFI is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_QFIM_noise(populations, c, cr, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final QFI is ", maximum(p_fit))
        end
    else    
        println("multiparameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial value of Tr(WF^{-1}) is $(f_ini)")

        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_state(f_list, populations[indx].psi)
            for i in 1:(max_episode-1)
                p_fit = train_QFIM_noise(populations, c, cr, p_num, dim, p_fit)
                indx = findmax(p_fit)[2]
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_state(f_list, populations[indx].psi)
                print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_QFIM_noise(populations, c, cr, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_QFIM_noise(populations, c, cr, p_num, dim, p_fit)
                append!(f_list, 1.0/maximum(p_fit))
                print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_QFIM_noise(populations, c, cr, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
        end
    end
end

function DE_CFIM(Measurement, DE::TimeIndepend_noise{T}, popsize, ini_population, c, cr, seed, max_episode, save_file) where {T<: Complex}
    println("state optimization")
    Random.seed!(seed)
    dim = length(DE.psi)

    p_num = popsize
    populations = repeat(DE, p_num)
    # initialize 
    if length(ini_population) > popsize
        ini_population = [ini_population[i] for i in 1:popsize]
    end 
    for pj in 1:length(ini_population)
        populations[pj].psi = [ini_population[pj][i] for i in 1:dim]
    end
    for pj in (length(ini_population)+1):p_num
        r_ini = 2*rand(dim)-ones(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        populations[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    p_fit = [0.0 for i in 1:p_num] 
    for pj in 1:p_num
        rho = populations[pj].psi*(populations[pj].psi)'
        F_tp = CFIM_TimeIndepend(Measurement, DE.freeHamiltonian, DE.Hamiltonian_derivative, rho, DE.decay_opt, DE.γ, DE.tspan, DE.accuracy)
        p_fit[pj] = 1.0/real(tr(DE.W*pinv(F_tp)))
    end

    F = CFIM_TimeIndepend(Measurement, DE.freeHamiltonian, DE.Hamiltonian_derivative, DE.psi*(DE.psi)', DE.decay_opt, DE.γ, DE.tspan, DE.accuracy)
    f_ini= real(tr(DE.W*pinv(F)))
    f_list = [f_ini]

    if length(DE.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial CFI is $(f_ini)")
    
        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_state(f_list, populations[indx].psi)
            for i in 1:(max_episode-1)
                p_fit = train_CFIM_noise(Measurement, populations, c, cr, p_num, dim, p_fit)
                indx = findmax(p_fit)[2]
                append!(f_list, maximum(p_fit))
                SaveFile_state(f_list, populations[indx].psi)
                print("current CFI is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_CFIM_noise(Measurement, populations, c, cr, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final CFI is ", maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_CFIM_noise(Measurement, populations, c, cr, p_num, dim, p_fit)
                append!(f_list, maximum(p_fit))
                print("current CFI is ", maximum(p_fit), " ($i eposides)    \r")   
            end
            p_fit = train_CFIM_noise(Measurement, populations, c, cr, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final CFI is ", maximum(p_fit))
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial value of Tr(WF^{-1}) is $(f_ini)")

        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_state(f_list, populations[indx].psi)
            for i in 1:(max_episode-1)
                p_fit = train_CFIM_noise(Measurement, populations, c, cr, p_num, dim, p_fit)
                indx = findmax(p_fit)[2]
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_state(f_list, populations[indx].psi)
                print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_CFIM_noise(Measurement, populations, c, cr, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_CFIM_noise(Measurement, populations, c, cr, p_num, dim, p_fit)
                append!(f_list, 1.0/maximum(p_fit))
                print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_CFIM_noise(Measurement, populations, c, cr, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
        end
    end
end

function train_QFIM_noise(populations, c, cr, p_num, dim, p_fit)
    f_mean = p_fit |> mean
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace=false)
        ctrl_mut = zeros(ComplexF64, dim)
        for ci in 1:dim
            ctrl_mut[ci] = populations[mut_num[1]].psi[ci]+c*(populations[mut_num[2]].psi[ci]-populations[mut_num[3]].psi[ci])
        end
        #crossover
        # if p_fit[pj] > f_mean
        #     cr = c0 + (c1-c0)*(p_fit[pj]-minimum(p_fit))/(maximum(p_fit)-minimum(p_fit))
        # else
        #     cr = c0
        # end
        ctrl_cross = zeros(ComplexF64, dim)
        cross_int = sample(1:dim, 1, replace=false)
        for cj in 1:dim
            rand_num = rand()
            if rand_num <= cr
                ctrl_cross[cj] = ctrl_mut[cj]
            else
                ctrl_cross[cj] = populations[pj].psi[cj]
            end
            ctrl_cross[cross_int] = ctrl_mut[cross_int]
        end
        psi_cross = ctrl_cross/norm(ctrl_cross)

        #selection
        F_tp = QFIM_TimeIndepend(populations[pj].freeHamiltonian, populations[pj].Hamiltonian_derivative, psi_cross*psi_cross', 
                                 populations[pj].decay_opt, populations[pj].γ, populations[pj].tspan, populations[pj].accuracy)
        f_cross = 1.0/real(tr(populations[pj].W*pinv(F_tp)))
        if f_cross > p_fit[pj]
            p_fit[pj] = f_cross
            for ck in 1:dim
                populations[pj].psi[ck] = psi_cross[ck]
            end
        end
    end
    return p_fit
end

function train_CFIM_noise(Measurement, populations, c, cr, p_num, dim, p_fit)
    f_mean = p_fit |> mean
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace=false)
        ctrl_mut = zeros(ComplexF64, dim)
        for ci in 1:dim
            ctrl_mut[ci] = populations[mut_num[1]].psi[ci]+c*(populations[mut_num[2]].psi[ci]-populations[mut_num[3]].psi[ci])
        end
        #crossover
        # if p_fit[pj] > f_mean
        #     cr = c0 + (c1-c0)*(p_fit[pj]-minimum(p_fit))/(maximum(p_fit)-minimum(p_fit))
        # else
        #     cr = c0
        # end
        ctrl_cross = zeros(ComplexF64, dim)
        cross_int = sample(1:dim, 1, replace=false)
        for cj in 1:dim
            rand_num = rand()
            if rand_num <= cr
                ctrl_cross[cj] = ctrl_mut[cj]
            else
                ctrl_cross[cj] = populations[pj].psi[cj]
            end
            ctrl_cross[cross_int] = ctrl_mut[cross_int]
        end
        psi_cross = ctrl_cross/norm(ctrl_cross)

        #selection
        F_tp = CFIM_TimeIndepend(Measurement, populations[pj].freeHamiltonian, populations[pj].Hamiltonian_derivative, psi_cross*psi_cross', 
                                 populations[pj].decay_opt, populations[pj].γ, populations[pj].tspan, populations[pj].accuracy)
        f_cross = 1.0/real(tr(populations[pj].W*pinv(F_tp)))
        if f_cross > p_fit[pj]
            p_fit[pj] = f_cross
            for ck in 1:dim
                populations[pj].psi[ck] = psi_cross[ck]
            end
        end
    end
    return p_fit
end
