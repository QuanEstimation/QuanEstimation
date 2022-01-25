############# time-independent Hamiltonian (noiseless) ################
function QFIM_DE_Sopt(DE::TimeIndepend_noiseless{T}, popsize, ini_population, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("QFIM_TimeIndepend_noiseless")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    Measurement = [zeros(ComplexF64, size(DE.psi)[1], size(DE.psi)[1])]
    return info_DE_noiseless(Measurement, DE, popsize, ini_population, c, cr, seed, max_episode, save_file, sym, str1, str2, str3)
end

function CFIM_DE_Sopt(Measurement, DE::TimeIndepend_noiseless{T}, popsize, ini_population, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM_TimeIndepend_noiseless")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_DE_noiseless(Measurement, DE, popsize, ini_population, c, cr, seed, max_episode, save_file, sym, str1, str2, str3)
end

function HCRB_DE_Sopt(DE::TimeIndepend_noiseless{T}, popsize, ini_population, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("HCRB_TimeIndepend_noiseless")
    str1 = ""
    str2 = "HCRB"
    str3 = "HCRB"
    Measurement = [zeros(ComplexF64, size(DE.psi)[1], size(DE.psi)[1])]
    if length(DE.Hamiltonian_derivative) == 1
        println("In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the objection function for state optimization.")
        return nothing
    else
        return info_DE_noiseless(Measurement, DE, popsize, ini_population, c, cr, seed, max_episode, save_file, sym, str1, str2, str3)
    end
end

function info_DE_noiseless(Measurement, DE, popsize, ini_population, c, cr, seed, max_episode, save_file, sym, str1, str2, str3) where {T<:Complex}
    println("$str1 state optimization")
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
        p_fit[pj] = 1.0/obj_func(Val{sym}(), populations[pj], Measurement)
    end

    f_ini = obj_func(Val{sym}(), DE, Measurement)

    if length(DE.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial $str2 is $(1.0/f_ini)")
    
        f_list = [1.0/f_ini]
        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_state(f_list, populations[indx].psi)
            for i in 1:(max_episode-1)
                p_fit = train_noiseless_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
                indx = findmax(p_fit)[2]
                append!(f_list, maximum(p_fit))
                SaveFile_state(f_list, populations[indx].psi)
                print("current $str2 is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_noiseless_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str2 is ", maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_noiseless_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
                append!(f_list, maximum(p_fit))
                print("current $str2 is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_noiseless_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
            indx = findmax(p_fit)[2]
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str2 is ", maximum(p_fit))
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial value of $str3 is $(f_ini)")

        f_list = [f_ini]
        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_state(f_list, populations[indx].psi)
            for i in 1:(max_episode-1)
                p_fit = train_noiseless_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
                indx = findmax(p_fit)[2]
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_state(f_list, populations[indx].psi)
                print("current value of $str3 is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_noiseless_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is ", 1.0/maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_noiseless_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
                append!(f_list, 1.0/maximum(p_fit))
                print("current value of $str3 is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_noiseless_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is ", 1.0/maximum(p_fit))
        end
    end
end

function train_noiseless_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace=false)
        ctrl_mut = zeros(ComplexF64, dim)
        for ci in 1:dim
            ctrl_mut[ci] = populations[mut_num[1]].psi[ci]+c*(populations[mut_num[2]].psi[ci]-populations[mut_num[3]].psi[ci])
        end

        ctrl_cross = zeros(ComplexF64, dim)
        cross_int = sample(1:dim, 1, replace=false)[1]
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
        f_cross = obj_func(Val{sym}(), populations[pj], Measurement, psi_cross)
        f_cross = 1.0/f_cross
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
function QFIM_DE_Sopt(DE::TimeIndepend_noise{T}, popsize, ini_population, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("QFIM_TimeIndepend_noise")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    Measurement = [zeros(ComplexF64, size(DE.psi)[1], size(DE.psi)[1])]
    return info_DE_noise(Measurement, DE, popsize, ini_population, c, cr, seed, max_episode, save_file, sym, str1, str2, str3)
end

function CFIM_DE_Sopt(Measurement, DE::TimeIndepend_noise{T}, popsize, ini_population, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM_TimeIndepend_noise")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_DE_noise(Measurement, DE, popsize, ini_population, c, cr, seed, max_episode, save_file, sym, str1, str2, str3)
end

function HCRB_DE_Sopt(DE::TimeIndepend_noise{T}, popsize, ini_population, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("HCRB_TimeIndepend_noise")
    str1 = ""
    str2 = "HCRB"
    str3 = "HCRB"
    Measurement = [zeros(ComplexF64, size(DE.psi)[1], size(DE.psi)[1])]
    if length(DE.Hamiltonian_derivative) == 1
        println("In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the objection function for state optimization.")
        return nothing
    else
        return info_DE_noise(Measurement, DE, popsize, ini_population, c, cr, seed, max_episode, save_file, sym, str1, str2, str3)
    end
end

function info_DE_noise(Measurement, DE, popsize, ini_population, c, cr, seed, max_episode, save_file, sym, str1, str2, str3) where {T<:Complex}
    println("$str1 state optimization")
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
        p_fit[pj] = 1.0/obj_func(Val{sym}(), populations[pj], Measurement)
    end

    f_ini = obj_func(Val{sym}(), DE, Measurement)
    f_list = [f_ini]

    if length(DE.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial $str2 is $(1.0/f_ini)")
    
        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_state(f_list, populations[indx].psi)
            for i in 1:(max_episode-1)
                p_fit = train_noise_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
                indx = findmax(p_fit)[2]
                append!(f_list, maximum(p_fit))
                SaveFile_state(f_list, populations[indx].psi)
                print("current $str2 is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_noise_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str2 is ", maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_noise_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
                append!(f_list, maximum(p_fit))
                print("current $str2 is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_noise_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str2 is ", maximum(p_fit))
        end
    else    
        println("multiparameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial value of $str3 is $(f_ini)")

        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_state(f_list, populations[indx].psi)
            for i in 1:(max_episode-1)
                p_fit = train_noise_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
                indx = findmax(p_fit)[2]
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_state(f_list, populations[indx].psi)
                print("current value of $str3 is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_noise_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is ", 1.0/maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_noise_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
                append!(f_list, 1.0/maximum(p_fit))
                print("current value of $str3 is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_noise_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_state(f_list, populations[indx].psi)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is ", 1.0/maximum(p_fit))
        end
    end
end

function train_noise_DE(Measurement, populations, c, cr, p_num, dim, p_fit, sym)
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace=false)
        ctrl_mut = zeros(ComplexF64, dim)
        for ci in 1:dim
            ctrl_mut[ci] = populations[mut_num[1]].psi[ci]+c*(populations[mut_num[2]].psi[ci]-populations[mut_num[3]].psi[ci])
        end

        ctrl_cross = zeros(ComplexF64, dim)
        cross_int = sample(1:dim, 1, replace=false)[1]
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
        f_cross = obj_func(Val{sym}(), populations[pj], Measurement, psi_cross)
        f_cross = 1.0/f_cross
        if f_cross > p_fit[pj]
            p_fit[pj] = f_cross
            for ck in 1:dim
                populations[pj].psi[ck] = psi_cross[ck]
            end
        end
    end
    return p_fit
end
