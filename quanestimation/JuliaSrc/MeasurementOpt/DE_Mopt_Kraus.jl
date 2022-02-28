function CFIM_DE_Mopt(DE::projection_Mopt_Kraus{T}, popsize, ini_population, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM_noctrl_Kraus")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_DE_projection_Kraus(DE, popsize, ini_population, c, cr, seed, max_episode, save_file, sym, str1, str2)
end


function CFIM_DE_Mopt(DE::LinearComb_Mopt_Kraus{T}, popsize, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM_noctrl_Kraus")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_DE_LinearComb_Kraus(DE, popsize, c, cr, seed, max_episode, save_file, sym, str1, str2)
end

function CFIM_DE_Mopt(DE::RotateBasis_Mopt_Kraus{T}, popsize, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM_noctrl_Kraus")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_DE_RotateBasis_Kraus(DE, popsize, c, cr, seed, max_episode, save_file, sym, str1, str2)
end

function info_DE_projection_Kraus(DE, popsize, ini_population, c, cr, seed, max_episode, save_file, sym, str1, str2)
    println("measurement optimization")
    Random.seed!(seed)
    dim = size(DE.ρ0)[1]
    M_num = length(DE.C)

    p_num = popsize
    populations = repeat(DE, p_num)
    # initialize 
    if length(ini_population) > popsize
        ini_population = [ini_population[i] for i in 1:popsize]
    end
    for pj in 1:length(ini_population)
        populations[pj].C = [[ini_population[pj][i,j] for j in 1:dim] for i in 1:M_num]
    end
    for pj in (length(ini_population)+1):p_num
        M_tp = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
        for mi in 1:M_num
            r_ini = 2*rand(dim)-ones(dim)
            r = r_ini/norm(r_ini)
            phi = 2*pi*rand(dim)
            M_tp[mi] = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
        end
        populations[pj].C = [[M_tp[i][j] for j in 1:dim] for i in 1:M_num]
        # orthogonality and normalization 
        populations[pj].C = gramschmidt(populations[pj].C)
    end

    p_fit = [0.0 for i in 1:p_num] 
    for pj in 1:p_num
        M = [populations[pj].C[i]*(populations[pj].C[i])' for i in 1:M_num]
        p_fit[pj] = 1.0/obj_func(Val{sym}(), DE, M)
    end

    f_ini= p_fit[1]
    f_opt = obj_func(Val{:QFIM_noctrl_Kraus}(), DE, DE.C)
    f_opt= 1.0/f_opt

    if length(DE.dK) == 1
        f_list = [f_ini]

        println("single parameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial $str1 is $(f_ini)")
        println("QFI is $(f_opt)")
        
        if save_file == true
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i]*(populations[indx].C[i])' for i in 1:M_num]
            SaveFile_meas(f_list, M)
            for i in 1:(max_episode-1)
                p_fit = train_projection(populations, c, cr, p_num, dim, M_num, p_fit, sym)
                indx = findmax(p_fit)[2]
                M = [populations[indx].C[i]*(populations[indx].C[i])' for i in 1:M_num]
                append!(f_list, maximum(p_fit))
                SaveFile_meas(f_list, M)
                print("current $str1 is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_projection(populations, c, cr, p_num, dim, M_num, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i]*(populations[indx].C[i])' for i in 1:M_num]
            append!(f_list, maximum(p_fit))
            SaveFile_meas(f_list, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str1 is ", maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_projection(populations, c, cr, p_num, dim, M_num, p_fit, sym)
                append!(f_list, maximum(p_fit))
                print("current $str1 is ", maximum(p_fit), " ($i eposides)    \r")   
            end
            p_fit = train_projection(populations, c, cr, p_num, dim, M_num, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i]*(populations[indx].C[i])' for i in 1:M_num]
            append!(f_list, maximum(p_fit))
            SaveFile_meas(f_list, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str1 is ", maximum(p_fit))
        end
    else
        f_list = [1.0/f_ini]
        println("multiparameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial value of $str2 is $(1.0/f_ini)")
        println("tr(WF^{-1}) is $(1.0/f_opt)")

        if save_file == true
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i]*(populations[indx].C[i])' for i in 1:M_num]
            SaveFile_meas(f_list, M)
            for i in 1:(max_episode-1)
                p_fit = train_projection(populations, c, cr, p_num, dim, M_num, p_fit, sym)
                indx = findmax(p_fit)[2]
                M = [populations[indx].C[i]*(populations[indx].C[i])' for i in 1:M_num]
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_meas(f_list, M)
                print("current value of $str2 is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_projection(populations, c, cr, p_num, dim, M_num, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i]*(populations[indx].C[i])' for i in 1:M_num]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_meas(f_list, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str2 is ", 1.0/maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_projection(populations, c, cr, p_num, dim, M_num, p_fit, sym)
                append!(f_list, 1.0/maximum(p_fit))
                print("current value of $str2 is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_projection(populations, c, cr, p_num, dim, M_num, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i]*(populations[indx].C[i])' for i in 1:M_num]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_meas(f_list, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str2 is ", 1.0/maximum(p_fit))
        end
    end
end

function info_DE_LinearComb_Kraus(DE, popsize, c, cr, seed, max_episode, save_file, sym, str1, str2) where {T<:Complex}
    println("measurement optimization")
    Random.seed!(seed)
    dim = size(DE.ρ0)[1]
    POVM_basis = DE.povm_basis
    basis_num = length(POVM_basis)
    M_num = DE.M_num
    p_num = popsize
    populations = repeat(DE, p_num)
    # initialize 
    B_all = [[zeros(basis_num) for i in 1:M_num] for j in 1:p_num]
    for pj in 1:p_num
        B_all[pj] = [rand(basis_num) for i in 1:M_num]
        B_all[pj] = bound_LC_coeff(B_all[pj])
    end

    p_fit = [0.0 for i in 1:p_num] 
    for pj in 1:p_num
        M = [sum([B_all[pj][i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
        p_fit[pj] = 1.0/obj_func(Val{sym}(), DE, M)
    end

    f_ini= p_fit[1]
    f_opt = obj_func(Val{:QFIM_noctrl_Kraus}(), DE, POVM_basis)
    f_opt = 1.0/f_opt

    f_povm = obj_func(Val{sym}(), DE, POVM_basis)
    f_povm = 1.0/f_povm

    if length(DE.dK) == 1
        f_list = [f_ini]

        println("single parameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial $str1 is $(f_ini)")
        println("CFI under the given POVMs is $(f_povm)")
        println("QFI is $(f_opt)")
        
        if save_file == true
            indx = findmax(p_fit)[2]
            M = [sum([B_all[indx][i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
            SaveFile_meas(f_list, M)
            for i in 1:(max_episode-1)
                p_fit, B_all = train_LinearComb(populations, B_all, POVM_basis, c, cr, p_num, basis_num, M_num, p_fit, sym)
                indx = findmax(p_fit)[2]
                M = [sum([B_all[indx][i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
                append!(f_list, maximum(p_fit))
                SaveFile_meas(f_list, M)
                print("current $str1 is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit, B_all = train_LinearComb(populations, B_all, POVM_basis, c, cr, p_num, basis_num, M_num, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [sum([B_all[indx][i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
            append!(f_list, maximum(p_fit))
            SaveFile_meas(f_list, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str1 is ", maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit, B_all = train_LinearComb(populations, B_all, POVM_basis, c, cr, p_num, basis_num, M_num, p_fit, sym)
                append!(f_list, maximum(p_fit))
                print("current $str1 is ", maximum(p_fit), " ($i eposides)    \r")   
            end
            p_fit, B_all = train_LinearComb(populations, B_all, POVM_basis, c, cr, p_num, basis_num, M_num, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [sum([B_all[indx][i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
            append!(f_list, maximum(p_fit))
            SaveFile_meas(f_list, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str1 is ", maximum(p_fit))
        end
    else
        f_list = [1.0/f_ini]
        println("multiparameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial value of $str2 is $(1.0/f_ini)")
        println("tr(WI^{-1}) under the given POVMs is $(1.0/f_povm)")
        println("tr(WF^{-1}) is $(1.0/f_opt)")

        if save_file == true
            indx = findmax(p_fit)[2]
            M = [sum([B_all[indx][i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
            SaveFile_meas(f_list, M)
            for i in 1:(max_episode-1)
                p_fit, B_all = train_LinearComb(populations, B_all, POVM_basis, c, cr, p_num, basis_num, M_num, p_fit, sym)
                indx = findmax(p_fit)[2]
                M = [sum([B_all[indx][i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_meas(f_list, M)
                print("current value of $str2 is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit, B_all = train_LinearComb(populations, B_all, POVM_basis, c, cr, p_num, basis_num, M_num, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [sum([B_all[indx][i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_meas(f_list, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str2 is ", 1.0/maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit, B_all = train_LinearComb(populations, B_all, POVM_basis, c, cr, p_num, basis_num, M_num, p_fit, sym)
                append!(f_list, 1.0/maximum(p_fit))
                print("current value of $str2 is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit, B_all = train_LinearComb(populations, B_all, POVM_basis, c, cr, p_num, basis_num, M_num, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [sum([B_all[indx][i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_meas(f_list, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str2 is ", 1.0/maximum(p_fit))
        end
    end
end

function info_DE_RotateBasis_Kraus(DE, popsize, c, cr, seed, max_episode, save_file, sym, str1, str2) where {T<:Complex}
    println("measurement optimization")
    Random.seed!(seed)
    dim = size(DE.ρ0)[1]
    suN = suN_generator(dim)
    Lambda = [Matrix{ComplexF64}(I,dim,dim)]
    append!(Lambda, [suN[i] for i in 1:length(suN)])

    POVM_basis = DE.povm_basis
    M_num = length(POVM_basis)
    p_num = popsize
    populations = repeat(DE, p_num)
    s_all = [zeros(dim*dim) for i in 1:p_num]
    # initialize 
    p_fit = [0.0 for i in 1:p_num] 
    for pj in 1:p_num
        # generate a rotation matrix randomly
        s_all[pj] = rand(dim*dim)
        U = rotation_matrix(s_all[pj], Lambda)
        M = [U*POVM_basis[i]*U' for i in 1:M_num]
        p_fit[pj] = 1.0/obj_func(Val{sym}(), DE, M)
    end

    f_ini= p_fit[1]
    f_opt = obj_func(Val{:QFIM_noctrl_Kraus}(), DE, POVM_basis)
    f_opt = 1.0/f_opt

    f_povm = obj_func(Val{sym}(), DE, POVM_basis)
    f_povm = 1.0/f_povm

    if length(DE.dK) == 1
        f_list = [f_ini]

        println("single parameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial $str1 is $(f_ini)")
        println("CFI under the given POVMs is $(f_povm)")
        println("QFI is $(f_opt)")
        
        if save_file == true
            indx = findmax(p_fit)[2]
            U = rotation_matrix(s_all[indx], Lambda)
            M = [U*POVM_basis[i]*U' for i in 1:M_num]
            SaveFile_meas(f_list, M)
            for i in 1:(max_episode-1)
                p_fit, s_all = train_RotateBasis(populations, s_all, POVM_basis, Lambda, c, cr, p_num, dim, M_num, p_fit, sym)
                indx = findmax(p_fit)[2]
                U = rotation_matrix(s_all[indx], Lambda)
                M = [U*POVM_basis[i]*U' for i in 1:M_num]
                append!(f_list, maximum(p_fit))
                SaveFile_meas(f_list, M)
                print("current $str1 is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit, s_all = train_RotateBasis(populations, s_all, POVM_basis, Lambda, c, cr, p_num, dim, M_num, p_fit, sym)
            indx = findmax(p_fit)[2]
            U = rotation_matrix(s_all[indx], Lambda)
            M = [U*POVM_basis[i]*U' for i in 1:M_num]
            append!(f_list, maximum(p_fit))
            SaveFile_meas(f_list, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str1 is ", maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit, s_all = train_RotateBasis(populations, s_all, POVM_basis, Lambda, c, cr, p_num, dim, M_num, p_fit, sym)
                append!(f_list, maximum(p_fit))
                print("current $str1 is ", maximum(p_fit), " ($i eposides)    \r")   
            end
            p_fit, s_all = train_RotateBasis(populations, s_all, POVM_basis, Lambda, c, cr, p_num, dim, M_num, p_fit, sym)
            indx = findmax(p_fit)[2]
            U = rotation_matrix(s_all[indx], Lambda)
            M = [U*POVM_basis[i]*U' for i in 1:M_num]
            append!(f_list, maximum(p_fit))
            SaveFile_meas(f_list, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str1 is ", maximum(p_fit))
        end
    else
        f_list = [1.0/f_ini]
        println("multiparameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial value of $str2 is $(1.0/f_ini)")
        println("tr(WI^{-1}) under the given POVMs is $(1.0/f_povm)")
        println("tr(WF^{-1}) is $(1.0/f_opt)")

        if save_file == true
            indx = findmax(p_fit)[2]
            U = rotation_matrix(s_all[indx], Lambda)
            M = [U*POVM_basis[i]*U' for i in 1:M_num]
            SaveFile_meas(f_list, M)
            for i in 1:(max_episode-1)
                p_fit, s_all = train_RotateBasis(populations, s_all, POVM_basis, Lambda, c, cr, p_num, dim, M_num, p_fit, sym)
                indx = findmax(p_fit)[2]
                U = rotation_matrix(s_all[indx], Lambda)
                M = [U*POVM_basis[i]*U' for i in 1:M_num]
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_meas(f_list, M)
                print("current value of $str2 is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit, s_all = train_RotateBasis(populations, s_all, POVM_basis, Lambda, c, cr, p_num, dim, M_num, p_fit, sym)
            indx = findmax(p_fit)[2]
            U = rotation_matrix(s_all[indx], Lambda)
            M = [U*POVM_basis[i]*U' for i in 1:M_num]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_meas(f_list, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str2 is ", 1.0/maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit, s_all = train_RotateBasis(populations, s_all, POVM_basis, Lambda, c, cr, p_num, dim, M_num, p_fit, sym)
                append!(f_list, 1.0/maximum(p_fit))
                print("current value of $str2 is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit, s_all = train_RotateBasis(populations, s_all, POVM_basis, Lambda, c, cr, p_num, dim, M_num, p_fit, sym)
            indx = findmax(p_fit)[2]
            U = rotation_matrix(s_all[indx], Lambda)
            M = [U*POVM_basis[i]*U' for i in 1:M_num]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_meas(f_list, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str2 is ", 1.0/maximum(p_fit))
        end
    end
end