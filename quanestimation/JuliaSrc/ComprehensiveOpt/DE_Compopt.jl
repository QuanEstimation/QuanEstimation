################ state and control optimization ###############
function SC_DE_Compopt(DE::SC_Compopt{T}, popsize, psi0, ctrl0, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("QFIM_SCopt")
    str1 = "QFI"
    str2 = "tr(WF^{-1})"
    M = [zeros(ComplexF64, size(DE.psi)[1], size(DE.psi)[1])]
    return info_DE_SCopt(M, DE, popsize, psi0, ctrl0, c, cr, seed, max_episode, save_file, sym, str1, str2)
end

function SC_DE_Compopt(M, DE::SC_Compopt{T}, popsize, psi0, ctrl0, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM_SCopt")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_DE_SCopt(M, DE, popsize, psi0, ctrl0, c, cr, seed, max_episode, save_file, sym, str1, str2)
end

function info_DE_SCopt(M, DE, popsize, psi0, ctrl0, c, cr, seed, max_episode, save_file, sym, str1, str2) where {T<:Complex}
    println("comprehensive optimization")
    Random.seed!(seed)
    dim = length(DE.psi)
    ctrl_num = length(DE.control_Hamiltonian)
    ctrl_length = length(DE.control_coefficients[1])
    p_num = popsize
    populations = repeat(DE, p_num)

    # initialize 
    if length(psi0) > popsize
        psi0 = [psi0[i] for i in 1:popsize]
    end
    if length(ctrl0) > popsize
        ctrl0 = [ctrl0[i] for i in 1:popsize]
    end
    for pj in 1:length(psi0)
        populations[pj].psi = [psi0[pj][i] for i in 1:dim]
    end
    for pj in 1:length(ctrl0)
        populations[pj].control_coefficients = [[ctrl0[pj][i, j] for j in 1:ctrl_length] for i in 1:ctrl_num]
    end
    for pj in (length(psi0)+1):p_num
        r_ini = 2 * rand(dim) - ones(dim)
        r = r_ini / norm(r_ini)
        phi = 2 * pi * rand(dim)
        populations[pj].psi = [r[i] * exp(1.0im * phi[i]) for i in 1:dim]
    end

    if DE.ctrl_bound[1] == -Inf || DE.ctrl_bound[2] == Inf
        for pj in (length(ctrl0)+1):p_num
            populations[pj].control_coefficients = [[2 * rand() - 1.0 for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
    else
        a = DE.ctrl_bound[1]
        b = DE.ctrl_bound[2]
        for pj in (length(ctrl0)+1):p_num
            populations[pj].control_coefficients = [[(b - a) * rand() + a for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
    end

    p_fit = [0.0 for i in 1:p_num]
    for pj in 1:p_num
        p_fit[pj] = 1.0 / obj_func(Val{sym}(), populations[pj], M, populations[pj].psi, populations[pj].control_coefficients)
    end

    f_ini = p_fit[1]

    if length(DE.Hamiltonian_derivative) == 1
        f_list = [f_ini]

        println("single parameter scenario")
        println("algorithm: Differential Evolution (DE)")
        println("initial $str1 is $(f_ini)")

        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_SC(f_list, populations[indx].psi, populations[indx].control_coefficients)
            for i in 1:(max_episode-1)
                p_fit = train_DE_SCopt(populations, M, c, cr, p_num, dim, ctrl_num, ctrl_length, p_fit, sym)
                indx = findmax(p_fit)[2]
                append!(f_list, maximum(p_fit))
                SaveFile_SC(f_list, populations[indx].psi, populations[indx].control_coefficients)
                print("current $str1 is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_DE_SCopt(populations, M, c, cr, p_num, dim, ctrl_num, ctrl_length, p_fit, sym)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_SC(f_list, populations[indx].psi, populations[indx].control_coefficients)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str1 is ", maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_DE_SCopt(populations, M, c, cr, p_num, dim, ctrl_num, ctrl_length, p_fit, sym)
                append!(f_list, maximum(p_fit))
                print("current $str1 is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_DE_SCopt(populations, M, c, cr, p_num, dim, ctrl_num, ctrl_length, p_fit, sym)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_SC(f_list, populations[indx].psi, populations[indx].control_coefficients)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str1 is ", maximum(p_fit))
        end
    else
        f_list = [1.0 / f_ini]
        println("multiparameter scenario")
        println("algorithm: Differential Evolution (DE)")
        println("initial value of $str2 is $(1.0/f_ini)")

        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_SC(f_list, populations[indx].psi, populations[indx].control_coefficients)
            for i in 1:(max_episode-1)
                p_fit = train_DE_SCopt(populations, M, c, cr, p_num, dim, ctrl_num, ctrl_length, p_fit, sym)
                indx = findmax(p_fit)[2]
                append!(f_list, 1.0 / maximum(p_fit))
                SaveFile_SC(f_list, populations[indx].psi, populations[indx].control_coefficients)
                print("current value of $str2 is ", 1.0 / maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_DE_SCopt(populations, M, c, cr, p_num, dim, ctrl_num, ctrl_length, p_fit, sym)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0 / maximum(p_fit))
            SaveFile_SC(f_list, populations[indx].psi, populations[indx].control_coefficients)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str2 is ", 1.0 / maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_DE_SCopt(populations, M, c, cr, p_num, dim, ctrl_num, ctrl_length, p_fit, sym)
                append!(f_list, 1.0 / maximum(p_fit))
                print("current value of $str2 is ", 1.0 / maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_DE_SCopt(populations, M, c, cr, p_num, dim, ctrl_num, ctrl_length, p_fit, sym)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0 / maximum(p_fit))
            SaveFile_SC(f_list, populations[indx].psi, populations[indx].control_coefficients)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str2 is ", 1.0 / maximum(p_fit))
        end
    end
end

function train_DE_SCopt(populations, M, c, cr, p_num, dim, ctrl_num, ctrl_length, p_fit, sym)
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace = false)
        state_mut = zeros(ComplexF64, dim)
        for ci in 1:dim
            state_mut[ci] = populations[mut_num[1]].psi[ci] + c * (populations[mut_num[2]].psi[ci] - populations[mut_num[3]].psi[ci])
        end

        ctrl_mut = [Vector{Float64}(undef, ctrl_length) for i in 1:ctrl_num]
        for ci in 1:ctrl_num
            for ti in 1:ctrl_length
                ctrl_mut[ci][ti] = populations[mut_num[1]].control_coefficients[ci][ti] +
                                   c * (populations[mut_num[2]].control_coefficients[ci][ti] -
                                        populations[mut_num[3]].control_coefficients[ci][ti])
            end
        end

        #crossover
        state_cross = zeros(ComplexF64, dim)
        cross_int1 = sample(1:dim, 1, replace = false)[1]
        for cj in 1:dim
            rand_num = rand()
            if rand_num <= cr
                state_cross[cj] = state_mut[cj]
            else
                state_cross[cj] = populations[pj].psi[cj]
            end
            state_cross[cross_int1] = state_mut[cross_int1]
        end
        psi_cross = state_cross / norm(state_cross)

        ctrl_cross = [Vector{Float64}(undef, ctrl_length) for i in 1:ctrl_num]
        for cj in 1:ctrl_num
            cross_int2 = sample(1:ctrl_length, 1, replace = false)[1]
            for tj in 1:ctrl_length
                rand_num = rand()
                if rand_num <= cr
                    ctrl_cross[cj][tj] = ctrl_mut[cj][tj]
                else
                    ctrl_cross[cj][tj] = populations[pj].control_coefficients[cj][tj]
                end
            end
            ctrl_cross[cj][cross_int2] = ctrl_mut[cj][cross_int2]
        end
        bound!(ctrl_cross, populations[pj].ctrl_bound)

        #selection
        f_cross = obj_func(Val{sym}(), populations[pj], M, psi_cross, ctrl_cross)
        f_cross = 1.0 / f_cross

        if f_cross > p_fit[pj]
            p_fit[pj] = f_cross
            for ck in 1:dim
                populations[pj].psi[ck] = psi_cross[ck]
            end

            for ck in 1:ctrl_num
                for tk in 1:ctrl_length
                    populations[pj].control_coefficients[ck][tk] = ctrl_cross[ck][tk]
                end
            end
        end
    end
    return p_fit
end


################ state and measurement optimization ###############
function SM_DE_Compopt(DE::SM_Compopt{T}, popsize, psi0, measurement0, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM_SMopt")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    M = [zeros(ComplexF64, size(DE.psi)[1], size(DE.psi)[1])]
    return info_DE_SMopt(M, DE, popsize, psi0, measurement0, c, cr, seed, max_episode, save_file, sym, str1, str2)
end

function info_DE_SMopt(M, DE, popsize, psi0, measurement0, c, cr, seed, max_episode, save_file, sym, str1, str2) where {T<:Complex}
    println("comprehensive optimization")
    Random.seed!(seed)
    dim = length(DE.psi)
    M_num = length(DE.C)
    p_num = popsize
    populations = repeat(DE, p_num)

    # initialize 
    if length(psi0) > popsize
        psi0 = [psi0[i] for i in 1:popsize]
    end
    if length(measurement0) > popsize
        measurement0 = [measurement0[i] for i in 1:popsize]
    end
    for pj in 1:length(psi0)
        populations[pj].psi = [psi0[pj][i] for i in 1:dim]
    end
    for pj in 1:length(measurement0)
        populations[pj].C = [[measurement0[pj][i, j] for j in 1:dim] for i in 1:M_num]
    end
    for pj in (length(psi0)+1):p_num
        r_ini = 2 * rand(dim) - ones(dim)
        r = r_ini / norm(r_ini)
        phi = 2 * pi * rand(dim)
        populations[pj].psi = [r[i] * exp(1.0im * phi[i]) for i in 1:dim]
    end

    for pj in (length(measurement0)+1):p_num
        M_tp = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
        for mi in 1:M_num
            r_ini = 2 * rand(dim) - ones(dim)
            r = r_ini / norm(r_ini)
            phi = 2 * pi * rand(dim)
            M_tp[mi] = [r[i] * exp(1.0im * phi[i]) for i in 1:dim]
        end
        populations[pj].C = [[M_tp[i][j] for j in 1:dim] for i in 1:M_num]
        # orthogonality and normalization 
        populations[pj].C = gramschmidt(populations[pj].C)
    end

    p_fit = [0.0 for i in 1:p_num]
    for pj in 1:p_num
        M = [populations[pj].C[i] * (populations[pj].C[i])' for i in 1:M_num]
        p_fit[pj] = 1.0 / obj_func(Val{sym}(), populations[pj], populations[pj].psi, M)
    end

    f_ini = p_fit[1]

    if length(DE.Hamiltonian_derivative) == 1
        f_list = [f_ini]

        println("single parameter scenario")
        println("algorithm: Differential Evolution (DE)")
        println("initial $str1 is $(f_ini)")

        if save_file == true
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            SaveFile_SM(f_list, populations[indx].psi, M)
            for i in 1:(max_episode-1)
                p_fit = train_DE_SMopt(populations, M, c, cr, p_num, dim, M_num, p_fit, sym)
                indx = findmax(p_fit)[2]
                M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
                append!(f_list, maximum(p_fit))
                SaveFile_SM(f_list, populations[indx].psi, M)
                print("current $str1 is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_DE_SMopt(populations, M, c, cr, p_num, dim, M_num, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            append!(f_list, maximum(p_fit))
            SaveFile_SM(f_list, populations[indx].psi, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str1 is ", maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_DE_SMopt(populations, M, c, cr, p_num, dim, M_num, p_fit, sym)
                append!(f_list, maximum(p_fit))
                print("current $str1 is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_DE_SMopt(populations, M, c, cr, p_num, dim, M_num, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            append!(f_list, maximum(p_fit))
            SaveFile_SM(f_list, populations[indx].psi, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str1 is ", maximum(p_fit))
        end
    else
        f_list = [1.0 / f_ini]
        println("multiparameter scenario")
        println("algorithm: Differential Evolution (DE)")
        println("initial value of $str2 is $(1.0/f_ini)")

        if save_file == true
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            SaveFile_SM(f_list, populations[indx].psi, M)
            for i in 1:(max_episode-1)
                p_fit = train_DE_SMopt(populations, M, c, cr, p_num, dim, M_num, p_fit, sym)
                indx = findmax(p_fit)[2]
                M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
                append!(f_list, 1.0 / maximum(p_fit))
                SaveFile_SM(f_list, populations[indx].psi, M)
                print("current value of $str2 is ", 1.0 / maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_DE_SMopt(populations, M, c, cr, p_num, dim, M_num, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            append!(f_list, 1.0 / maximum(p_fit))
            SaveFile_SM(f_list, populations[indx].psi, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str2 is ", 1.0 / maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_DE_SMopt(populations, M, c, cr, p_num, dim, M_num, p_fit, sym)
                append!(f_list, 1.0 / maximum(p_fit))
                print("current value of $str2 is ", 1.0 / maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_DE_SMopt(populations, M, c, cr, p_num, dim, M_num, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            append!(f_list, 1.0 / maximum(p_fit))
            SaveFile_SM(f_list, populations[indx].psi, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str2 is ", 1.0 / maximum(p_fit))
        end
    end
end

function train_DE_SMopt(populations, M, c, cr, p_num, dim, M_num, p_fit, sym)
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace = false)
        state_mut = zeros(ComplexF64, dim)
        for ci in 1:dim
            state_mut[ci] = populations[mut_num[1]].psi[ci] + c * (populations[mut_num[2]].psi[ci] - populations[mut_num[3]].psi[ci])
        end

        M_mut = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
        for ci in 1:M_num
            for ti in 1:dim
                M_mut[ci][ti] = populations[mut_num[1]].C[ci][ti] + c * (populations[mut_num[2]].C[ci][ti] -
                                                                         populations[mut_num[3]].C[ci][ti])
            end
        end

        #crossover
        state_cross = zeros(ComplexF64, dim)
        cross_int1 = sample(1:dim, 1, replace = false)[1]
        for cj in 1:dim
            rand_num = rand()
            if rand_num <= cr
                state_cross[cj] = state_mut[cj]
            else
                state_cross[cj] = populations[pj].psi[cj]
            end
            state_cross[cross_int1] = state_mut[cross_int1]
        end
        psi_cross = state_cross / norm(state_cross)

        M_cross = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
        for cj in 1:M_num
            cross_int = sample(1:dim, 1, replace = false)[1]
            for tj in 1:dim
                rand_num = rand()
                if rand_num <= cr
                    M_cross[cj][tj] = M_mut[cj][tj]
                else
                    M_cross[cj][tj] = populations[pj].C[cj][tj]
                end
            end
            M_cross[cj][cross_int] = M_mut[cj][cross_int]
        end
        # orthogonality and normalization 
        M_cross = gramschmidt(M_cross)
        M = [M_cross[i] * (M_cross[i])' for i in 1:M_num]

        #selection
        f_cross = obj_func(Val{sym}(), populations[pj], psi_cross, M)
        f_cross = 1.0 / f_cross

        if f_cross > p_fit[pj]
            p_fit[pj] = f_cross
            for ck in 1:dim
                populations[pj].psi[ck] = psi_cross[ck]
            end

            for ck in 1:M_num
                for tk in 1:dim
                    populations[pj].C[ck][tk] = M_cross[ck][tk]
                end
            end
        end
    end
    return p_fit
end


################ control and measurement optimization ###############
function CM_DE_Compopt(rho0, DE::CM_Compopt{T}, popsize, ctrl0, measurement0, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM_CMopt")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    M = [zeros(ComplexF64, size(rho0)[1], size(rho0)[1])]
    return info_DE_CMopt(rho0, M, DE, popsize, ctrl0, measurement0, c, cr, seed, max_episode, save_file, sym, str1, str2)
end

function info_DE_CMopt(rho0, M, DE, popsize, ctrl0, measurement0, c, cr, seed, max_episode, save_file, sym, str1, str2) where {T<:Complex}
    println("comprehensive optimization")
    Random.seed!(seed)
    dim = size(rho0)[1]
    M_num = length(DE.C)
    ctrl_num = length(DE.control_Hamiltonian)
    ctrl_length = length(DE.control_coefficients[1])
    p_num = popsize
    populations = repeat(DE, p_num)

    # initialize 
    if length(ctrl0) > popsize
        ctrl0 = [ctrl0[i] for i in 1:popsize]
    end
    if length(measurement0) > popsize
        measurement0 = [measurement0[i] for i in 1:popsize]
    end

    for pj in 1:length(ctrl0)
        populations[pj].control_coefficients = [[ctrl0[pj][i, j] for j in 1:ctrl_length] for i in 1:ctrl_num]
    end
    for pj in 1:length(measurement0)
        populations[pj].C = [[measurement0[pj][i, j] for j in 1:dim] for i in 1:M_num]
    end

    if DE.ctrl_bound[1] == -Inf || DE.ctrl_bound[2] == Inf
        for pj in (length(ctrl0)+1):p_num
            populations[pj].control_coefficients = [[2 * rand() - 1.0 for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
    else
        a = DE.ctrl_bound[1]
        b = DE.ctrl_bound[2]
        for pj in (length(ctrl0)+1):p_num
            populations[pj].control_coefficients = [[(b - a) * rand() + a for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
    end


    for pj in (length(measurement0)+1):p_num
        M_tp = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
        for mi in 1:M_num
            r_ini = 2 * rand(dim) - ones(dim)
            r = r_ini / norm(r_ini)
            phi = 2 * pi * rand(dim)
            M_tp[mi] = [r[i] * exp(1.0im * phi[i]) for i in 1:dim]
        end
        populations[pj].C = [[M_tp[i][j] for j in 1:dim] for i in 1:M_num]
        # orthogonality and normalization 
        populations[pj].C = gramschmidt(populations[pj].C)
    end

    p_fit = [0.0 for i in 1:p_num]
    for pj in 1:p_num
        M = [populations[pj].C[i] * (populations[pj].C[i])' for i in 1:M_num]
        p_fit[pj] = 1.0 / obj_func(Val{sym}(), populations[pj], M, rho0, populations[pj].control_coefficients)
    end

    f_ini = p_fit[1]

    if length(DE.Hamiltonian_derivative) == 1
        f_list = [f_ini]

        println("single parameter scenario")
        println("algorithm: Differential Evolution (DE)")
        println("initial $str1 is $(f_ini)")

        if save_file == true
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            SaveFile_CM(f_list, populations[indx].control_coefficients, M)
            for i in 1:(max_episode-1)
                p_fit = train_DE_CMopt(populations, rho0, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
                indx = findmax(p_fit)[2]
                M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
                append!(f_list, maximum(p_fit))
                SaveFile_CM(f_list, populations[indx].control_coefficients, M)
                print("current $str1 is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_DE_CMopt(populations, rho0, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            SaveFile_CM(f_list, populations[indx].control_coefficients, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str1 is ", maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_DE_CMopt(populations, rho0, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
                append!(f_list, maximum(p_fit))
                print("current $str1 is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_DE_CMopt(populations, rho0, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            append!(f_list, maximum(p_fit))
            SaveFile_CM(f_list, populations[indx].control_coefficients, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str1 is ", maximum(p_fit))
        end
    else
        f_list = [1.0 / f_ini]
        println("multiparameter scenario")
        println("algorithm: Differential Evolution (DE)")
        println("initial value of $str2 is $(1.0/f_ini)")

        if save_file == true
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            SaveFile_CM(f_list, populations[indx].control_coefficients, M)
            for i in 1:(max_episode-1)
                p_fit = train_DE_CMopt(populations, rho0, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
                indx = findmax(p_fit)[2]
                M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
                append!(f_list, 1.0 / maximum(p_fit))
                SaveFile_CM(f_list, populations[indx].control_coefficients, M)
                print("current value of $str2 is ", 1.0 / maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_DE_CMopt(populations, rho0, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            append!(f_list, 1.0 / maximum(p_fit))
            SaveFile_CM(f_list, populations[indx].control_coefficients, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str2 is ", 1.0 / maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_DE_CMopt(populations, rho0, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
                append!(f_list, 1.0 / maximum(p_fit))
                print("current value of $str2 is ", 1.0 / maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_DE_CMopt(populations, rho0, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            append!(f_list, 1.0 / maximum(p_fit))
            SaveFile_CM(f_list, populations[indx].control_coefficients, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str2 is ", 1.0 / maximum(p_fit))
        end
    end
end

function train_DE_CMopt(populations, rho0, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace = false)
        ctrl_mut = [Vector{Float64}(undef, ctrl_length) for i in 1:ctrl_num]
        for ci in 1:ctrl_num
            for ti in 1:ctrl_length
                ctrl_mut[ci][ti] = populations[mut_num[1]].control_coefficients[ci][ti] +
                                   c * (populations[mut_num[2]].control_coefficients[ci][ti] -
                                        populations[mut_num[3]].control_coefficients[ci][ti])
            end
        end

        M_mut = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
        for ci in 1:M_num
            for ti in 1:dim
                M_mut[ci][ti] = populations[mut_num[1]].C[ci][ti] + c * (populations[mut_num[2]].C[ci][ti] -
                                                                         populations[mut_num[3]].C[ci][ti])
            end
        end

        #crossover   
        ctrl_cross = [Vector{Float64}(undef, ctrl_length) for i in 1:ctrl_num]
        for cj in 1:ctrl_num
            cross_int2 = sample(1:ctrl_length, 1, replace = false)[1]
            for tj in 1:ctrl_length
                rand_num = rand()
                if rand_num <= cr
                    ctrl_cross[cj][tj] = ctrl_mut[cj][tj]
                else
                    ctrl_cross[cj][tj] = populations[pj].control_coefficients[cj][tj]
                end
            end
            ctrl_cross[cj][cross_int2] = ctrl_mut[cj][cross_int2]
        end
        bound!(ctrl_cross, populations[pj].ctrl_bound)

        M_cross = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
        for cj in 1:M_num
            cross_int = sample(1:dim, 1, replace = false)[1]
            for tj in 1:dim
                rand_num = rand()
                if rand_num <= cr
                    M_cross[cj][tj] = M_mut[cj][tj]
                else
                    M_cross[cj][tj] = populations[pj].C[cj][tj]
                end
            end
            M_cross[cj][cross_int] = M_mut[cj][cross_int]
        end
        # orthogonality and normalization 
        M_cross = gramschmidt(M_cross)
        M = [M_cross[i] * (M_cross[i])' for i in 1:M_num]

        #selection
        f_cross = obj_func(Val{sym}(), populations[pj], M, rho0, ctrl_cross)
        f_cross = 1.0 / f_cross

        if f_cross > p_fit[pj]
            p_fit[pj] = f_cross

            for ck in 1:ctrl_num
                for tk in 1:ctrl_length
                    populations[pj].control_coefficients[ck][tk] = ctrl_cross[ck][tk]
                end
            end

            for ck in 1:M_num
                for tk in 1:dim
                    populations[pj].C[ck][tk] = M_cross[ck][tk]
                end
            end
        end
    end
    return p_fit
end


################ state, control and measurement optimization ###############
function SCM_DE_Compopt(DE::SCM_Compopt{T}, popsize, psi0, ctrl0, measurement0, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM_SCopt")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    M = [zeros(ComplexF64, size(DE.psi)[1], size(DE.psi)[1])]
    return info_DE_SCMopt(M, DE, popsize, psi0, ctrl0, measurement0, c, cr, seed, max_episode, save_file, sym, str1, str2)
end

function info_DE_SCMopt(M, DE, popsize, psi0, ctrl0, measurement0, c, cr, seed, max_episode, save_file, sym, str1, str2) where {T<:Complex}
    println("comprehensive optimization")
    Random.seed!(seed)
    dim = length(DE.psi)
    M_num = length(DE.C)
    ctrl_num = length(DE.control_Hamiltonian)
    ctrl_length = length(DE.control_coefficients[1])
    p_num = popsize
    populations = repeat(DE, p_num)

    # initialize 
    if length(psi0) > popsize
        psi0 = [psi0[i] for i in 1:popsize]
    end
    if length(ctrl0) > popsize
        ctrl0 = [ctrl0[i] for i in 1:popsize]
    end
    if length(measurement0) > popsize
        measurement0 = [measurement0[i] for i in 1:popsize]
    end

    for pj in 1:length(psi0)
        populations[pj].psi = [psi0[pj][i] for i in 1:dim]
    end
    for pj in 1:length(ctrl0)
        populations[pj].control_coefficients = [[ctrl0[pj][i, j] for j in 1:ctrl_length] for i in 1:ctrl_num]
    end
    for pj in 1:length(measurement0)
        populations[pj].C = [[measurement0[pj][i, j] for j in 1:dim] for i in 1:M_num]
    end

    for pj in (length(psi0)+1):p_num
        r_ini = 2 * rand(dim) - ones(dim)
        r = r_ini / norm(r_ini)
        phi = 2 * pi * rand(dim)
        populations[pj].psi = [r[i] * exp(1.0im * phi[i]) for i in 1:dim]
    end

    if DE.ctrl_bound[1] == -Inf || DE.ctrl_bound[2] == Inf
        for pj in (length(ctrl0)+1):p_num
            populations[pj].control_coefficients = [[2 * rand() - 1.0 for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
    else
        a = DE.ctrl_bound[1]
        b = DE.ctrl_bound[2]
        for pj in (length(ctrl0)+1):p_num
            populations[pj].control_coefficients = [[(b - a) * rand() + a for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
    end


    for pj in (length(measurement0)+1):p_num
        M_tp = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
        for mi in 1:M_num
            r_ini = 2 * rand(dim) - ones(dim)
            r = r_ini / norm(r_ini)
            phi = 2 * pi * rand(dim)
            M_tp[mi] = [r[i] * exp(1.0im * phi[i]) for i in 1:dim]
        end
        populations[pj].C = [[M_tp[i][j] for j in 1:dim] for i in 1:M_num]
        # orthogonality and normalization 
        populations[pj].C = gramschmidt(populations[pj].C)
    end

    p_fit = [0.0 for i in 1:p_num]
    for pj in 1:p_num
        M = [populations[pj].C[i] * (populations[pj].C[i])' for i in 1:M_num]
        p_fit[pj] = 1.0 / obj_func(Val{sym}(), populations[pj], M, populations[pj].psi, populations[pj].control_coefficients)
    end

    f_ini = p_fit[1]

    if length(DE.Hamiltonian_derivative) == 1
        f_list = [f_ini]

        println("single parameter scenario")
        println("algorithm: Differential Evolution (DE)")
        println("initial $str1 is $(f_ini)")

        if save_file == true
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            SaveFile_SCM(f_list, populations[indx].psi, populations[indx].control_coefficients, M)
            for i in 1:(max_episode-1)
                p_fit = train_DE_SCMopt(populations, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
                indx = findmax(p_fit)[2]
                M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
                append!(f_list, maximum(p_fit))
                SaveFile_SCM(f_list, populations[indx].psi, populations[indx].control_coefficients, M)
                print("current $str1 is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_DE_SCMopt(populations, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            SaveFile_SCM(f_list, populations[indx].psi, populations[indx].control_coefficients, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str1 is ", maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_DE_SCMopt(populations, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
                append!(f_list, maximum(p_fit))
                print("current $str1 is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_DE_SCMopt(populations, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            append!(f_list, maximum(p_fit))
            SaveFile_SCM(f_list, populations[indx].psi, populations[indx].control_coefficients, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str1 is ", maximum(p_fit))
        end
    else
        f_list = [1.0 / f_ini]
        println("multiparameter scenario")
        println("algorithm: Differential Evolution (DE)")
        println("initial value of $str2 is $(1.0/f_ini)")

        if save_file == true
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            SaveFile_SCM(f_list, populations[indx].psi, populations[indx].control_coefficients, M)
            for i in 1:(max_episode-1)
                p_fit = train_DE_SCMopt(populations, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
                indx = findmax(p_fit)[2]
                M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
                append!(f_list, 1.0 / maximum(p_fit))
                SaveFile_SCM(f_list, populations[indx].psi, populations[indx].control_coefficients, M)
                print("current value of $str2 is ", 1.0 / maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_DE_SCMopt(populations, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            append!(f_list, 1.0 / maximum(p_fit))
            SaveFile_SCM(f_list, populations[indx].psi, populations[indx].control_coefficients, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str2 is ", 1.0 / maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_DE_SCMopt(populations, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
                append!(f_list, 1.0 / maximum(p_fit))
                print("current value of $str2 is ", 1.0 / maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_DE_SCMopt(populations, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
            indx = findmax(p_fit)[2]
            M = [populations[indx].C[i] * (populations[indx].C[i])' for i in 1:M_num]
            append!(f_list, 1.0 / maximum(p_fit))
            SaveFile_SCM(f_list, populations[indx].psi, populations[indx].control_coefficients, M)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str2 is ", 1.0 / maximum(p_fit))
        end
    end
end

function train_DE_SCMopt(populations, M, c, cr, p_num, dim, M_num, ctrl_num, ctrl_length, p_fit, sym)
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace = false)
        state_mut = zeros(ComplexF64, dim)
        for ci in 1:dim
            state_mut[ci] = populations[mut_num[1]].psi[ci] + c * (populations[mut_num[2]].psi[ci] - populations[mut_num[3]].psi[ci])
        end

        ctrl_mut = [Vector{Float64}(undef, ctrl_length) for i in 1:ctrl_num]
        for ci in 1:ctrl_num
            for ti in 1:ctrl_length
                ctrl_mut[ci][ti] = populations[mut_num[1]].control_coefficients[ci][ti] +
                                   c * (populations[mut_num[2]].control_coefficients[ci][ti] -
                                        populations[mut_num[3]].control_coefficients[ci][ti])
            end
        end

        M_mut = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
        for ci in 1:M_num
            for ti in 1:dim
                M_mut[ci][ti] = populations[mut_num[1]].C[ci][ti] + c * (populations[mut_num[2]].C[ci][ti] -
                                                                         populations[mut_num[3]].C[ci][ti])
            end
        end

        #crossover
        state_cross = zeros(ComplexF64, dim)
        cross_int1 = sample(1:dim, 1, replace = false)[1]
        for cj in 1:dim
            rand_num = rand()
            if rand_num <= cr
                state_cross[cj] = state_mut[cj]
            else
                state_cross[cj] = populations[pj].psi[cj]
            end
            state_cross[cross_int1] = state_mut[cross_int1]
        end
        psi_cross = state_cross / norm(state_cross)

        ctrl_cross = [Vector{Float64}(undef, ctrl_length) for i in 1:ctrl_num]
        for cj in 1:ctrl_num
            cross_int2 = sample(1:ctrl_length, 1, replace = false)[1]
            for tj in 1:ctrl_length
                rand_num = rand()
                if rand_num <= cr
                    ctrl_cross[cj][tj] = ctrl_mut[cj][tj]
                else
                    ctrl_cross[cj][tj] = populations[pj].control_coefficients[cj][tj]
                end
            end
            ctrl_cross[cj][cross_int2] = ctrl_mut[cj][cross_int2]
        end
        bound!(ctrl_cross, populations[pj].ctrl_bound)

        M_cross = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
        for cj in 1:M_num
            cross_int = sample(1:dim, 1, replace = false)[1]
            for tj in 1:dim
                rand_num = rand()
                if rand_num <= cr
                    M_cross[cj][tj] = M_mut[cj][tj]
                else
                    M_cross[cj][tj] = populations[pj].C[cj][tj]
                end
            end
            M_cross[cj][cross_int] = M_mut[cj][cross_int]
        end
        # orthogonality and normalization 
        M_cross = gramschmidt(M_cross)
        M = [M_cross[i] * (M_cross[i])' for i in 1:M_num]

        #selection
        f_cross = obj_func(Val{sym}(), populations[pj], M, psi_cross, ctrl_cross)
        f_cross = 1.0 / f_cross

        if f_cross > p_fit[pj]
            p_fit[pj] = f_cross
            for ck in 1:dim
                populations[pj].psi[ck] = psi_cross[ck]
            end

            for ck in 1:ctrl_num
                for tk in 1:ctrl_length
                    populations[pj].control_coefficients[ck][tk] = ctrl_cross[ck][tk]
                end
            end

            for ck in 1:M_num
                for tk in 1:dim
                    populations[pj].C[ck][tk] = M_cross[ck][tk]
                end
            end
        end
    end
    return p_fit
end
