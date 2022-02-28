
function SM_DE_Compopt(DE::SM_Compopt_Kraus{T}, popsize, psi0, measurement0, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM_SMopt_Kraus")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    M = [zeros(ComplexF64, size(DE.psi)[1], size(DE.psi)[1])]
    return info_DE_SMopt_Kraus(M, DE, popsize, psi0, measurement0, c, cr, seed, max_episode, save_file, sym, str1, str2)
end


function info_DE_SMopt_Kraus(M, DE, popsize, psi0, measurement0, c, cr, seed, max_episode, save_file, sym, str1, str2) where {T<:Complex}
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
        populations[pj].psi = [psi0[i] for i in 1:dim]
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

    if length(DE.dK) == 1
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
