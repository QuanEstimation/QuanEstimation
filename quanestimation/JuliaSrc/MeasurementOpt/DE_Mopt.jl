################ projection measurement ###############
function CFIM_DE_Mopt(DE::projection_Mopt{T}, popsize, ini_population, c, cr, seed, max_episode, save_file) where {T<:Complex}
    println("measurement optimization")
    Random.seed!(seed)
    dim = size(DE.ρ0)[1]
    M_num = length(DE.Measurement)

    p_num = popsize
    populations = repeat(DE, p_num)
    # initialize 
    if length(ini_population) > popsize
        ini_population = [ini_population[i] for i in 1:popsize]
    end 
    for pj in 1:length(ini_population)
        populations[pj].Measurement = [[ini_population[pj][i,j] for j in 1:dim] for i in 1:M_num]
    end
    for pj in (length(ini_population)+1):p_num
        M_tp = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
        for mi in 1:M_num
            r_ini = 2*rand(dim)-ones(dim)
            r = r_ini/norm(r_ini)
            phi = 2*pi*rand(dim)
            M_tp[mi] = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
        end
        populations[pj].Measurement = [[M_tp[i][j] for j in 1:dim] for i in 1:M_num]
        # orthogonality and normalization 
        populations[pj].Measurement = gramschmidt(populations[pj].Measurement)
    end

    p_fit = [0.0 for i in 1:p_num] 
    for pj in 1:p_num
        Measurement = [populations[pj].Measurement[i]*(populations[pj].Measurement[i])' for i in 1:M_num]
        F_tp = CFIM(Measurement, DE.freeHamiltonian, DE.Hamiltonian_derivative, DE.ρ0, DE.decay_opt, DE.γ, DE.tspan, DE.accuracy)
        p_fit[pj] = 1.0/real(tr(DE.W*pinv(F_tp)))
    end

    f_ini= p_fit[1]
    F_opt = QFIM(DE.freeHamiltonian, DE.Hamiltonian_derivative, DE.ρ0, DE.decay_opt, DE.γ, DE.tspan, DE.accuracy)
    f_opt= real(tr(DE.W*pinv(F_opt)))

    if length(DE.Hamiltonian_derivative) == 1
        f_list = [f_ini]

        println("single parameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial CFI is $(f_ini)")
        println("QFI is $(1.0/f_opt)")
        
        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_meas(f_list, populations[indx].Measurement)
            for i in 1:(max_episode-1)
                p_fit = train_CFIM_projection(populations, c, cr, p_num, dim, M_num, p_fit)
                indx = findmax(p_fit)[2]
                append!(f_list, maximum(p_fit))
                SaveFile_meas(f_list, populations[indx].Measurement)
                print("current CFI is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_CFIM_projection(populations, c, cr, p_num, dim, M_num, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_meas(f_list, populations[indx].Measurement)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final CFI is ", maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_CFIM_projection(populations, c, cr, p_num, dim, M_num, p_fit)
                append!(f_list, maximum(p_fit))
                print("current CFI is ", maximum(p_fit), " ($i eposides)    \r")   
            end
            p_fit = train_CFIM_projection(populations, c, cr, p_num, dim, M_num, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_meas(f_list, populations[indx].Measurement)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final CFI is ", maximum(p_fit))
        end
    else
        f_list = [1.0/f_ini]
        println("multiparameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial value of Tr(WI^{-1}) is $(1.0/f_ini)")
        println("Tr(WF^{-1}) is $(f_opt)")

        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_meas(f_list, populations[indx].Measurement)
            for i in 1:(max_episode-1)
                p_fit = train_CFIM_projection(populations, c, cr, p_num, dim, M_num, p_fit)
                indx = findmax(p_fit)[2]
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_meas(f_list, populations[indx].Measurement)
                print("current value of Tr(WI^{-1}) is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_CFIM_projection(populations, c, cr, p_num, dim, M_num, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_meas(f_list, populations[indx].Measurement)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of Tr(WI^{-1}) is ", 1.0/maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = train_CFIM_projection(populations, c, cr, p_num, dim, M_num, p_fit)
                append!(f_list, 1.0/maximum(p_fit))
                print("current value of Tr(WI^{-1}) is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit = train_CFIM_projection(populations, c, cr, p_num, dim, M_num, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_meas(f_list, populations[indx].Measurement)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of Tr(WI^{-1}) is ", 1.0/maximum(p_fit))
        end
    end
end

function train_CFIM_projection(populations, c, cr, p_num, dim, M_num, p_fit)
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace=false)
        M_mut = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
        for ci in 1:M_num
            for ti in 1:dim
                M_mut[ci][ti] = populations[mut_num[1]].Measurement[ci][ti] + c*(populations[mut_num[2]].Measurement[ci][ti]-
                                populations[mut_num[3]].Measurement[ci][ti])
            end
        end
        #crossover
        # f_mean = p_fit |> mean
        # if p_fit[pj] > f_mean
        #     cr = c0 + (c1-c0)*(p_fit[pj]-minimum(p_fit))/(maximum(p_fit)-minimum(p_fit))
        # else
        #     cr = c0
        # end
        M_cross = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
        for cj in 1:M_num
            cross_int = sample(1:dim, 1, replace=false)
            for tj in 1:dim
                rand_num = rand()
                if rand_num <= cr
                    M_cross[cj][tj] = M_mut[cj][tj]
                else
                    M_cross[cj][tj] = populations[pj].Measurement[cj][tj]
                end
            end
            M_cross[cj][cross_int] = M_mut[cj][cross_int]
        end

        # orthogonality and normalization 
        M_cross = gramschmidt(M_cross)

        Measurement = [M_cross[i]*(M_cross[i])' for i in 1:M_num]

        #selection
        F_tp = CFIM_TimeIndepend(Measurement, populations[pj].freeHamiltonian, populations[pj].Hamiltonian_derivative, populations[pj].ρ0, 
                                 populations[pj].decay_opt, populations[pj].γ, populations[pj].tspan, populations[pj].accuracy)
        f_cross = 1.0/real(tr(populations[pj].W*pinv(F_tp)))
        if f_cross > p_fit[pj]
            p_fit[pj] = f_cross
            for ck in 1:M_num
                for tk in 1:dim
                    populations[pj].Measurement[ck][tk] = M_cross[ck][tk]
                end
            end
        end
    end
    return p_fit
end


################## update the coefficients according to the given basis ############

function CFIM_DE_Mopt(DE::givenpovm_Mopt{T}, popsize, c, cr, seed, max_episode, save_file) where {T<:Complex}
    println("measurement optimization")
    Random.seed!(seed)
    dim = size(DE.ρ0)[1]
    POVM_basis = DE.povm_basis
    M_num = DE.M_num
    p_num = popsize
    populations = repeat(DE, p_num)
    # initialize 
    coeff = [[zeros(dim^2) for i in 1:M_num] for j in 1:p_num]
    for pj in 1:p_num
        coeff[pj] = generate_coeff(M_num, dim)
    end

    p_fit = [0.0 for i in 1:p_num] 
    for pj in 1:p_num
        Measurement = [sum([coeff[pj][i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
        F_tp = CFIM(Measurement, DE.freeHamiltonian, DE.Hamiltonian_derivative, DE.ρ0, DE.decay_opt, DE.γ, DE.tspan, DE.accuracy)
        p_fit[pj] = 1.0/real(tr(DE.W*pinv(F_tp)))
    end

    f_ini= p_fit[1]
    F_opt = QFIM(DE.freeHamiltonian, DE.Hamiltonian_derivative, DE.ρ0, DE.decay_opt, DE.γ, DE.tspan, DE.accuracy)
    f_opt= real(tr(DE.W*pinv(F_opt)))

    F_povm = CFIM(POVM_basis, DE.freeHamiltonian, DE.Hamiltonian_derivative, DE.ρ0, DE.decay_opt, DE.γ, DE.tspan, DE.accuracy)
    f_povm= real(tr(DE.W*pinv(F_povm)))

    if length(DE.Hamiltonian_derivative) == 1
        f_list = [f_ini]

        println("single parameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial CFI is $(f_ini)")
        println("CFI under the given POVMs is $(1.0/f_povm)")
        println("QFI is $(1.0/f_opt)")
        
        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_meas(f_list, coeff[indx])
            for i in 1:(max_episode-1)
                p_fit, coeff = train_CFIM_givenpovm(populations, coeff, POVM_basis, c, cr, p_num, dim, M_num, p_fit)
                indx = findmax(p_fit)[2]
                append!(f_list, maximum(p_fit))
                SaveFile_meas(f_list, coeff[indx])
                print("current CFI is ", maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit, coeff = train_CFIM_givenpovm(populations, coeff, POVM_basis, c, cr, p_num, dim, M_num, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_meas(f_list, coeff[indx])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final CFI is ", maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit, coeff = train_CFIM_givenpovm(populations, coeff, POVM_basis, c, cr, p_num, dim, M_num, p_fit)
                append!(f_list, maximum(p_fit))
                print("current CFI is ", maximum(p_fit), " ($i eposides)    \r")   
            end
            p_fit, coeff = train_CFIM_givenpovm(populations, coeff, POVM_basis, c, cr, p_num, dim, M_num, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_meas(f_list, coeff[indx])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final CFI is ", maximum(p_fit))
        end
    else
        f_list = [1.0/f_ini]
        println("multiparameter scenario")
        println("search algorithm: Differential Evolution (DE)")
        println("initial value of Tr(WI^{-1}) is $(1.0/f_ini)")
        println("Tr(WI^{-1}) under the given POVMs is $(f_povm)")
        println("Tr(WF^{-1}) is $(f_opt)")

        if save_file == true
            indx = findmax(p_fit)[2]
            SaveFile_meas(f_list, coeff[indx])
            for i in 1:(max_episode-1)
                p_fit, coeff = train_CFIM_givenpovm(populations, coeff, POVM_basis, c, cr, p_num, dim, M_num, p_fit)
                indx = findmax(p_fit)[2]
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_meas(f_list, coeff[indx])
                print("current value of Tr(WI^{-1}) is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit, coeff = train_CFIM_givenpovm(populations, coeff, POVM_basis, c, cr, p_num, dim, M_num, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_meas(f_list, coeff[indx])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of Tr(WI^{-1}) is ", 1.0/maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit, coeff = train_CFIM_givenpovm(populations, coeff, POVM_basis, c, cr, p_num, dim, M_num, p_fit)
                append!(f_list, 1.0/maximum(p_fit))
                print("current value of Tr(WI^{-1}) is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            end
            p_fit, coeff = train_CFIM_givenpovm(populations, coeff, POVM_basis, c, cr, p_num, dim, M_num, p_fit)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_meas(f_list, coeff[indx])
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of Tr(WI^{-1}) is ", 1.0/maximum(p_fit))
        end
    end
end

function train_CFIM_givenpovm(populations, coeff, POVM_basis, c, cr, p_num, dim, M_num, p_fit)
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace=false)
        M_mut = [Vector{Float64}(undef, dim^2) for i in 1:M_num]
        for ci in 1:M_num
            for ti in 1:dim^2
                M_mut[ci][ti] = coeff[mut_num[1]][ci][ti] + c*(coeff[mut_num[2]][ci][ti]-coeff[mut_num[3]][ci][ti])
            end
        end
        #crossover
        # f_mean = p_fit |> mean
        # if p_fit[pj] > f_mean
        #     cr = c0 + (c1-c0)*(p_fit[pj]-minimum(p_fit))/(maximum(p_fit)-minimum(p_fit))
        # else
        #     cr = c0
        # end
        M_cross = [Vector{Float64}(undef, dim^2) for i in 1:M_num]
        for cj in 1:M_num
            cross_int = sample(1:dim^2, 1, replace=false)
            for tj in 1:dim^2
                rand_num = rand()
                if rand_num <= cr
                    M_cross[cj][tj] = M_mut[cj][tj]
                else
                    M_cross[cj][tj] = coeff[pj][cj][tj]
                end
            end
            M_cross[cj][cross_int] = M_mut[cj][cross_int]
        end

        # normalize the coefficients 
        bound!(M_cross)
        for i in 1:M_num
            for j in 1:dim^2
                M_cross[i][j] = M_cross[i][j]/sum([M_cross[m][j] for m in 1:M_num])
            end
        end
        Measurement = [sum([M_cross[i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]

        #selection
        F_tp = CFIM_TimeIndepend(Measurement, populations[pj].freeHamiltonian, populations[pj].Hamiltonian_derivative, populations[pj].ρ0, 
                                 populations[pj].decay_opt, populations[pj].γ, populations[pj].tspan, populations[pj].accuracy)
        f_cross = 1.0/real(tr(populations[pj].W*pinv(F_tp)))
        if f_cross > p_fit[pj]
            p_fit[pj] = f_cross
            for ck in 1:M_num
                for tk in 1:dim^2
                    coeff[pj][ck][tk] = M_cross[ck][tk]
                end
            end
        end
    end
    return p_fit, coeff
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
