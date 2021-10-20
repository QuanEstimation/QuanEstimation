mutable struct StateOpt_DE{T <: Complex,M <: Real}
    freeHamiltonian::Matrix{T}
    Hamiltonian_derivative::Vector{Matrix{T}}
    psi::Vector{T}
    times::Vector{M}
    Liouville_operator::Vector{Matrix{T}}
    γ::Vector{M}
    control_Hamiltonian::Vector{Matrix{T}}
    control_coefficients::Vector{Vector{M}}
    ctrl_bound::M
    W::Matrix{M}
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    StateOpt_DE(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, psi::Vector{T},
             times::Vector{M}, Liouville_operator::Vector{Matrix{T}},γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}},
             control_coefficients::Vector{Vector{M}}, ctrl_bound::M, W::Matrix{M}, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = new{T,M}(freeHamiltonian, 
                Hamiltonian_derivative, psi, times, Liouville_operator, γ, control_Hamiltonian, control_coefficients, ctrl_bound, W, ρ, ∂ρ_∂x) 
end

function DiffEvo_QFI(DE::StateOpt_DE{T}, popsize, ini_population, c, c0, c1, seed, max_episodes, save_file) where {T<: Complex}
    println("state optimization")
    println("single parameter scenario")
    println("control algorithm: DE")

    Random.seed!(seed)
    dim = length(DE.psi)

    p_num = popsize
    populations = repeat(DE, p_num)
    # initialize 
    for pj in 1:length(ini_population)
        populations[pj].psi = [ini_population[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_population)+1):(p_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        populations[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    p_fit = [0.0 for i in 1:p_num] 
    for pj in 1:p_num
        rho_ini = populations[pj].psi*(populations[pj].psi)'
        p_fit[pj] = QFI_ori(DE.freeHamiltonian, DE.Hamiltonian_derivative[1], rho_ini, DE.Liouville_operator, DE.γ, DE.control_Hamiltonian, 
                            DE.control_coefficients, DE.times)
    end

    f_ini = QFI_ori(DE.freeHamiltonian, DE.Hamiltonian_derivative[1], DE.psi*(DE.psi)', DE.Liouville_operator, DE.γ, DE.control_Hamiltonian, 
                    DE.control_coefficients, DE.times)
    f_list = [f_ini]
    println("initial QFI is $(f_ini)")
    Tend = (DE.times)[end]
    if save_file == true
        for i in 1:max_episodes
            p_fit = DE_train_QFI(populations, c, c0, c1, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            print("current QFI is ", maximum(p_fit), " ($i eposides)    \r")
            open("f_de_T$Tend.csv","a") do f
                writedlm(f, [maximum(p_fit)])
            end
            open("state_de_T$Tend.csv","w") do g
                writedlm(g, populations[indx].psi)
            end
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final QFI is ", maximum(p_fit))

    else
        for i in 1:max_episodes
            p_fit = DE_train_QFI(populations, c, c0, c1, p_num, dim, p_fit)
            print("current QFI is ", maximum(p_fit), " ($i eposides)    \r")
            append!(f_list,maximum(p_fit))
        end
        indx = findmax(p_fit)[2]
        open("f_de_T$Tend.csv","a") do f
            writedlm(f, [f_list])
        end
        open("state_de_T$Tend.csv","a") do g
            writedlm(g, populations[indx].psi)
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final QFI is ", maximum(p_fit))
    end
end

function DiffEvo_QFIM(DE::StateOpt_DE{T}, popsize, ini_population, c, c0, c1, seed, max_episodes, save_file) where {T<: Complex}
    println("state optimization")
    println("multiparameter scenario")
    println("control algorithm: DE")

    Random.seed!(seed)
    dim = length(DE.psi)

    p_num = popsize
    populations = repeat(DE, p_num)
    # initialize 
    for pj in 1:length(ini_population)
        populations[pj].psi = [ini_population[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_population)+1):(p_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        populations[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    p_fit = [0.0 for i in 1:p_num] 
    for pj in 1:p_num
        rho_ini = populations[pj].psi*(populations[pj].psi)'
        F_tp = QFIM_ori(DE.freeHamiltonian, DE.Hamiltonian_derivative, rho_ini, DE.Liouville_operator, DE.γ, DE.control_Hamiltonian, 
                            DE.control_coefficients, DE.times)
        p_fit[pj] = 1.0/real(tr(DE.W*pinv(F_tp)))
    end

    F = QFIM_ori(DE.freeHamiltonian, DE.Hamiltonian_derivative, DE.psi*(DE.psi)', DE.Liouville_operator, DE.γ, DE.control_Hamiltonian, 
                    DE.control_coefficients, DE.times)
    f_ini= real(tr(DE.W*pinv(F)))
    f_list = [f_ini]
    println("initial value of Tr(WF^{-1}) is $(f_ini)")
    Tend = (DE.times)[end]
    if save_file == true
        for i in 1:max_episodes
            p_fit = DE_train_QFIM(populations, c, c0, c1, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            open("f_de_T$Tend.csv","a") do f
                writedlm(f, [1.0/maximum(p_fit)])
            end
            open("state_de_T$Tend.csv","w") do g
                writedlm(g, populations[indx].psi)
            end
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))

    else
        for i in 1:max_episodes
            p_fit = DE_train_QFIM(populations, c, c0, c1, p_num, dim, p_fit)
            print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            append!(f_list,1.0/maximum(p_fit))
        end
        indx = findmax(p_fit)[2]
        open("f_de_T$Tend.csv","a") do f
            writedlm(f, [f_list])
        end
        open("state_de_T$Tend.csv","a") do g
            writedlm(g, populations[indx].psi)
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
    end
end

function DiffEvo_CFI(M, DE::StateOpt_DE{T}, popsize, ini_population, c, c0, c1, seed, max_episodes, save_file) where {T<: Complex}
    println("state optimization")
    println("single parameter scenario")
    println("control algorithm: DE")

    Random.seed!(seed)
    dim = length(DE.psi)

    p_num = popsize
    populations = repeat(DE, p_num)
    # initialize 
    for pj in 1:length(ini_population)
        populations[pj].psi = [ini_population[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_population)+1):(p_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        populations[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    p_fit = [0.0 for i in 1:p_num] 
    for pj in 1:p_num
        rho_ini = populations[pj].psi*(populations[pj].psi)'
        p_fit[pj] = CFI(M, DE.freeHamiltonian, DE.Hamiltonian_derivative[1], rho_ini, DE.Liouville_operator, DE.γ, DE.control_Hamiltonian, 
                            DE.control_coefficients, DE.times)
    end

    f_ini = CFI(M, DE.freeHamiltonian, DE.Hamiltonian_derivative[1], DE.psi*(DE.psi)', DE.Liouville_operator, DE.γ, DE.control_Hamiltonian, 
                    DE.control_coefficients, DE.times)
    f_list = [f_ini]
    println("initial CFI is $(f_ini)")
    Tend = (DE.times)[end]
    if save_file == true
        for i in 1:max_episodes
            p_fit = DE_train_CFI(M, populations, c, c0, c1, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            print("current CFI is ", maximum(p_fit), " ($i eposides)    \r")
            open("f_de_T$Tend.csv","a") do f
                writedlm(f, [maximum(p_fit)])
            end
            open("state_de_T$Tend.csv","w") do g
                writedlm(g, populations[indx].psi)
            end
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final CFI is ", maximum(p_fit))

    else
        for i in 1:max_episodes
            p_fit = DE_train_CFI(M, populations, c, c0, c1, p_num, dim, p_fit)
            print("current CFI is ", maximum(p_fit), " ($i eposides)    \r")
            append!(f_list,maximum(p_fit))
        end
        indx = findmax(p_fit)[2]
        open("f_de_T$Tend.csv","a") do f
            writedlm(f, [f_list])
        end
        open("state_de_T$Tend.csv","a") do g
            writedlm(g, populations[indx].psi)
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final CFI is ", maximum(p_fit))
    end
end

function DiffEvo_CFIM(M, DE::StateOpt_DE{T}, popsize, ini_population, c, c0, c1, seed, max_episodes, save_file) where {T<: Complex}
    println("state optimization")
    println("multiparameter scenario")
    println("control algorithm: DE")

    Random.seed!(seed)
    dim = length(DE.psi)

    p_num = popsize
    populations = repeat(DE, p_num)
    # initialize 
    for pj in 1:length(ini_population)
        populations[pj].psi = [ini_population[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_population)+1):(p_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        populations[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    p_fit = [0.0 for i in 1:p_num] 
    for pj in 1:p_num
        rho_ini = populations[pj].psi*(populations[pj].psi)'
        F_tp = CFIM(M, DE.freeHamiltonian, DE.Hamiltonian_derivative, rho_ini, DE.Liouville_operator, DE.γ, DE.control_Hamiltonian, 
                            DE.control_coefficients, DE.times)
        p_fit[pj] = 1.0/real(tr(DE.W*pinv(F_tp)))
    end

    F = CFIM(M, DE.freeHamiltonian, DE.Hamiltonian_derivative, DE.psi*(DE.psi)', DE.Liouville_operator, DE.γ, DE.control_Hamiltonian, 
                    DE.control_coefficients, DE.times)
    f_ini= real(tr(DE.W*pinv(F)))
    f_list = [f_ini]
    println("initial value of Tr(WF^{-1}) is $(f_ini)")
    Tend = (DE.times)[end]
    if save_file == true
        for i in 1:max_episodes
            p_fit = DE_train_CFIM(M, populations, c, c0, c1, p_num, dim, p_fit)
            indx = findmax(p_fit)[2]
            print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            open("f_de_T$Tend.csv","a") do f
                writedlm(f, [1.0/maximum(p_fit)])
            end
            open("state_de_T$Tend.csv","w") do g
                writedlm(g, populations[indx].psi)
            end
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))

    else
        for i in 1:max_episodes
            p_fit = DE_train_CFIM(M, populations, c, c0, c1, p_num, dim, p_fit)
            print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($i eposides)    \r")
            append!(f_list,1.0/maximum(p_fit))
        end
        indx = findmax(p_fit)[2]
        open("f_de_T$Tend.csv","a") do f
            writedlm(f, [f_list])
        end
        open("state_de_T$Tend.csv","a") do g
            writedlm(g, populations[indx].psi)
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
    end
end

function DE_train_QFI(populations, c, c0, c1, p_num, dim, p_fit)
    f_mean = p_fit |> mean
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace=false)
        ctrl_mut = zeros(ComplexF64, dim)
        for ci in 1:dim
            ctrl_mut[ci] = populations[mut_num[1]].psi[ci]+c*(populations[mut_num[2]].psi[ci]-populations[mut_num[3]].psi[ci])
        end
        #crossover
        if p_fit[pj] > f_mean
            cr = c0 + (c1-c0)*(p_fit[pj]-minimum(p_fit))/(maximum(p_fit)-minimum(p_fit))
        else
            cr = c0
        end
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
        f_cross = QFI_ori(populations[pj].freeHamiltonian, populations[pj].Hamiltonian_derivative[1], psi_cross*psi_cross', 
                          populations[pj].Liouville_operator, populations[pj].γ, populations[pj].control_Hamiltonian, 
                          populations[pj].control_coefficients, populations[pj].times)

        if f_cross > p_fit[pj]
            p_fit[pj] = f_cross
            for ck in 1:dim
                populations[pj].psi[ck] = psi_cross[ck]
            end
        end
    end
    return p_fit
end

function DE_train_QFIM(populations, c, c0, c1, p_num, dim, p_fit)
    f_mean = p_fit |> mean
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace=false)
        ctrl_mut = zeros(ComplexF64, dim)
        for ci in 1:dim
            ctrl_mut[ci] = populations[mut_num[1]].psi[ci]+c*(populations[mut_num[2]].psi[ci]-populations[mut_num[3]].psi[ci])
        end
        #crossover
        if p_fit[pj] > f_mean
            cr = c0 + (c1-c0)*(p_fit[pj]-minimum(p_fit))/(maximum(p_fit)-minimum(p_fit))
        else
            cr = c0
        end
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
        F_tp = QFIM_ori(populations[pj].freeHamiltonian, populations[pj].Hamiltonian_derivative, psi_cross*psi_cross', 
                          populations[pj].Liouville_operator, populations[pj].γ, populations[pj].control_Hamiltonian, 
                          populations[pj].control_coefficients, populations[pj].times)
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

function DE_train_CFI(M, populations, c, c0, c1, p_num, dim, p_fit)
    f_mean = p_fit |> mean
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace=false)
        ctrl_mut = zeros(ComplexF64, dim)
        for ci in 1:dim
            ctrl_mut[ci] = populations[mut_num[1]].psi[ci]+c*(populations[mut_num[2]].psi[ci]-populations[mut_num[3]].psi[ci])
        end
        #crossover
        if p_fit[pj] > f_mean
            cr = c0 + (c1-c0)*(p_fit[pj]-minimum(p_fit))/(maximum(p_fit)-minimum(p_fit))
        else
            cr = c0
        end
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
        f_cross = CFI(M, populations[pj].freeHamiltonian, populations[pj].Hamiltonian_derivative[1], psi_cross*psi_cross', 
                          populations[pj].Liouville_operator, populations[pj].γ, populations[pj].control_Hamiltonian, 
                          populations[pj].control_coefficients, populations[pj].times)

        if f_cross > p_fit[pj]
            p_fit[pj] = f_cross
            for ck in 1:dim
                populations[pj].psi[ck] = psi_cross[ck]
            end
        end
    end
    return p_fit
end

function DE_train_CFIM(M, populations, c, c0, c1, p_num, dim, p_fit)
    f_mean = p_fit |> mean
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace=false)
        ctrl_mut = zeros(ComplexF64, dim)
        for ci in 1:dim
            ctrl_mut[ci] = populations[mut_num[1]].psi[ci]+c*(populations[mut_num[2]].psi[ci]-populations[mut_num[3]].psi[ci])
        end
        #crossover
        if p_fit[pj] > f_mean
            cr = c0 + (c1-c0)*(p_fit[pj]-minimum(p_fit))/(maximum(p_fit)-minimum(p_fit))
        else
            cr = c0
        end
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
        F_tp = CFIM(M, populations[pj].freeHamiltonian, populations[pj].Hamiltonian_derivative, psi_cross*psi_cross', 
                          populations[pj].Liouville_operator, populations[pj].γ, populations[pj].control_Hamiltonian, 
                          populations[pj].control_coefficients, populations[pj].times)
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
