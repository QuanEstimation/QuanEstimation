mutable struct DiffEvo{T <: Complex,M <: Real} <: ControlSystem
    freeHamiltonian::Matrix{T}
    Hamiltonian_derivative::Vector{Matrix{T}}
    ρ_initial::Matrix{T}
    times::Vector{M}
    Liouville_operator::Vector{Matrix{T}}
    γ::Vector{M}
    control_Hamiltonian::Vector{Matrix{T}}
    control_coefficients::Vector{Vector{M}}
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    DiffEvo(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, ρ_initial::Matrix{T},
             times::Vector{M}, Liouville_operator::Vector{Matrix{T}},γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}},
             control_coefficients::Vector{Vector{M}}, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = new{T,M}(freeHamiltonian, 
                Hamiltonian_derivative, ρ_initial, times, Liouville_operator, γ, control_Hamiltonian, control_coefficients, ρ, ∂ρ_∂x) 
end

function DiffEvo_QFI(DE::DiffEvo{T}, populations, ctrl_max, c, c0, c1, seed, max_episodes, save_file) where {T<: Complex}
    println("quantum parameter estimation")
    println("single parameter scenario")
    println("control algorithm: DE")

    Random.seed!(seed)
    ctrl_num = length(DE.control_Hamiltonian)
    ctrl_length = length(DE.control_coefficients[1])

    p_num = populations
    populations = repeat(DE, p_num)
    # initialize
    for pj in 1:p_num
        populations[pj].control_coefficients = [ctrl_max*rand(ctrl_length)  for i in 1:ctrl_num]
    end

    p_fit = [QFI_ori(populations[i]) for i in 1:p_num]
    f_ini = maximum(p_fit)
    f_list = [f_ini]
    println("initial QFI is $(f_ini)")
    
    Tend = (DE.times)[end]
    if save_file == true
        for i in 1:max_episodes
            p_fit = DE_train_QFI(populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit, ctrl_max)
            indx = findmax(p_fit)[2]
            print("current QFI is ", maximum(p_fit), " ($i episodes)    \r")
            open("f_de_T$Tend.csv","a") do f
                writedlm(f, [maximum(p_fit)])
            end
            open("ctrl_de_T$Tend.csv","w") do g
                writedlm(g, populations[indx].control_coefficients)
            end
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final QFI is ", maximum(p_fit))

    else
        for i in 1:max_episodes
            p_fit = DE_train_QFI(populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit, ctrl_max)
            print("current QFI is ", maximum(p_fit), " ($i episodes)    \r")
            append!(f_list,maximum(p_fit))
        end
        indx = findmax(p_fit)[2]
        open("f_de_T$Tend.csv","a") do f
            writedlm(f, [f_list])
        end
        open("ctrl_de_T$Tend.csv","a") do g
            writedlm(g, populations[indx].control_coefficients)
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final QFI is ", maximum(p_fit))
    end
end

function DiffEvo_QFIM(DE::DiffEvo{T}, populations, ctrl_max, c, c0, c1, seed, max_episodes, save_file) where {T<: Complex}
    println("quantum parameter estimation")
    println("multiparameter scenario")
    println("control algorithm: DE")

    Random.seed!(seed)
    ctrl_num = length(DE.control_Hamiltonian)
    ctrl_length = length(DE.control_coefficients[1])

    p_num = populations
    populations = repeat(DE, p_num)
    # initialize
    for pj in 1:p_num
        populations[pj].control_coefficients = [ctrl_max*rand(ctrl_length)  for i in 1:ctrl_num]
    end

    p_fit = [1.0/real(tr(pinv(QFIM_ori(populations[i])))) for i in 1:p_num]

    f_ini = maximum(p_fit)
    f_list = [1.0/f_ini]
    println("initial value of Tr(WF^{-1}) is $(1.0/f_ini)")
    
    Tend = (DE.times)[end]
    if save_file == true
        for i in 1:max_episodes
            F = DE_train_QFIM(populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit, ctrl_max)
            indx = findmax(p_fit)[2]
            print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($i episodes)    \r")
            open("f_de_T$Tend.csv","a") do f
                writedlm(f, [1.0/maximum(p_fit)])
            end
            open("ctrl_de_T$Tend.csv","w") do g
                writedlm(g, populations[indx].control_coefficients)
            end
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))

    else
        for i in 1:max_episodes
            p_fit = DE_train_QFIM(populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit, ctrl_max)
            print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($i episodes)    \r")
            append!(f_list, 1.0/maximum(p_fit))
        end
        indx = findmax(p_fit)[2]
        open("f_de_T$Tend.csv","a") do f
            writedlm(f, [f_list])
        end
        open("ctrl_de_T$Tend.csv","a") do g
            writedlm(g, populations[indx].control_coefficients)
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
    end
end

function DiffEvo_CFI(M, DE::DiffEvo{T}, populations, ctrl_max, c, c0, c1, seed, max_episodes, save_file) where {T<: Complex}
    println("classical parameter estimation")
    println("single parameter scenario")
    println("control algorithm: DE")

    Random.seed!(seed)
    ctrl_num = length(DE.control_Hamiltonian)
    ctrl_length = length(DE.control_coefficients[1])

    p_num = populations
    populations = repeat(DE, p_num)
    # initialize
    for pj in 1:p_num
        populations[pj].control_coefficients = [ctrl_max*rand(ctrl_length)  for i in 1:ctrl_num]
    end

    p_fit = [CFI(M,populations[i]) for i in 1:p_num]
    f_ini = maximum(p_fit)
    f_list = [f_ini]
    println("initial CFI is $(f_ini)")
    
    Tend = (DE.times)[end]
    if save_file == true
        for i in 1:max_episodes
            p_fit = DE_train_CFI(M, populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit, ctrl_max)
            indx = findmax(p_fit)[2]
            print("current CFI is ", maximum(p_fit), " ($i episodes)    \r")
            open("f_de_T$Tend.csv","a") do f
                writedlm(f, [maximum(p_fit)])
            end
            open("ctrl_de_T$Tend.csv","w") do g
                writedlm(g, populations[indx].control_coefficients)
            end
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final CFI is ", maximum(p_fit))

    else
        for i in 1:max_episodes
            p_fit = DE_train_CFI(M, populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit, ctrl_max)
            print("current CFI is ", maximum(p_fit), " ($i episodes)    \r")
            append!(f_list,maximum(p_fit))
        end
        indx = findmax(p_fit)[2]
        open("f_de_T$Tend.csv","a") do f
            writedlm(f, [f_list])
        end
        open("ctrl_de_T$Tend.csv","a") do g
            writedlm(g, populations[indx].control_coefficients)
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final CFI is ", maximum(p_fit))
    end
end

function DiffEvo_CFIM(M, DE::DiffEvo{T}, populations, ctrl_max, c, c0, c1, seed, max_episodes, save_file) where {T<: Complex}
    println("classical parameter estimation")
    println("multiparameter scenario")
    println("control algorithm: DE")

    Random.seed!(seed)
    ctrl_num = length(DE.control_Hamiltonian)
    ctrl_length = length(DE.control_coefficients[1])

    p_num = populations
    populations = repeat(DE, p_num)
    # initialize
    for pj in 1:p_num
        populations[pj].control_coefficients = [ctrl_max*rand(ctrl_length)  for i in 1:ctrl_num]
    end

    p_fit = [1.0/real(tr(pinv(CFIM(M, populations[i])))) for i in 1:p_num]

    f_ini = maximum(p_fit)
    f_list = [1.0/f_ini]
    println("initial value of Tr(WF^{-1}) is $(1.0/f_ini)")
    
    Tend = (DE.times)[end]
    if save_file == true
        for i in 1:max_episodes
            F = DE_train_CFIM(M, populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit, ctrl_max)
            indx = findmax(p_fit)[2]
            print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($i episodes)    \r")
            open("f_de_T$Tend.csv","a") do f
                writedlm(f, [1.0/maximum(p_fit)])
            end
            open("ctrl_de_T$Tend.csv","w") do g
                writedlm(g, populations[indx].control_coefficients)
            end
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))

    else
        for i in 1:max_episodes
            p_fit = DE_train_CFIM(M, populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit, ctrl_max)
            print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($i episodes)    \r")
            append!(f_list, 1.0/maximum(p_fit))
        end
        indx = findmax(p_fit)[2]
        open("f_de_T$Tend.csv","a") do f
            writedlm(f, [f_list])
        end
        open("ctrl_de_T$Tend.csv","a") do g
            writedlm(g, populations[indx].control_coefficients)
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
    end
end

function DE_train_QFI(populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit, ctrl_max)
    f_mean = p_fit |> mean
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace=false)
        ctrl_mut = [Vector{Float64}(undef, ctrl_length)  for i in 1:ctrl_num]
        for ci in 1:ctrl_num
            for ti in 1:ctrl_length
                ctrl_mut[ci][ti] = populations[mut_num[1]].control_coefficients[ci][ti]+
                                   c*(populations[mut_num[2]].control_coefficients[ci][ti]-
                                   populations[mut_num[3]].control_coefficients[ci][ti])
            end
        end
        #crossover
        if p_fit[pj] > f_mean
            cr = c0 + (c1-c0)*(p_fit[pj]-minimum(p_fit))/(maximum(p_fit)-minimum(p_fit))
        else
            cr = c0
        end
        ctrl_cross = [Vector{Float64}(undef, ctrl_length)  for i in 1:ctrl_num]
        for cj in 1:ctrl_num
            cross_int = sample(1:ctrl_length, 1, replace=false)
            for tj in 1:ctrl_length
                rand_num = rand()
                if rand_num <= cr
                    ctrl_cross[cj][tj] = ctrl_mut[cj][tj]
                else
                    ctrl_cross[cj][tj] = populations[pj].control_coefficients[cj][tj]
                end
            end
            ctrl_cross[cj][cross_int] = ctrl_mut[cj][cross_int]
        end
        #selection
        for ck in 1:ctrl_num
            for tk in 1:ctrl_length
                ctrl_cross[ck][tk] = (x-> (x|>abs) < ctrl_max ? x : ctrl_max)(ctrl_cross[ck][tk])
            end
        end
        f_cross = QFI_ori(populations[pj].freeHamiltonian, populations[pj].Hamiltonian_derivative[1], populations[pj].ρ_initial, 
                      populations[pj].Liouville_operator, populations[pj].γ, populations[pj].control_Hamiltonian, 
                      ctrl_cross, populations[pj].times)

        if f_cross > p_fit[pj]
            p_fit[pj] = f_cross
            for ck in 1:ctrl_num
                for tk in 1:ctrl_length
                    populations[pj].control_coefficients[ck][tk] = ctrl_cross[ck][tk]
                end
            end
        end
    end
    return p_fit
end

function DE_train_QFIM(populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit, ctrl_max)
    f_mean = p_fit |> mean
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace=false)
        ctrl_mut = [Vector{Float64}(undef, ctrl_length)  for i in 1:ctrl_num]
        for ci in 1:ctrl_num
            for ti in 1:ctrl_length
                ctrl_mut[ci][ti] = populations[mut_num[1]].control_coefficients[ci][ti]+
                                   c*(populations[mut_num[2]].control_coefficients[ci][ti]-
                                   populations[mut_num[3]].control_coefficients[ci][ti])
            end
        end
        #crossover
        if p_fit[pj] > f_mean
            cr = c0 + (c1-c0)*(p_fit[pj]-minimum(p_fit))/(maximum(p_fit)-minimum(p_fit))
        else
            cr = c0
        end
        ctrl_cross = [Vector{Float64}(undef, ctrl_length)  for i in 1:ctrl_num]
        for cj in 1:ctrl_num
            cross_int = sample(1:ctrl_length, 1, replace=false)
            for tj in 1:ctrl_length
                rand_num = rand()
                if rand_num <= cr
                    ctrl_cross[cj][tj] = ctrl_mut[cj][tj]
                else
                    ctrl_cross[cj][tj] = populations[pj].control_coefficients[cj][tj]
                end
            end
            ctrl_cross[cj][cross_int] = ctrl_mut[cj][cross_int]
        end
        #selection
        for ck in 1:ctrl_num
            for tk in 1:ctrl_length
                ctrl_cross[ck][tk] = (x-> (x|>abs) < ctrl_max ? x : ctrl_max)(ctrl_cross[ck][tk])
            end
        end

        F = QFIM_ori(populations[pj].freeHamiltonian, populations[pj].Hamiltonian_derivative, populations[pj].ρ_initial, 
                      populations[pj].Liouville_operator, populations[pj].γ, populations[pj].control_Hamiltonian, 
                      ctrl_cross, populations[pj].times)
        f_cross = 1.0/real(tr(pinv(F)))

        if f_cross > p_fit[pj]
            p_fit[pj] = f_cross
            for ck in 1:ctrl_num
                for tk in 1:ctrl_length
                    populations[pj].control_coefficients[ck][tk] = ctrl_cross[ck][tk]
                end
            end
        end
    end
    return p_fit
end

function DE_train_CFI(M, populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit, ctrl_max)
    f_mean = p_fit |> mean
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace=false)
        ctrl_mut = [Vector{Float64}(undef, ctrl_length)  for i in 1:ctrl_num]
        for ci in 1:ctrl_num
            for ti in 1:ctrl_length
                ctrl_mut[ci][ti] = populations[mut_num[1]].control_coefficients[ci][ti]+
                                   c*(populations[mut_num[2]].control_coefficients[ci][ti]-
                                   populations[mut_num[3]].control_coefficients[ci][ti])
            end
        end
        #crossover
        if p_fit[pj] > f_mean
            cr = c0 + (c1-c0)*(p_fit[pj]-minimum(p_fit))/(maximum(p_fit)-minimum(p_fit))
        else
            cr = c0
        end
        ctrl_cross = [Vector{Float64}(undef, ctrl_length)  for i in 1:ctrl_num]
        for cj in 1:ctrl_num
            cross_int = sample(1:ctrl_length, 1, replace=false)
            for tj in 1:ctrl_length
                rand_num = rand()
                if rand_num <= cr
                    ctrl_cross[cj][tj] = ctrl_mut[cj][tj]
                else
                    ctrl_cross[cj][tj] = populations[pj].control_coefficients[cj][tj]
                end
            end
            ctrl_cross[cj][cross_int] = ctrl_mut[cj][cross_int]
        end
        #selection
        for ck in 1:ctrl_num
            for tk in 1:ctrl_length
                ctrl_cross[ck][tk] = (x-> (x|>abs) < ctrl_max ? x : ctrl_max)(ctrl_cross[ck][tk])
            end
        end
        f_cross = CFI(M, populations[pj].freeHamiltonian, populations[pj].Hamiltonian_derivative[1], populations[pj].ρ_initial, 
                      populations[pj].Liouville_operator, populations[pj].γ, populations[pj].control_Hamiltonian, 
                      ctrl_cross, populations[pj].times)

        if f_cross > p_fit[pj]
            p_fit[pj] = f_cross
            for ck in 1:ctrl_num
                for tk in 1:ctrl_length
                    populations[pj].control_coefficients[ck][tk] = ctrl_cross[ck][tk]
                end
            end
        end
    end
    return p_fit
end

function DE_train_CFIM(M, populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit, ctrl_max)
    f_mean = p_fit |> mean
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace=false)
        ctrl_mut = [Vector{Float64}(undef, ctrl_length)  for i in 1:ctrl_num]
        for ci in 1:ctrl_num
            for ti in 1:ctrl_length
                ctrl_mut[ci][ti] = populations[mut_num[1]].control_coefficients[ci][ti]+
                                   c*(populations[mut_num[2]].control_coefficients[ci][ti]-
                                   populations[mut_num[3]].control_coefficients[ci][ti])
            end
        end
        #crossover
        if p_fit[pj] > f_mean
            cr = c0 + (c1-c0)*(p_fit[pj]-minimum(p_fit))/(maximum(p_fit)-minimum(p_fit))
        else
            cr = c0
        end
        ctrl_cross = [Vector{Float64}(undef, ctrl_length)  for i in 1:ctrl_num]
        for cj in 1:ctrl_num
            cross_int = sample(1:ctrl_length, 1, replace=false)
            for tj in 1:ctrl_length
                rand_num = rand()
                if rand_num <= cr
                    ctrl_cross[cj][tj] = ctrl_mut[cj][tj]
                else
                    ctrl_cross[cj][tj] = populations[pj].control_coefficients[cj][tj]
                end
            end
            ctrl_cross[cj][cross_int] = ctrl_mut[cj][cross_int]
        end
        #selection
        for ck in 1:ctrl_num
            for tk in 1:ctrl_length
                ctrl_cross[ck][tk] = (x-> (x|>abs) < ctrl_max ? x : ctrl_max)(ctrl_cross[ck][tk])
            end
        end
        F = CFIM(populations[pj].freeHamiltonian, populations[pj].Hamiltonian_derivative, populations[pj].ρ_initial, 
                      populations[pj].Liouville_operator, populations[pj].γ, populations[pj].control_Hamiltonian, 
                      ctrl_cross, populations[pj].times)
        f_cross = 1.0/real(tr(pinv(F)))

        if f_cross > p_fit[pj]
            p_fit[pj] = f_cross
            for ck in 1:ctrl_num
                for tk in 1:ctrl_length
                    populations[pj].control_coefficients[ck][tk] = ctrl_cross[ck][tk]
                end
            end
        end
    end
    return p_fit
end

