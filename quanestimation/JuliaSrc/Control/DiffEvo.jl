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

function DiffEvo_QFI(DE::DiffEvo{T}, populations, c, c0, c1, seed, max_episodes, save_file) where {T<: Complex}
    println("Differential Evolution:")
    println("single parameter scenario")

    Random.seed!(seed)
    ctrl_num = length(DE.control_Hamiltonian)
    ctrl_length = length(DE.control_coefficients[1])

    p_num = populations
    populations = repeat(DE, p_num)
    # initialize
    for pi in 1:p_num
        populations[pi].control_coefficients = [0.1*rand(ctrl_length)  for i in 1:ctrl_num]
    end

    p_fit = [QFI(populations[i]) for i in 1:p_num]
    f_ini = maximum(p_fit)
    f_list = [f_ini]
    
    Tend = (DE.times)[end] |> Int
    if save_file == true
        for i in 1:max_episodes
            p_fit = DE_train_QFI(populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit)
            indx = findmax(p_fit)[2]
            print("current QFI is ", maximum(p_fit), " ($i epochs)    \r")
            open("f_T$Tend.csv","a") do f
                writedlm(f, [maximum(p_fit)])
            end
            open("ctrl_T$Tend.csv","w") do g
                writedlm(g, populations[indx].control_coefficients)
            end
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final QFI is ", maximum(p_fit))

    else
        for i in 1:max_episodes
            p_fit = DE_train_QFI(populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit)
            print("current QFI is ", maximum(p_fit), " ($i epochs)    \r")
            append!(f_list,maximum(p_fit))
        end
        indx = findmax(p_fit)[2]
        open("f_T$Tend.csv","a") do f
            writedlm(f, [f_list])
        end
        open("ctrl_T$Tend.csv","a") do g
            writedlm(g, populations[indx].control_coefficients)
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final QFI is ", maximum(p_fit))
    end
end

function DiffEvo_QFIM(DE::DiffEvo{T}, populations, c, c0, c1, seed, max_episodes, save_file) where {T<: Complex}
    println("Differential Evolution:")
    println("multiparameter scenario")

    Random.seed!(seed)
    ctrl_num = length(DE.control_Hamiltonian)
    ctrl_length = length(DE.control_coefficients[1])

    p_num = populations
    populations = repeat(DE, p_num)
    # initialize
    for pi in 1:p_num
        populations[pi].control_coefficients = [rand(ctrl_length)  for i in 1:ctrl_num]
    end

    p_fit = [1.0/real(tr(pinv(QFIM(populations[i])))) for i in 1:p_num]

    f_ini = maximum(p_fit)
    f_list = [f_ini]
    
    Tend = (DE.times)[end] |> Int
    if save_file == true
        for i in 1:max_episodes
            F = DE_train_QFIM(populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit)
            indx = findmax(p_fit)[2]
            print("current target value is ", 1.0/maximum(p_fit), " ($i epochs)    \r")
            open("f_T$Tend.csv","a") do f
                writedlm(f, [1.0/maximum(p_fit)])
            end
            open("ctrl_T$Tend.csv","w") do g
                writedlm(g, populations[indx].control_coefficients)
            end
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final target value is ", 1.0/maximum(p_fit))

    else
        for i in 1:max_episodes
            p_fit = DE_train_QFIM(populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit)
            print("current target value is ", 1.0/maximum(p_fit), " ($i epochs)    \r")
            append!(f_list,1.0/maximum(p_fit))
        end
        indx = findmax(p_fit)[2]
        open("f_T$Tend.csv","a") do f
            writedlm(f, [f_list])
        end
        open("ctrl_T$Tend.csv","a") do g
            writedlm(g, populations[indx].control_coefficients)
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final target value is ", 1.0/maximum(p_fit))
    end
end

function DE_train_QFI(populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit)
    f_mean = p_fit |> mean
    for pi in 1:p_num
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
        if p_fit[pi] > f_mean
            cr = c0 + (c1-c0)*(p_fit[pi]-minimum(p_fit))/(maximum(p_fit)-minimum(p_fit))
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
                    ctrl_cross[cj][tj] = populations[pi].control_coefficients[cj][tj]
                end
            end
            ctrl_cross[cj][cross_int] = ctrl_mut[cj][cross_int]
        end
        #selection
        f_cross = QFI(populations[pi].freeHamiltonian, populations[pi].Hamiltonian_derivative[1], populations[pi].ρ_initial, 
                      populations[pi].Liouville_operator, populations[pi].γ, populations[pi].control_Hamiltonian, 
                      ctrl_cross, populations[pi].times)

        if f_cross > p_fit[pi]
            p_fit[pi] = f_cross
            for ck in 1:ctrl_num
                for tk in 1:ctrl_length
                    populations[pi].control_coefficients[ck][tk] = ctrl_cross[ck][tk]
                end
            end
        end
    end
    return p_fit
end

function DE_train_QFIM(populations, c, c0, c1, p_num, ctrl_num, ctrl_length, p_fit)
    f_mean = p_fit |> mean
    for pi in 1:p_num
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
        if p_fit[pi] > f_mean
            cr = c0 + (c1-c0)*(p_fit[pi]-minimum(p_fit))/(maximum(p_fit)-minimum(p_fit))
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
                    ctrl_cross[cj][tj] = populations[pi].control_coefficients[cj][tj]
                end
            end
            ctrl_cross[cj][cross_int] = ctrl_mut[cj][cross_int]
        end
        #selection
        F = QFIM(populations[pi].freeHamiltonian, populations[pi].Hamiltonian_derivative, populations[pi].ρ_initial, 
                      populations[pi].Liouville_operator, populations[pi].γ, populations[pi].control_Hamiltonian, 
                      ctrl_cross, populations[pi].times)
        f_cross = 1.0/real(tr(pinv(F)))

        if f_cross > p_fit[pi]
            p_fit[pi] = f_cross
            for ck in 1:ctrl_num
                for tk in 1:ctrl_length
                    populations[pi].control_coefficients[ck][tk] = ctrl_cross[ck][tk]
                end
            end
        end
    end
    return p_fit
end
