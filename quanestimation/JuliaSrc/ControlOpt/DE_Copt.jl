mutable struct DE_Copt{T <: Complex,M <: Real} <: ControlSystem
    freeHamiltonian
    Hamiltonian_derivative::Vector{Matrix{T}}
    ρ0::Matrix{T}
    tspan::Vector{M}
    decay_opt::Vector{Matrix{T}}
    γ::Vector{M}
    control_Hamiltonian::Vector{Matrix{T}}
    control_coefficients::Vector{Vector{M}}
    ctrl_bound::Vector{M}
    W::Matrix{M}
    accuracy::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    DE_Copt(freeHamiltonian, Hamiltonian_derivative::Vector{Matrix{T}}, ρ0::Matrix{T},
             tspan::Vector{M}, decay_opt::Vector{Matrix{T}},γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}},
             control_coefficients::Vector{Vector{M}}, ctrl_bound::Vector{M}, W::Matrix{M}, accuracy::M, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = new{T,M}(freeHamiltonian, 
                Hamiltonian_derivative, ρ0, tspan, decay_opt, γ, control_Hamiltonian, control_coefficients, ctrl_bound, W, accuracy, ρ, ∂ρ_∂x) 
end

function QFIM_DE_Copt(DE::DE_Copt{T}, popsize, ini_population, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("QFIM")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    Measurement = [zeros(ComplexF64, size(DE.ρ0)[1], size(DE.ρ0)[1])]
    return info_DE_Copt(Measurement, DE, popsize, ini_population, c, cr, seed, max_episode, save_file, sym, str1, str2, str3)
end

function CFIM_DE_Copt(Measurement, DE::DE_Copt{T}, popsize, ini_population, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_DE_Copt(Measurement, DE, popsize, ini_population, c, cr, seed, max_episode, save_file, sym, str1, str2, str3)
end

function HCRB_DE_Copt(DE::DE_Copt{T}, popsize, ini_population, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("HCRB")
    str1 = ""
    str2 = "HCRB"
    str3 = "HCRB"
    Measurement = [zeros(ComplexF64, size(DE.ρ0)[1], size(DE.ρ0)[1])]
    if length(DE.Hamiltonian_derivative) == 1
        println("In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the objection function for control optimization.")
        return nothing
    else
        return info_DE_Copt(Measurement, DE, popsize, ini_population, c, cr, seed, max_episode, save_file, sym, str1, str2, str3)
    end
end

function info_DE_Copt(Measurement, DE, popsize, ini_population, c, cr, seed, max_episode, save_file, sym, str1, str2, str3) where {T<:Complex}
    println("$str1 parameter estimation")
    Random.seed!(seed)
    ctrl_num = length(DE.control_Hamiltonian)
    ctrl_length = length(DE.control_coefficients[1])

    p_num = popsize
    populations = repeat(DE, p_num)
    # initialize
    if length(ini_population) > popsize
        ini_population = [ini_population[i] for i in 1:popsize]
    end
    for pj in 1:length(ini_population)
        populations[pj].control_coefficients = [[ini_population[pj][i,j] for j in 1:ctrl_length] for i in 1:ctrl_num]
    end
    if DE.ctrl_bound[1] == -Inf || DE.ctrl_bound[2] == Inf
        for pj in (length(ini_population)+1):p_num
            populations[pj].control_coefficients = [[2*rand()-1.0 for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
    else
        a = DE.ctrl_bound[1]
        b = DE.ctrl_bound[2]
        for pj in (length(ini_population)+1):p_num
            populations[pj].control_coefficients = [[(b-a)*rand()+a for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
    end

    p_fit = [1.0/obj_func(Val{sym}(), populations[i], Measurement) for i in 1:p_num]
    f_noctrl = obj_func(Val{sym}(), DE, Measurement, [zeros(ctrl_length) for i in 1:ctrl_num])
    f_noctrl = 1.0/f_noctrl

    f_ini = p_fit[1]
    
    if length(DE.Hamiltonian_derivative) == 1 
        println("single parameter scenario")
        println("control algorithm: Differential Evolution (DE)")
        println("non-controlled $str2 is $(f_noctrl)")
        println("initial $str2 is $(f_ini)")
    
        f_list = [1.0/f_ini]
        if save_file == true
            for i in 1:(max_episode-1)
                p_fit = DE_train_Copt(Measurement, populations, c, cr, p_num, ctrl_num, ctrl_length, p_fit, sym)
                indx = findmax(p_fit)[2]
                append!(f_list, maximum(p_fit))
                print("current $str2 is ", maximum(p_fit), " ($i episodes)    \r")
                SaveFile_ctrl(f_list, populations[indx].control_coefficients)
            end
            p_fit = DE_train_Copt(Measurement, populations, c, cr, p_num, ctrl_num, ctrl_length, p_fit, sym)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_ctrl(f_list, populations[indx].control_coefficients)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str2 is ", maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = DE_train_Copt(Measurement, populations, c, cr, p_num, ctrl_num, ctrl_length, p_fit, sym)
                print("current $str2 is ", maximum(p_fit), " ($i episodes)    \r")
                append!(f_list, maximum(p_fit))
            end
            p_fit = DE_train_Copt(Measurement, populations, c, cr, p_num, ctrl_num, ctrl_length, p_fit, sym)
            indx = findmax(p_fit)[2]
            append!(f_list, maximum(p_fit))
            SaveFile_ctrl(f_list, populations[indx].control_coefficients)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final $str2 is ", maximum(p_fit))
        end
    else
        println("multiparameter scenario")
        println("control algorithm: Differential Evolution (DE)")
        println("non-controlled value of $str3 is $(1.0/f_noctrl)")
        println("initial value of $str3 is $(1.0/f_ini)")
    
        f_list = [1.0/f_ini]
        if save_file == true
            for i in 1:(max_episode-1)
                p_fit = DE_train_Copt(Measurement, populations, c, cr, p_num, ctrl_num, ctrl_length, p_fit, sym)
                indx = findmax(p_fit)[2]
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_ctrl(f_list, populations[indx].control_coefficients)
                print("current value of $str3 is ", 1.0/maximum(p_fit), " ($i episodes)    \r")
            end
            p_fit = DE_train_Copt(Measurement, populations, c, cr, p_num, ctrl_num, ctrl_length, p_fit, sym)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_ctrl(f_list, populations[indx].control_coefficients)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is ", 1.0/maximum(p_fit))
        else
            for i in 1:(max_episode-1)
                p_fit = DE_train_Copt(Measurement, populations, c, cr, p_num, ctrl_num, ctrl_length, p_fit, sym)
                append!(f_list, 1.0/maximum(p_fit))
                print("current value of $str3 is ", 1.0/maximum(p_fit), " ($i episodes)    \r")
            end
            p_fit = DE_train_Copt(Measurement, populations, c, cr, p_num, ctrl_num, ctrl_length, p_fit, sym)
            indx = findmax(p_fit)[2]
            append!(f_list, 1.0/maximum(p_fit))
            SaveFile_ctrl(f_list, populations[indx].control_coefficients)
            print("\e[2K")
            println("Iteration over, data saved.")
            println("Final value of $str3 is ", 1.0/maximum(p_fit))
        end
    end
end

function DE_train_Copt(Measurement, populations, c, cr, p_num, ctrl_num, ctrl_length, p_fit, sym)
    for pj in 1:p_num
        #mutations
        mut_num = sample(1:p_num, 3, replace=false)
        ctrl_mut = [Vector{Float64}(undef, ctrl_length) for i in 1:ctrl_num]
        for ci in 1:ctrl_num
            for ti in 1:ctrl_length
                ctrl_mut[ci][ti] = populations[mut_num[1]].control_coefficients[ci][ti]+
                                   c*(populations[mut_num[2]].control_coefficients[ci][ti]-
                                   populations[mut_num[3]].control_coefficients[ci][ti])
            end
        end
        #crossover
        ctrl_cross = [Vector{Float64}(undef, ctrl_length) for i in 1:ctrl_num]
        for cj in 1:ctrl_num
            cross_int = sample(1:ctrl_length, 1, replace=false)[1]
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
        bound!(ctrl_cross, populations[pj].ctrl_bound)

        f_cross = obj_func(Val{sym}(), populations[pj], Measurement, ctrl_cross)
        f_cross = 1.0/f_cross

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
