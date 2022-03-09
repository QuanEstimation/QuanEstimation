abstract type ControlSystem end
mutable struct GRAPE_Copt{T<:Complex,M <: Real} <: ControlSystem
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
    mt::M
    vt::M
    ϵ::M
    beta1::M
    beta2::M
    eps::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    GRAPE_Copt(freeHamiltonian, Hamiltonian_derivative::Vector{Matrix{T}}, ρ0::Matrix{T}, tspan::Vector{M}, decay_opt::Vector{Matrix{T}}, 
             γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{M}}, ctrl_bound::Vector{M}, 
             W::Matrix{M}, mt::M, vt::M, ϵ::M, beta1::M, beta2::M, eps::M, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1), ∂ρ_∂V=Vector{Vector{Matrix{T}}}(undef, 1)) where {T<:Complex,M <: Real}= 
             new{T,M}(freeHamiltonian, Hamiltonian_derivative, ρ0, tspan, decay_opt, γ, control_Hamiltonian, control_coefficients,
                      ctrl_bound, W, mt, vt, ϵ, beta1, beta2, eps, ρ, ∂ρ_∂x) 
end

###################### auto-GRAPE #########################
function QFIM_autoGRAPE_Copt(grape::GRAPE_Copt{T}, max_episode, Adam, save_file) where {T<:Complex}
    sym = Symbol("QFIM")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    M = [zeros(ComplexF64, size(grape.ρ0)[1], size(grape.ρ0)[1])]
    return info_autoGRAPE_QFIM(M, grape, max_episode, Adam, save_file, sym, str1, str2, str3)
end

function CFIM_autoGRAPE_Copt(M, grape::GRAPE_Copt{T}, max_episode, Adam, save_file) where {T<:Complex}
    sym = Symbol("CFIM")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_autoGRAPE_CFIM(M, grape, max_episode, Adam, save_file, sym, str1, str2, str3)
end

CFIM_autoGRAPE_Copt(grape::GRAPE_Copt, M, max_episode, Adam, save_file) = CFIM_autoGRAPE_Copt(M, grape, max_episode, Adam, save_file) 

function gradient_QFI!(grape::GRAPE_Copt{T}) where {T<:Complex}
    δF = gradient(x->QFI(grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan, grape.eps), grape.control_coefficients)[1].|>real
    grape.control_coefficients += grape.ϵ*δF
    bound!(grape.control_coefficients, grape.ctrl_bound)
    return δF
end

function gradient_QFI_Adam!(grape::GRAPE_Copt{T}) where {T<:Complex}
    δF = gradient(x->QFI(grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan, grape.eps), grape.control_coefficients)[1].|>real
    Adam!(grape, δF)
    bound!(grape.control_coefficients, grape.ctrl_bound)
    return δF
end

function gradient_QFIM!(grape::GRAPE_Copt{T}) where {T<:Complex}
    δF = gradient(x->1/(grape.W*pinv(QFIM(grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan, grape.eps), rtol=grape.eps) |> tr |>real), grape.control_coefficients).|>real |>sum
    grape.control_coefficients += grape.ϵ*δF
    bound!(grape.control_coefficients, grape.ctrl_bound)
    return δF
end

function gradient_QFIM_Adam!(grape::GRAPE_Copt{T}) where {T<:Complex}
    δF = gradient(x->1/(grape.W*pinv(QFIM(grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan, grape.eps), rtol=grape.eps) |> tr |>real), grape.control_coefficients).|>real |>sum
    Adam!(grape, δF)
    bound!(grape.control_coefficients, grape.ctrl_bound)
    return δF
end

function gradient_CFI!(grape::GRAPE_Copt{T}, M) where {T<:Complex}
    δI = gradient(x->CFI(M, grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan, grape.eps), grape.control_coefficients)[1].|>real
    grape.control_coefficients += grape.ϵ*δI
    bound!(grape.control_coefficients, grape.ctrl_bound)
    return δI
end

function gradient_CFI_Adam!(grape::GRAPE_Copt{T}, M) where {T<:Complex}
    δI = gradient(x->CFI(M, grape.freeHamiltonian, grape.Hamiltonian_derivative[1], grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan, grape.eps), grape.control_coefficients)[1].|>real
    Adam!(grape, δI)
    bound!(grape.control_coefficients, grape.ctrl_bound)
    return δI
end

function gradient_CFIM!(grape::GRAPE_Copt{T}, M) where {T<:Complex}
    δI = gradient(x->1/(grape.W*pinv(CFIM(M, grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan, grape.eps), rtol=grape.eps) |> tr |>real), grape.control_coefficients).|>real |>sum
    grape.control_coefficients += grape.ϵ*δI
    bound!(grape.control_coefficients, grape.ctrl_bound)
    return δI
end

function gradient_CFIM_Adam!(grape::GRAPE_Copt{T}, M) where {T<:Complex}
    δI = gradient(x->1/(grape.W*pinv(CFIM(M, grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ0, grape.decay_opt, grape.γ, grape.control_Hamiltonian, x, grape.tspan, grape.eps), rtol=grape.eps) |> tr |>real), grape.control_coefficients).|>real |>sum
    Adam!(grape, δI)
    bound!(grape.control_coefficients, grape.ctrl_bound)
    return δI
end

function info_autoGRAPE_QFIM(M, grape, max_episode, Adam, save_file, sym, str1, str2, str3)
    println("$str1 parameter estimation")
    ctrl_num = length(grape.control_Hamiltonian)
    ctrl_length = length(grape.control_coefficients[1])
    episodes = 1
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: auto-GRAPE")
        f_noctrl = obj_func(Val{sym}(), grape, M, [zeros(ctrl_length) for i in 1:ctrl_num])
        f_ini = obj_func(Val{sym}(), grape, M)

        f_list = [1.0/f_ini]
        println("non-controlled $str2 is $(1.0/f_noctrl)")
        println("initial $str2 is $(1.0/f_ini)")
        if save_file == true
            SaveFile_ctrl(1.0/f_ini, grape.control_coefficients)
            if Adam == true
                gradient_QFI_Adam!(grape)
                while true
                    f_now = obj_func(Val{sym}(), grape, M)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFI_Adam!(grape)
                end
            else
                gradient_QFI!(grape)
                while true
                    f_now = obj_func(Val{sym}(), grape, M)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFI!(grape)
                end
            end
        else
            if Adam == true
                gradient_QFI_Adam!(grape)
                while true
                    f_now = obj_func(Val{sym}(), grape, M)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, grape.control_coefficients)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFI_Adam!(grape)
                end
            else
                gradient_QFI!(grape)
                while true
                    f_now = obj_func(Val{sym}(), grape, M)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, grape.control_coefficients)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFI!(grape)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("control algorithm: auto-GRAPE")
        f_noctrl = obj_func(Val{sym}(), grape, M, [zeros(ctrl_length) for i in 1:ctrl_num])
        f_ini = obj_func(Val{sym}(), grape, M)

        f_list = [f_ini]
        println("non-controlled value of $str3 is $(f_noctrl)")
        println("initial value of $str3 is $(f_ini)")
        if save_file == true
            SaveFile_ctrl(f_ini, grape.control_coefficients)
            if Adam == true
                gradient_QFIM_Adam!(grape)
                while true
                    f_now = obj_func(Val{sym}(), grape, M)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        return 1/f_now, grape.control_coefficients
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(grape)
                end
            else
                gradient_QFIM!(grape)
                while true
                    f_now = obj_func(Val{sym}(), grape, M)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        return 1/f_now, grape.control_coefficients
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFIM!(grape)
                end
            end
        else
            if Adam == true
                gradient_QFIM_Adam!(grape)
                while true
                    f_now = obj_func(Val{sym}(), grape, M)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, grape.control_coefficients)
                        return 1/f_now, grape.control_coefficients
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(grape)
                end
            else
                gradient_QFIM!(grape)
                while true
                    f_now = obj_func(Val{sym}(), grape, M)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, grape.control_coefficients)
                        return 1/f_now, grape.control_coefficients
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFIM!(grape)
                end
            end
        end
    end
end

function info_autoGRAPE_CFIM(M, grape, max_episode, Adam, save_file, sym, str1, str2, str3)
    println("$str1 parameter estimation")
    ctrl_num = length(grape.control_Hamiltonian)
    ctrl_length = length(grape.control_coefficients[1])
    episodes = 1
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: auto-GRAPE")
        f_noctrl = obj_func(Val{sym}(), grape, M, [zeros(ctrl_length) for i in 1:ctrl_num])
        f_ini = obj_func(Val{sym}(), grape, M)

        f_list = [1.0/f_ini]
        println("non-controlled $str2 is $(1.0/f_noctrl)")
        println("initial $str2 is $(1.0/f_ini)")
        if save_file == true
            SaveFile_ctrl(f_ini, grape.control_coefficients)
            if Adam == true
                gradient_CFI_Adam!(grape, M)
                while true
                    f_now = obj_func(Val{sym}(), grape, M)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFI_Adam!(grape, M)
                end
            else
                gradient_CFI!(grape, M)
                while true
                    f_now = obj_func(Val{sym}(), grape, M)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFI!(grape, M)
                end
            end
        else
            if Adam == true
                gradient_CFI_Adam!(grape, M)
                while true
                    f_now = obj_func(Val{sym}(), grape, M)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, grape.control_coefficients)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFI_Adam!(grape, M)
                end
            else
                gradient_CFI!(grape, M)
                while true
                    f_now = obj_func(Val{sym}(), grape, M)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, grape.control_coefficients)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFI!(grape, M)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("control algorithm: auto-GRAPE")
        f_noctrl = obj_func(Val{sym}(), grape, M, [zeros(ctrl_length) for i in 1:ctrl_num])
        f_ini = obj_func(Val{sym}(), grape, M)

        f_list = [f_ini]
        println("non-controlled value of $str3 is $(f_noctrl)")
        println("initial value of $str3 is $(f_ini)")
        if save_file == true
            SaveFile_ctrl(f_ini, grape.control_coefficients)
            if Adam == true
                gradient_CFIM_Adam!(grape, M)
                while true
                    f_now = obj_func(Val{sym}(), grape, M)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        return 1/f_now, grape.control_coefficients
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(grape, M)
                end
            else
                gradient_CFIM!(grape, M)
                while true
                    f_now = obj_func(Val{sym}(), grape, M)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        return 1/f_now, grape.control_coefficients
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, grape.control_coefficients)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFIM!(grape, M)
                end
            end
        else
            if Adam == true
                gradient_CFIM_Adam!(grape, M)
                while true
                    f_now = obj_func(Val{sym}(), grape, M)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, grape.control_coefficients)
                        return 1/f_now, grape.control_coefficients
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(grape, M)
                end
            else
                gradient_CFIM!(grape, M)
                while true
                    f_now = obj_func(Val{sym}(), grape, M)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, grape.control_coefficients)
                        return 1/f_now, grape.control_coefficients
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_CFIM!(grape, M)
                end
            end
        end
    end
    return f_now, grape.control_coefficients
end


###################### GRAPE #########################
function QFIM_GRAPE_Copt(grape::GRAPE_Copt{T}, max_episode, Adam, save_file) where {T<:Complex}
    sym = Symbol("QFIM")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    M = [zeros(ComplexF64, size(grape.ρ0)[1], size(grape.ρ0)[1])]
    return info_GRAPE_QFIM(M, grape, max_episode, Adam, save_file, sym, str1, str2, str3)
end

function CFIM_GRAPE_Copt(M, grape::GRAPE_Copt{T}, max_episode, Adam, save_file) where {T<:Complex}
    sym = Symbol("CFIM")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_GRAPE_CFIM(M, grape, max_episode, Adam, save_file, sym, str1, str2, str3)
end

CFIM_GRAPE_Copt(grape::GRAPE_Copt, M, max_episode, Adam, save_file) = CFIM_GRAPE_Copt(M, grape, max_episode, Adam, save_file) 

function info_GRAPE_QFIM(M, grape, max_episode, Adam, save_file, sym, str1, str2, str3)
    println("$str1 parameter estimation")
    ctrl_num = length(grape.control_Hamiltonian)
    ctrl_length = length(grape.control_coefficients[1])
    episodes = 1
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: GRAPE")
        f_noctrl = obj_func(Val{sym}(), grape, M, [zeros(ctrl_length) for i in 1:ctrl_num])
        println("non-controlled $str2 is $(1.0/f_noctrl)")
        ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
        if Adam == true
            grape.control_coefficients, f_ini = gradient_QFIM_analy_Adam(grape)
        else
            grape.control_coefficients, f_ini = gradient_QFIM_analy(grape)
        end
        f_list = [f_ini]
        println("initial $str2 is $(f_ini)")
        if save_file == true
            SaveFile_ctrl(f_ini, ctrl_pre)
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy_Adam(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        SaveFile_ctrl(f_now, ctrl_pre) 
                        return f_now, grape.control_coefficients 
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, ctrl_pre)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end  
                end 
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        SaveFile_ctrl(f_now, ctrl_pre) 
                        return f_now, grape.control_coefficients 
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, ctrl_pre)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end 
            end                    
        else
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy_Adam(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, ctrl_pre)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, ctrl_pre)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            end
        end
    else
        println("multiparameter scenario")
        println("control algorithm: GRAPE")
        f_noctrl = obj_func(Val{sym}(), grape, M, [zeros(ctrl_length) for i in 1:ctrl_num])
        println("non-controlled value of $str3 is $(f_noctrl)")
        ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
        if Adam == true
            grape.control_coefficients, f_ini = gradient_QFIM_analy_Adam(grape)
        else
            grape.control_coefficients, f_ini = gradient_QFIM_analy(grape)
        end
        f_list = [f_ini]
        println("initial value of $str3 is $(f_ini)")
        if save_file == true
            SaveFile_ctrl(f_ini, ctrl_pre)
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy_Adam(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        SaveFile_ctrl(f_now, ctrl_pre)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, ctrl_pre)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        SaveFile_ctrl(f_now, ctrl_pre)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, ctrl_pre)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy_Adam(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, ctrl_pre)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_QFIM_analy(grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, ctrl_pre)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            end
        end
    end
    return f_now, grape.control_coefficients
end

function info_GRAPE_CFIM(M, grape, max_episode, Adam, save_file, sym, str1, str2, str3)
    println("$str1 parameter estimation")
    ctrl_num = length(grape.control_Hamiltonian)
    ctrl_length = length(grape.control_coefficients[1])
    episodes = 1
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: GRAPE")
        f_noctrl = obj_func(Val{sym}(), grape, M, [zeros(ctrl_length) for i in 1:ctrl_num])
        println("non-controlled $str2 is $(1.0/f_noctrl)")
        ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
        if Adam == true
            grape.control_coefficients, f_ini = gradient_CFIM_analy_Adam(M, grape)
        else
            grape.control_coefficients, f_ini = gradient_CFIM_analy(M, grape)
        end
        f_list = [f_ini]
        println("initial $str2 is $(f_ini)")
        if save_file == true
            SaveFile_ctrl(f_ini, ctrl_pre)
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy_Adam(M, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        SaveFile_ctrl(f_now, ctrl_pre) 
                        return f_now, grape.control_coefficients 
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, ctrl_pre)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end  
                end 
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy(M, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        SaveFile_ctrl(f_now, ctrl_pre) 
                        return f_now, grape.control_coefficients 
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, ctrl_pre)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end 
            end                    
        else
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy_Adam(M, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, ctrl_pre)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy(M, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, ctrl_pre)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            end
        end
    else
        println("multiparameter scenario")
        println("control algorithm: GRAPE")
        f_noctrl = obj_func(Val{sym}(), grape, M, [zeros(ctrl_length) for i in 1:ctrl_num])
        println("non-controlled value of $str3 is $(f_noctrl)")
        ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
        if Adam == true
            grape.control_coefficients, f_ini = gradient_CFIM_analy_Adam(M, grape)
        else
            grape.control_coefficients, f_ini = gradient_CFIM_analy(M, grape)
        end
        f_list = [f_ini]
        println("initial value of $str3 is $(f_ini)")
        if save_file == true
            SaveFile_ctrl(f_ini, ctrl_pre)
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy_Adam(M, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        SaveFile_ctrl(f_now, ctrl_pre)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, ctrl_pre)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy(M, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        SaveFile_ctrl(f_now, ctrl_pre)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        SaveFile_ctrl(f_now, ctrl_pre)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy_Adam(M, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, ctrl_pre)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            else
                while true
                    ctrl_pre = [[grape.control_coefficients[i][j] for j in 1:ctrl_length] for i in 1:ctrl_num]
                    grape.control_coefficients, f_now = gradient_CFIM_analy(M, grape)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_ctrl(f_list, ctrl_pre)
                        return f_now, grape.control_coefficients
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                end
            end
        end
    end
    return f_now, grape.control_coefficients
end

function dynamics_analy(grape::GRAPE_Copt{T}, dim, tnum, para_num, ctrl_num) where {T<:Complex}
    Δt = grape.tspan[2] - grape.tspan[1]
    H = Htot(grape.freeHamiltonian, grape.control_Hamiltonian, grape.control_coefficients)

    ρt = [Vector{ComplexF64}(undef, dim^2) for i in 1:tnum]
    ∂ρt_∂x = [[Vector{ComplexF64}(undef, dim^2) for para in 1:para_num] for i in 1:tnum]
    δρt_δV = [[] for ctrl in 1:ctrl_num]
    ∂xδρt_δV = [[[] for ctrl in 1:ctrl_num] for i in 1:para_num]
    ∂H_L = [Matrix{ComplexF64}(undef, dim^2,dim^2) for i in 1:para_num]
    Hc_L = [Matrix{ComplexF64}(undef, dim^2,dim^2) for i in 1:ctrl_num]

    ρt[1] = grape.ρ0 |> vec
    for cj in 1:ctrl_num
        Hc_L[cj] = liouville_commu(grape.control_Hamiltonian[cj])
        append!(δρt_δV[cj], [-im*Δt*Hc_L[cj]*ρt[1]])
    end

    for pj in 1:para_num
        ∂ρt_∂x[1][pj] = ρt[1] |> zero
        ∂H_L[pj] = liouville_commu(grape.Hamiltonian_derivative[pj])
        for ci in 1:ctrl_num
            append!(∂xδρt_δV[pj][ci], [-im*Δt*Hc_L[ci]*∂ρt_∂x[1][pj]])
        end
    end

    for ti in 2:tnum
        
        expL = evolute(H[ti-1], grape.decay_opt, grape.γ, Δt, ti)
        ρt[ti] = expL * ρt[ti-1]
        for pk in 1:para_num
            ∂ρt_∂x[ti][pk] = -im * Δt * ∂H_L[pk] * ρt[ti] + expL * ∂ρt_∂x[ti-1][pk]
        end
        
        for ck in 1:ctrl_num
            for tk in 1:(ti-1)
                δρt_δV_first = popfirst!(δρt_δV[ck])
                δρt_δV_tp = expL * δρt_δV_first
                append!(δρt_δV[ck], [δρt_δV_tp])
                for pk in 1:para_num
                    ∂xδρt_δV_first = popfirst!(∂xδρt_δV[pk][ck])
                    ∂xδρt_δV_tp = -im * Δt * ∂H_L[pk] * expL * δρt_δV_first + expL * ∂xδρt_δV_first
                    append!(∂xδρt_δV[pk][ck], [∂xδρt_δV_tp])
                end
            end
            δρt_δV_last = -im * Δt * Hc_L[ck] * ρt[ti]
            append!(δρt_δV[ck], [δρt_δV_last])
            for pk in 1:para_num
                ∂xδρt_δV_last = -im * Δt * Hc_L[ck] * ∂ρt_∂x[ti][pk]
                append!(∂xδρt_δV[pk][ck], [∂xδρt_δV_last])
            end
        end
    end

    ρt_T = ρt[end] |> vec2mat
    ∂ρt_T = [(∂ρt_∂x[end][para] |> vec2mat) for para in 1:para_num]

    return ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV
end

function gradient_QFIM_analy_Adam(grape::GRAPE_Copt{T}) where {T<:Complex}
    dim = size(grape.ρ0)[1]
    tnum = length(grape.tspan)
    para_num = length(grape.Hamiltonian_derivative)
    ctrl_num = length(grape.control_Hamiltonian)
    
    ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV = dynamics_analy(grape, dim, tnum, para_num, ctrl_num)

    Lx = SLD(ρt_T, ∂ρt_T, grape.eps)
    F_T = QFIM(ρt_T, ∂ρt_T, grape.eps)

    if para_num == 1
        cost_function = F_T[1]
        anti_commu = 2*Lx[1]*Lx[1]
        for cm in 1:ctrl_num
            mt = grape.mt
            vt = grape.vt
            for tm in 1:(tnum-1)
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                ∂xδρt_T_δV = ∂xδρt_δV[1][cm][tm] |> vec2mat
                term1 = tr(∂xδρt_T_δV*Lx[1])
                term2 = tr(∂ρt_T_δV*anti_commu)
                δF = ((2*term1-0.5*term2) |> real)
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.eps)
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)

    elseif para_num == 2
        coeff1 = real(det(F))
        coeff2 = grape.W[1,1]*F_T[2,2]+grape.W[2,2]*F_T[1,1]-grape.W[1,2]*F_T[2,1]-grape.W[2,1]*F_T[1,2]
        cost_function = (abs(det(F_T)) < grape.eps ? (1.0/grape.eps) : real(tr(grape.W*inv(F_T))))
        for cm in 1:ctrl_num
            mt = grape.mt
            vt = grape.vt
            for tm in 1:(tnum-1)
                δF_all = [[0.0 for i in 1:para_num] for j in 1:para_num]
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                for pm in 1:para_num
                    for pn in 1:para_num
                        ∂xδρt_T_δV_a = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                        ∂xδρt_T_δV_b = ∂xδρt_δV[pn][cm][tm] |> vec2mat
                        term1 = tr(∂xδρt_T_δV_a * Lx[pn])
                        term2 = tr(∂xδρt_T_δV_b * Lx[pm])

                        anti_commu = Lx[pm] * Lx[pn] + Lx[pn] * Lx[pm]
                        term2 = tr(∂ρt_T_δV * anti_commu)
                        δF_all[pm][pn] = ((2*term1-0.5*term2) |> real)
                    end
                end
                item1 = -coeff2*(F_T[2,2]*δF_all[1][1]+F_T[1,1]*δF_all[2][2]-F_T[2,1]*δF_all[1][2]-F_T[1,2]*δF_all[2][1])/coeff1^2
                item2 = (grape.W[1,1]*δF_all[2][2]+grape.W[2,2]*δF_all[1][1]-grape.W[1,2]*δF_all[2][1]-grape.W[2,1]*δF_all[1][2])/coeff1
                δF = -(item1+item2)*cost_function^2
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.eps)
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)
    else       
        cost_function = (abs(det(F_T)) < grape.eps ? (1.0/grape.eps) : real(tr(grape.W*inv(F_T))))
        coeff = [grape.W[para,para]/F_T[para,para] for para in 1:para_num] |>sum
        coeff = coeff^(-2)
        for cm in 1:ctrl_num
            mt = grape.mt
            vt = grape.vt
            for tm in 1:(tnum-1)
                δF = 0.0
                for pm in 1:para_num
                    ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                    ∂xδρt_T_δV = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                    term1 = tr(∂xδρt_T_δV * Lx[pm])
                    anti_commu = 2 * Lx[pm] * Lx[pm]
                    term2 = tr(∂ρt_T_δV * anti_commu)
                    δF = δF + grape.W[pm,pm]*(1.0/F_T[pm,pm]/F_T[pm,pm])*((2*term1-0.5*term2) |> real)
                end
                δF = δF*coeff
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.eps)
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)
    end
    grape.control_coefficients, cost_function
end

function gradient_QFIM_analy(grape::GRAPE_Copt{T}) where {T<:Complex}
    dim = size(grape.ρ0)[1]
    tnum = length(grape.tspan)
    para_num = length(grape.Hamiltonian_derivative)
    ctrl_num = length(grape.control_Hamiltonian)
    
    ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV = dynamics_analy(grape, dim, tnum, para_num, ctrl_num)

    Lx = SLD(ρt_T, ∂ρt_T, grape.eps)
    F_T = QFIM(ρt_T, ∂ρt_T, grape.eps)

    cost_function = F_T[1]
    
    if para_num == 1
        anti_commu = 2*Lx[1]*Lx[1]
        for cm in 1:ctrl_num
            mt = grape.mt
            vt = grape.vt
            for tm in 1:(tnum-1)
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                ∂xδρt_T_δV = ∂xδρt_δV[1][cm][tm] |> vec2mat
                term1 = tr(∂xδρt_T_δV*Lx[1])
                term2 = tr(∂ρt_T_δV*anti_commu)
                δF = ((2*term1-0.5*term2) |> real)
                grape.control_coefficients[cm][tm] = grape.control_coefficients[cm][tm] + grape.ϵ*δF
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)

    elseif para_num == 2
        coeff1 = real(det(F))
        coeff2 = grape.W[1,1]*F_T[2,2]+grape.W[2,2]*F_T[1,1]-grape.W[1,2]*F_T[2,1]-grape.W[2,1]*F_T[1,2]
        cost_function = (abs(det(F_T)) < grape.eps ? (1.0/grape.eps) : real(tr(grape.W*inv(F_T))))
        for cm in 1:ctrl_num
            for tm in 1:(tnum-1)
                δF_all = [[0.0 for i in 1:para_num] for j in 1:para_num]
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                for pm in 1:para_num
                    for pn in 1:para_num
                        ∂xδρt_T_δV_a = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                        ∂xδρt_T_δV_b = ∂xδρt_δV[pn][cm][tm] |> vec2mat
                        term1 = tr(∂xδρt_T_δV_a * Lx[pn])
                        term2 = tr(∂xδρt_T_δV_b * Lx[pm])

                        anti_commu = Lx[pm] * Lx[pn] + Lx[pn] * Lx[pm]
                        term2 = tr(∂ρt_T_δV * anti_commu)
                        δF_all[pm][pn] = ((2*term1-0.5*term2) |> real)
                    end
                end
                item1 = -coeff2*(F_T[2,2]*δF_all[1][1]+F_T[1,1]*δF_all[2][2]-F_T[2,1]*δF_all[1][2]-F_T[1,2]*δF_all[2][1])/coeff1^2
                item2 = (grape.W[1,1]*δF_all[2][2]+grape.W[2,2]*δF_all[1][1]-grape.W[1,2]*δF_all[2][1]-grape.W[2,1]*δF_all[1][2])/coeff1
                δF = -(item1+item2)*cost_function^2
                grape.control_coefficients[cm][tm] = grape.control_coefficients[cm][tm] + grape.ϵ*δF
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)

    else
        cost_function = (abs(det(F_T)) < grape.eps ? (1.0/grape.eps) : real(tr(grape.W*inv(F_T))))
        coeff = [grape.W[para,para]/F_T[para,para] for para in 1:para_num] |>sum
        coeff = coeff^(-2)
        for cm in 1:ctrl_num
            for tm in 1:(tnum-1)
                δF = 0.0
                for pm in 1:para_num
                    ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                    ∂xδρt_T_δV = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                    term1 = tr(∂xδρt_T_δV * Lx[pm])
                    anti_commu = 2 * Lx[pm] * Lx[pm]
                    term2 = tr(∂ρt_T_δV * anti_commu)
                    δF = δF + grape.W[pm,pm]*(1.0/F_T[pm,pm]/F_T[pm,pm])*((2*term1-0.5*term2) |> real)
                end
                δF = δF*coeff
                grape.control_coefficients[cm][tm] = grape.control_coefficients[cm][tm] + grape.ϵ*δF
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)
    end
    grape.control_coefficients, cost_function
end

function gradient_CFIM_analy_Adam(M::Vector{Matrix{T}}, grape::GRAPE_Copt{T}) where {T<:Complex}
    dim = size(grape.ρ0)[1]
    tnum = length(grape.tspan)
    para_num = length(grape.Hamiltonian_derivative)
    ctrl_num = length(grape.control_Hamiltonian)
    
    ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV = dynamics_analy(grape, dim, tnum, para_num, ctrl_num)

    if para_num == 1
        F_T = CFI(ρt_T, ∂ρt_T[1], M, grape.eps)
        cost_function = F_T
        L1_tidle = zeros(ComplexF64, dim, dim)
        L2_tidle = zeros(ComplexF64, dim, dim)

        for mi in 1:dim
            p = (tr(ρt_T*M[mi]) |> real)
            dp = (tr(∂ρt_T[1]*M[mi]) |> real)
            if p > grape.eps
                L1_tidle = L1_tidle + dp*M[mi]/p
                L2_tidle = L2_tidle + dp*dp*M[mi]/p^2
            end
        end

        for cm in 1:ctrl_num
            mt = grape.mt
            vt = grape.vt
            for tm in 1:(tnum-1)
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                ∂xδρt_T_δV = ∂xδρt_δV[1][cm][tm] |> vec2mat
                term1 = tr(∂xδρt_T_δV*L1_tidle)
                term2 = tr(∂ρt_T_δV*L2_tidle)
                δF = ((2*term1-term2) |> real)
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.eps)
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)

    elseif para_num == 2
        F_T = CFIM(ρt_T, ∂ρt_T, M, grape.eps)
        L1_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
        L2_tidle = [[zeros(ComplexF64, dim, dim) for i in 1:para_num] for j in 1:para_num]
    
        for para_i in 1:para_num
            for mi in 1:dim
                p = (tr(ρt_T*M[mi]) |> real)
                dp = (tr(∂ρt_T[para_i]*M[mi]) |> real)
                if p > grape.eps
                    L1_tidle[para_i] = L1_tidle[para_i] + dp*M[mi]/p
                end
            end
        end
    
        for para_i in 1:para_num
            dp_a = (tr(∂ρt_T[para_i]*M[mi]) |> real)
            for para_j in 1:para_num
                dp_b = (tr(∂ρt_T[para_j]*M[mi]) |> real)
                for mi in 1:dim
                    p = (tr(ρt_T*M[mi]) |> real)
                    if p > grape.eps
                        L2_tidle[para_i][para_j] = L2_tidle[para_i][para_j] + dp_a*dp_b*M[mi]/p^2
                    end
                end
            end
        end
        coeff1 = real(det(F))
        coeff2 = grape.W[1,1]*F_T[2,2]+grape.W[2,2]*F_T[1,1]-grape.W[1,2]*F_T[2,1]-grape.W[2,1]*F_T[1,2]
        cost_function = (abs(det(F_T)) < grape.eps ? (1.0/grape.eps) : real(tr(grape.W*inv(F_T))))
        for cm in 1:ctrl_num
            mt = grape.mt
            vt = grape.vt
            for tm in 1:(tnum-1)
                δF_all = [[0.0 for i in 1:para_num] for j in 1:para_num]
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                for pm in 1:para_num
                    for pn in 1:para_num
                        ∂xδρt_T_δV_a = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                        ∂xδρt_T_δV_b = ∂xδρt_δV[pn][cm][tm] |> vec2mat
                        term1 = tr(∂xδρt_T_δV_a * L1_tidle[pn])
                        term2 = tr(∂xδρt_T_δV_b * L1_tidle[pm])
                        term3 = tr(∂ρt_T_δV * L2_tidle[pm][pn])
                        δF_all[pm][pn] = ((term1+term2-term3) |> real)
                    end
                end
                item1 = -coeff2*(F_T[2,2]*δF_all[1][1]+F_T[1,1]*δF_all[2][2]-F_T[2,1]*δF_all[1][2]-F_T[1,2]*δF_all[2][1])/coeff1^2
                item2 = (grape.W[1,1]*δF_all[2][2]+grape.W[2,2]*δF_all[1][1]-grape.W[1,2]*δF_all[2][1]-grape.W[2,1]*δF_all[1][2])/coeff1
                δF = -(item1+item2)*cost_function^2
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.eps)
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)

    else
        F_T = CFIM(ρt_T, ∂ρt_T, M, grape.eps)
        L1_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
        L2_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
    
        for para_i in 1:para_num
            for mi in 1:dim
                p = (tr(ρt_T*M[mi]) |> real)
                dp = (tr(∂ρt_T[para_i]*M[mi]) |> real)
                if p > grape.eps
                    L1_tidle[para_i] = L1_tidle[para_i] + dp*M[mi]/p
                    L2_tidle[para_i] = L2_tidle[para_i] + dp*dp*M[mi]/p^2
                end
            end
        end

        cost_function = (abs(det(F_T)) < grape.eps ? (1.0/grape.eps) : real(tr(grape.W*inv(F_T))))
        coeff = [grape.W[para,para]/F_T[para,para] for para in 1:para_num] |>sum
        coeff = coeff^(-2)
        for cm in 1:ctrl_num
            mt = grape.mt
            vt = grape.vt
            for tm in 1:(tnum-1)
                δF = 0.0
                for pm in 1:para_num
                    ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                    ∂xδρt_T_δV = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                    term1 = tr(∂xδρt_T_δV * L1_tidle[pm])
                    term2 = tr(∂ρt_T_δV * L2_tidle[pm])
                    δF = δF + grape.W[pm,pm]*(1.0/F_T[pm,pm]/F_T[pm,pm])*((2*term1-term2) |> real)
                end
                δF = δF*coeff
                grape.control_coefficients[cm][tm], mt, vt = Adam(δF, tm, grape.control_coefficients[cm][tm], mt, vt, grape.ϵ, grape.beta1, grape.beta2, grape.eps)
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)
    end
    grape.control_coefficients, cost_function
end

function gradient_CFIM_analy(M::Vector{Matrix{T}}, grape::GRAPE_Copt{T}) where {T<:Complex}
    dim = size(grape.ρ0)[1]
    tnum = length(grape.tspan)
    para_num = length(grape.Hamiltonian_derivative)
    ctrl_num = length(grape.control_Hamiltonian)
    
    ρt_T, ∂ρt_T, δρt_δV, ∂xδρt_δV = dynamics_analy(grape, dim, tnum, para_num, ctrl_num)

    if para_num == 1
        F_T = CFI(ρt_T, ∂ρt_T[1], M, grape.eps)
        cost_function = F_T
        L1_tidle = zeros(ComplexF64, dim, dim)
        L2_tidle = zeros(ComplexF64, dim, dim)

        for mi in 1:dim
            p = (tr(ρt_T*M[mi]) |> real)
            dp = (tr(∂ρt_T[1]*M[mi]) |> real)
            if p > grape.eps
                L1_tidle = L1_tidle + dp*M[mi]/p
                L2_tidle = L2_tidle + dp*dp*M[mi]/p^2
            end
        end

        for cm in 1:ctrl_num
            for tm in 1:(tnum-1)
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                ∂xδρt_T_δV = ∂xδρt_δV[1][cm][tm] |> vec2mat
                term1 = tr(∂xδρt_T_δV*L1_tidle)
                term2 = tr(∂ρt_T_δV*L2_tidle)
                δF = ((2*term1-term2) |> real)
                grape.control_coefficients[cm][tm] = grape.control_coefficients[cm][tm] + grape.ϵ*δF
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)

    elseif para_num == 2
        F_T = CFIM(ρt_T, ∂ρt_T, M, grape.eps)
        L1_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
        L2_tidle = [[zeros(ComplexF64, dim, dim) for i in 1:para_num] for j in 1:para_num]
    
        for para_i in 1:para_num
            for mi in 1:dim
                p = (tr(ρt_T*M[mi]) |> real)
                dp = (tr(∂ρt_T[para_i]*M[mi]) |> real)
                if p > grape.eps
                    L1_tidle[para_i] = L1_tidle[para_i] + dp*M[mi]/p
                end
            end
        end
    
        for para_i in 1:para_num
            dp_a = (tr(∂ρt_T[para_i]*M[mi]) |> real)
            for para_j in 1:para_num
                dp_b = (tr(∂ρt_T[para_j]*M[mi]) |> real)
                for mi in 1:dim
                    p = (tr(ρt_T*M[mi]) |> real)
                    if p > grape.eps
                        L2_tidle[para_i][para_j] = L2_tidle[para_i][para_j] + dp_a*dp_b*M[mi]/p^2
                    end
                end
            end
        end
        coeff1 = real(det(F))
        coeff2 = grape.W[1,1]*F_T[2,2]+grape.W[2,2]*F_T[1,1]-grape.W[1,2]*F_T[2,1]-grape.W[2,1]*F_T[1,2]
        cost_function = (abs(det(F_T)) < grape.eps ? (1.0/grape.eps) : real(tr(grape.W*inv(F_T))))
        for cm in 1:ctrl_num
            for tm in 1:(tnum-1)
                δF_all = [[0.0 for i in 1:para_num] for j in 1:para_num]
                ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                for pm in 1:para_num
                    for pn in 1:para_num
                        ∂xδρt_T_δV_a = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                        ∂xδρt_T_δV_b = ∂xδρt_δV[pn][cm][tm] |> vec2mat
                        term1 = tr(∂xδρt_T_δV_a * L1_tidle[pn])
                        term2 = tr(∂xδρt_T_δV_b * L1_tidle[pm])
                        term3 = tr(∂ρt_T_δV * L2_tidle[pm][pn])
                        δF_all[pm][pn] = ((term1+term2-term3) |> real)
                    end
                end
                item1 = -coeff2*(F_T[2,2]*δF_all[1][1]+F_T[1,1]*δF_all[2][2]-F_T[2,1]*δF_all[1][2]-F_T[1,2]*δF_all[2][1])/coeff1^2
                item2 = (grape.W[1,1]*δF_all[2][2]+grape.W[2,2]*δF_all[1][1]-grape.W[1,2]*δF_all[2][1]-grape.W[2,1]*δF_all[1][2])/coeff1
                δF = -(item1+item2)*cost_function^2
                grape.control_coefficients[cm][tm] = grape.control_coefficients[cm][tm] + grape.ϵ*δF
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)

    else
        F_T = CFIM(ρt_T, ∂ρt_T, M, grape.eps)
        L1_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
        L2_tidle = [zeros(ComplexF64, dim, dim) for i in 1:para_num]
    
        for para_i in 1:para_num
            for mi in 1:dim
                p = (tr(ρt_T*M[mi]) |> real)
                dp = (tr(∂ρt_T[para_i]*M[mi]) |> real)
                if p > grape.eps
                    L1_tidle[para_i] = L1_tidle[para_i] + dp*M[mi]/p
                    L2_tidle[para_i] = L2_tidle[para_i] + dp*dp*M[mi]/p^2
                end
            end
        end

        cost_function = (abs(det(F_T)) < grape.eps ? (1.0/grape.eps) : real(tr(grape.W*inv(F_T))))
        coeff = [grape.W[para,para]/F_T[para,para] for para in 1:para_num] |>sum
        coeff = coeff^(-2)
        for cm in 1:ctrl_num
            for tm in 1:(tnum-1)
                δF = 0.0
                for pm in 1:para_num
                    ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                    ∂xδρt_T_δV = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                    term1 = tr(∂xδρt_T_δV * L1_tidle[pm])
                    term2 = tr(∂ρt_T_δV * L2_tidle[pm])
                    δF = δF + grape.W[pm,pm]*(1.0/F_T[pm,pm]/F_T[pm,pm])*((2*term1-term2) |> real)
                end
                δF = δF*coeff
                grape.control_coefficients[cm][tm] = grape.control_coefficients[cm][tm] + grape.ϵ*δF
            end
        end
        bound!(grape.control_coefficients, grape.ctrl_bound)
    end
    grape.control_coefficients, cost_function
end


