############# time-independent Hamiltonian (noiseless) ################
function QFIM_AD_Sopt(AD::TimeIndepend_noiseless{T}, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file) where {T<:Complex}
    sym = Symbol("QFIM_TimeIndepend_noiseless")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    Measurement = [zeros(ComplexF64, length(AD.psi), length(AD.psi))]
    return info_AD_noiseless(Measurement, AD, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file, sym, str1, str2, str3)
end

function CFIM_AD_Sopt(Measurement, AD::TimeIndepend_noiseless{T}, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file) where {T<:Complex}
    sym = Symbol("CFIM_TimeIndepend_noiseless")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_AD_noiseless(Measurement, AD, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file, sym, str1, str2, str3)
end

function gradient_QFI!(AD::TimeIndepend_noiseless{T}, epsilon) where {T <: Complex}
    δF = gradient(x->QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x, AD.tspan, AD.accuracy), AD.psi)[1]
    AD.psi += epsilon*δF
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFI_Adam!(AD::TimeIndepend_noiseless{T}, epsilon, mt, vt, beta1, beta2, accuracy) where {T <: Complex}
    δF = gradient(x->QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x, AD.tspan, AD.accuracy), AD.psi)[1]
    StateOpt_Adam!(AD, δF, epsilon, mt, vt, beta1, beta2, accuracy) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFIM!(AD::TimeIndepend_noiseless{T}, epsilon) where {T <: Complex}
    δF = gradient(x->1/(AD.W*pinv(QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, x, AD.tspan, AD.accuracy), rtol=AD.accuracy) |> tr |>real), AD.psi) |>sum
    AD.psi += epsilon*δF
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFIM_Adam!(AD::TimeIndepend_noiseless{T}, epsilon, mt, vt, beta1, beta2, accuracy) where {T <: Complex}
    δF = gradient(x->1/(AD.W*pinv(QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, x, AD.tspan, AD.accuracy), rtol=AD.accuracy) |> tr |>real), AD.psi) |>sum
    StateOpt_Adam!(AD, δF, epsilon, mt, vt, beta1, beta2, accuracy) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFI!(AD::TimeIndepend_noiseless{T}, Measurement, epsilon) where {T <: Complex}
    δI = gradient(x->CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x, AD.tspan, AD.accuracy), AD.psi)[1]
    AD.psi += epsilon*δI
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFI_Adam!(AD::TimeIndepend_noiseless{T}, Measurement, epsilon, mt, vt, beta1, beta2) where {T <: Complex}
    δI = gradient(x->CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x, AD.tspan, AD.accuracy), AD.psi)[1]
    StateOpt_Adam!(AD, δI, epsilon, mt, vt, beta1, beta2, AD.accuracy) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFIM!(AD::TimeIndepend_noiseless{T}, Measurement, epsilon) where {T <: Complex}
    δI = gradient(x->1/(AD.W*pinv(CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x, AD.tspan, AD.accuracy), rtol=AD.accuracy) |> tr |>real), AD.psi) |>sum
    AD.psi += epsilon*δI
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFIM_Adam!(AD::TimeIndepend_noiseless{T}, Measurement, epsilon, mt, vt, beta1, beta2) where {T <: Complex}
    δI = gradient(x->1/(AD.W*pinv(CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x, AD.tspan, AD.accuracy), rtol=AD.accuracy) |> tr |>real), AD.psi) |>sum
    StateOpt_Adam!(AD, δI, epsilon, mt, vt, beta1, beta2, AD.accuracy) 
    AD.psi = AD.psi/norm(AD.psi)
end

function info_AD_noiseless(Measurement, AD::TimeIndepend_noiseless{T}, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file, sym, str1, str2, str3) where {T <: Complex}
    println("state optimization")
    episodes = 1
    dim = length(AD.psi)
    if length(AD.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        f_ini = obj_func(Val{sym}(), AD, Measurement)
        f_list = [1.0/f_ini]
        println("initial $str2 is $(1.0/f_ini)")
        if Adam == true
            gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
        else
            gradient_QFI!(AD, epsilon)
        end
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                while true
                    f_now = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                end
            else
                while true
                    f_now = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI!(AD, epsilon)
                end
            end
        else
            if Adam == true
                while true
                    f_now = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                end
            else
                while true
                    f_now = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI!(AD, epsilon)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        f_ini = obj_func(Val{sym}(), AD, Measurement)
        f_list = [f_ini]
        println("initial value of $str3 is $(f_ini)")
        if Adam == true
            gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
        else
            gradient_QFIM!(AD, epsilon)
        end
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                while true
                    f_now = obj_func(Val{sym}(), AD, Measurement)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current value of $str3 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                end
            else
                while true
                    f_now = obj_func(Val{sym}(), AD, Measurement)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current value of $str3 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM!(AD, epsilon)
                end
            end
        else
            if Adam == true
                while true
                    f_now = obj_func(Val{sym}(), AD, Measurement)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str3 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                end
            else
                while true
                    f_now = obj_func(Val{sym}(), AD, Measurement)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str3 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM!(AD, epsilon)
                end
            end
        end
    end
end

############# time-independent Hamiltonian (noise) ################
function QFIM_AD_Sopt(AD::TimeIndepend_noise{T}, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file) where {T<:Complex}
    sym = Symbol("QFIM_TimeIndepend_noise")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    Measurement = [zeros(ComplexF64, length(AD.psi), length(AD.psi))]
    return info_AD_noise(Measurement, AD, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file, sym, str1, str2, str3)
end

function CFIM_AD_Sopt(Measurement, AD::TimeIndepend_noise{T}, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file) where {T<:Complex}
    sym = Symbol("CFIM_TimeIndepend_noise")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_AD_noise(Measurement, AD, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file, sym, str1, str2, str3)
end

function gradient_QFI!(AD::TimeIndepend_noise{T}, epsilon) where {T <: Complex}
    δF = gradient(x->QFIM_TimeIndepend_AD(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x*x', AD.decay_opt, AD.γ, AD.tspan), AD.psi)[1]
    AD.psi += epsilon*δF
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFI_Adam!(AD::TimeIndepend_noise{T}, epsilon, mt, vt, beta1, beta2, accuracy) where {T <: Complex}
    δF = gradient(x->QFIM_TimeIndepend_AD(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x*x', AD.decay_opt, AD.γ, AD.tspan), AD.psi)[1]
    StateOpt_Adam!(AD, δF, epsilon, mt, vt, beta1, beta2, accuracy) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFIM!(AD::TimeIndepend_noise{T}, epsilon) where {T <: Complex}
    δF = gradient(x->1/(AD.W*pinv(QFIM_TimeIndepend_AD(AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.decay_opt, AD.γ, AD.tspan), rtol=AD.accuracy) |> tr |>real), AD.psi) |>sum
    AD.psi += epsilon*δF
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFIM_Adam!(AD::TimeIndepend_noise{T}, epsilon, mt, vt, beta1, beta2, accuracy) where {T <: Complex}
    δF = gradient(x->1/(AD.W*pinv(QFIM_TimeIndepend_AD(AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.decay_opt, AD.γ, AD.tspan), rtol=AD.accuracy) |> tr |>real), AD.psi) |>sum
    StateOpt_Adam!(AD, δF, epsilon, mt, vt, beta1, beta2, accuracy) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFI!(AD::TimeIndepend_noise{T}, Measurement, epsilon) where {T <: Complex}
    δI = gradient(x->CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x*x', AD.decay_opt, AD.γ, AD.tspan, AD.accuracy), AD.psi)[1]
    AD.psi += epsilon*δI
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFI_Adam!(AD::TimeIndepend_noise{T}, Measurement, epsilon, mt, vt, beta1, beta2) where {T <: Complex}
    δI = gradient(x->CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x*x', AD.decay_opt, AD.γ, AD.tspan, AD.accuracy), AD.psi)[1]
    StateOpt_Adam!(AD, δI, epsilon, mt, vt, beta1, beta2, AD.accuracy) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFIM!(AD::TimeIndepend_noise{T}, Measurement, epsilon) where {T <: Complex}
    δI = gradient(x->1/(AD.W*pinv(CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.decay_opt, AD.γ, AD.tspan, AD.accuracy), rtol=AD.accuracy) |> tr |>real), AD.psi) |>sum
    AD.psi += epsilon*δI
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFIM_Adam!(AD::TimeIndepend_noise{T}, Measurement, epsilon, mt, vt, beta1, beta2) where {T <: Complex}
    δI = gradient(x->1/(AD.W*pinv(CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.decay_opt, AD.γ, AD.tspan, AD.accuracy), rtol=AD.accuracy) |> tr |>real), AD.psi) |>sum
    StateOpt_Adam!(AD, δI, epsilon, mt, vt, beta1, beta2, AD.accuracy) 
    AD.psi = AD.psi/norm(AD.psi)
end

function info_AD_noise(Measurement, AD::TimeIndepend_noise{T}, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file, sym, str1, str2, str3) where {T <: Complex}
    println("$str1 state optimization")
    episodes = 1
    dim = length(AD.psi)
    if length(AD.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        f_ini = obj_func(Val{sym}(), AD, Measurement)
        f_list = [1.0/f_ini]
        println("initial $str2 is $(1.0/f_ini)")
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                while true
                    f_now = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                end
            else
                gradient_QFI!(AD, epsilon)
                while true
                    f_now = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI!(AD, epsilon)
                end
            end
        else
            if Adam == true
                gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                while true
                    f_now = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                end
            else
                gradient_QFI!(AD, epsilon)
                while true
                    f_now = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI!(AD, epsilon)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        f_ini = obj_func(Val{sym}(), AD, Measurement)
        f_list = [f_ini]
        println("initial value of $str3 is $(f_ini)")
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                while true
                    f_now = obj_func(Val{sym}(), AD, Measurement)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current value of $str3 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                end
            else
                gradient_QFIM!(AD, epsilon)
                while true
                    f_now = obj_func(Val{sym}(), AD, Measurement)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current value of $str3 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM!(AD, epsilon)
                end
            end
        else
            if Adam == true
                gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                while true
                    f_now = obj_func(Val{sym}(), AD, Measurement)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str3 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                end
            else
                gradient_QFIM!(AD, epsilon)
                while true
                    f_now = obj_func(Val{sym}(), AD, Measurement)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str3 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM!(AD, epsilon)
                end
            end
        end
    end
end
