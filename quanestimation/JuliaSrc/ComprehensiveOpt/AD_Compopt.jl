################ state and control optimization ###############
function SC_AD_Compopt(AD::SC_Compopt{T}, max_episode, epsilon, mt, vt, beta1, beta2, accuracy, Adam, save_file) where {T<:Complex}
    sym = Symbol("QFIM_SCopt")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    M = [zeros(ComplexF64, size(AD.psi)[1], size(AD.psi)[1])]
    return info_AD_SCopt(M, AD, max_episode, epsilon, mt, vt, beta1, beta2, accuracy, Adam, save_file, sym, str1, str2, str3)
end
 
function gradient_QFI!(AD::SC_Compopt{T}, epsilon) where {T<:Complex}
    arr = [AD.psi, AD.control_coefficients]
    δF = gradient(x->QFI(x, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.decay_opt, AD.γ, AD.control_Hamiltonian, AD.tspan, AD.accuracy), arr)[1]
    δF1 = δF[1]
    δF2 = δF[2]
    AD.psi += epsilon*δF1
    AD.psi = AD.psi/norm(AD.psi)

    AD.control_coefficients += epsilon*real(δF2)
    bound!(AD.control_coefficients, AD.ctrl_bound)
end

function gradient_QFI_Adam!(AD::SC_Compopt{T}, epsilon, mt, vt, beta1, beta2, accuracy) where {T<:Complex}
    arr = [AD.psi, AD.control_coefficients]

    δF = gradient(x->QFI(x, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.decay_opt, AD.γ, AD.control_Hamiltonian, AD.tspan, AD.accuracy), arr)[1]
    δF1 = δF[1]
    δF2 = δF[2]
    StateOpt_Adam!(AD, δF1, epsilon, mt, vt, beta1, beta2, accuracy) 
    AD.psi = AD.psi/norm(AD.psi)

    Adam!(AD, real(δF2), epsilon, mt, vt, beta1, beta2, accuracy)
    bound!(AD.control_coefficients, AD.ctrl_bound)
end

function gradient_QFIM!(AD::SC_Compopt{T}, epsilon) where {T<:Complex}
    arr = [AD.psi, AD.control_coefficients]
    δF = gradient(x->1/(AD.W*pinv(QFIM(x, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.decay_opt, AD.γ, AD.control_Hamiltonian, AD.tspan, AD.accuracy), rtol=AD.accuracy) |> tr |>real), arr) |>sum
    δF1 = δF[1]
    δF2 = δF[2]

    AD.psi += epsilon*δF1
    AD.psi = AD.psi/norm(AD.psi)

    AD.control_coefficients += epsilon*real(δF2)
    bound!(AD.control_coefficients, AD.ctrl_bound)
end

function gradient_QFIM_Adam!(AD::SC_Compopt{T}, epsilon, mt, vt, beta1, beta2, accuracy) where {T<:Complex}
    arr = [AD.psi, AD.control_coefficients]
    δF = gradient(x->1/(AD.W*pinv(QFIM(x, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.decay_opt, AD.γ, AD.control_Hamiltonian, AD.tspan, AD.accuracy), rtol=AD.accuracy) |> tr |>real), arr) |>sum
    δF1 = δF[1]
    δF2 = δF[2]
    StateOpt_Adam!(AD, δF1, epsilon, mt, vt, beta1, beta2, accuracy) 
    AD.psi = AD.psi/norm(AD.psi)

    Adam!(AD, real(δF2), epsilon, mt, vt, beta1, beta2, accuracy)
    bound!(AD.control_coefficients, AD.ctrl_bound)
end

function info_AD_SCopt(M, AD, max_episode, epsilon, mt, vt, beta1, beta2, accuracy, Adam, save_file, sym, str1, str2, str3)
    println("comprehensive optimization")
    ctrl_num = length(AD.control_Hamiltonian)
    ctrl_length = length(AD.control_coefficients[1])
    episodes = 1
    if length(AD.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("algorithm: Automatic Differentiation (AD)")
        f_noctrl = obj_func(Val{:QFIM_noctrl}(), AD, M, AD.psi)
        f_ini = obj_func(Val{sym}(), AD, M, AD.psi, AD.control_coefficients)

        f_list = [1.0/f_ini]
        println("non-controlled $str2 is $(1.0/f_noctrl)")
        println("initial $str2 is $(1.0/f_ini)")
        if save_file == true
            SaveFile_SC(1.0/f_ini, AD.psi, AD.control_coefficients)
            if Adam == true
                gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, accuracy)
                while true
                    f_now = obj_func(Val{sym}(), AD, M, AD.psi, AD.control_coefficients)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        SaveFile_SC(f_now, AD.psi, AD.control_coefficients)
                        break
                    else
                        episodes += 1
                        SaveFile_SC(f_now, AD.psi, AD.control_coefficients)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, accuracy)
                end
            else
                gradient_QFI!(AD, epsilon)
                while true
                    f_now = obj_func(Val{sym}(), AD, M, AD.psi, AD.control_coefficients)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        SaveFile_SC(f_now, AD.psi, AD.control_coefficients)
                        break
                    else
                        episodes += 1
                        SaveFile_SC(f_now, AD.psi, AD.control_coefficients)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFI!(AD, epsilon)
                end
            end
        else
            if Adam == true
                gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, accuracy)
                while true
                    f_now = obj_func(Val{sym}(), AD, M, AD.psi, AD.control_coefficients)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_SC(f_list, AD.psi, AD.control_coefficients)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, accuracy)
                end
            else
                gradient_QFI!(AD, epsilon)
                while true
                    f_now = obj_func(Val{sym}(), AD, M, AD.psi, AD.control_coefficients)
                    f_now = 1.0/f_now
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_SC(f_list, AD.psi, AD.control_coefficients)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str2 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFI!(AD, epsilon)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("algorithm: Automatic Differentiation (AD)")
        f_noctrl = obj_func(Val{:QFIM_noctrl}(), AD, M, AD.psi)
        f_ini = obj_func(Val{sym}(), AD, M, AD.psi, AD.control_coefficients)

        f_list = [f_ini]
        println("non-controlled value of $str3 is $(f_noctrl)")
        println("initial value of $str3 is $(f_ini)")
        if save_file == true
            SaveFile_SC(f_ini, AD.psi, AD.control_coefficients)
            if Adam == true
                gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, accuracy)
                while true
                    f_now = obj_func(Val{sym}(), AD, M, AD.psi, AD.control_coefficients)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        SaveFile_SC(f_now, AD.psi, AD.control_coefficients)
                        break
                    else
                        episodes += 1
                        SaveFile_SC(f_now, AD.psi, AD.control_coefficients)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, accuracy)
                end
            else
                gradient_QFIM!(AD, epsilon)
                while true
                    f_now = obj_func(Val{sym}(), AD, M, AD.psi, AD.control_coefficients)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        SaveFile_SC(f_now, AD.psi, AD.control_coefficients)
                        break
                    else
                        episodes += 1
                        SaveFile_SC(f_now, AD.psi, AD.control_coefficients)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFIM!(AD, epsilon)
                end
            end
        else
            if Adam == true
                gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, accuracy)
                while true
                    f_now = obj_func(Val{sym}(), AD, M, AD.psi, AD.control_coefficients)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_SC(f_list, AD.psi, AD.control_coefficients)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, accuracy)
                end
            else
                gradient_QFIM!(AD, epsilon)
                while true
                    f_now = obj_func(Val{sym}(), AD, M, AD.psi, AD.control_coefficients)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str3 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_SC(f_list, AD.psi, AD.control_coefficients)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str3 is ", f_now, " ($(episodes-1) episodes)    \r")
                    end
                    gradient_QFIM!(AD, epsilon)
                end
            end
        end
    end
end
