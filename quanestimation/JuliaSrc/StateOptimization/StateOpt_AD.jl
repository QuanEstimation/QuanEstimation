############# time-independent Hamiltonian (noiseless) ################
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
    δF = gradient(x->1/(AD.W*(QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, x, AD.tspan, AD.accuracy) |> pinv) |> tr |>real), AD.psi) |>sum
    AD.psi += epsilon*δF
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFIM_Adam!(AD::TimeIndepend_noiseless{T}, epsilon, mt, vt, beta1, beta2, accuracy) where {T <: Complex}
    δF = gradient(x->1/(AD.W*(QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, x, AD.tspan, AD.accuracy) |> pinv) |> tr |>real), AD.psi) |>sum
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
    δI = gradient(x->1/(AD.W*(CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x, AD.tspan, AD.accuracy) |> pinv) |> tr |>real), AD.psi) |>sum
    AD.psi += epsilon*δI
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFIM_Adam!(AD::TimeIndepend_noiseless{T}, Measurement, epsilon, mt, vt, beta1, beta2) where {T <: Complex}
    δI = gradient(x->1/(AD.W*(CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x, AD.tspan, AD.accuracy) |> pinv) |> tr |>real), AD.psi) |>sum
    StateOpt_Adam!(AD, δI, epsilon, mt, vt, beta1, beta2, AD.accuracy) 
    AD.psi = AD.psi/norm(AD.psi)
end

function AD_QFIM(AD::TimeIndepend_noiseless{T}, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file) where {T <: Complex}
    println("state optimization")
    episodes = 1
    dim = length(AD.psi)
    if length(AD.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        f_ini = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.tspan, AD.accuracy)
        f_list = [f_ini]
        println("initial QFI is $(f_ini)")
        if Adam == true
            gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
        else
            gradient_QFI!(AD, epsilon)
        end
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.tspan, AD.accuracy)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current QFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                end
            else
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.tspan, AD.accuracy)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current QFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI!(AD, epsilon)
                end
            end
        else
            if Adam == true
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.tspan, AD.accuracy)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current QFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                end
            else
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.tspan, AD.accuracy)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current QFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI!(AD, epsilon)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.tspan, AD.accuracy)
        f_ini = real(tr(AD.W*pinv(F)))
        f_list = [f_ini]
        println("initial value of Tr(WF^{-1}) is $(f_ini)")
        if Adam == true
            gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
        else
            gradient_QFIM!(AD, epsilon)
        end
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                end
            else
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM!(AD, epsilon)
                end
            end
        else
            if Adam == true
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                end
            else
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM!(AD, epsilon)
                end
            end
        end
    end
end

function AD_CFIM(Measurement, AD::TimeIndepend_noiseless{T}, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file) where {T <: Complex}
    println("state optimization")
    episodes = 1
    dim = length(AD.psi)
    if length(AD.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        f_ini = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.tspan, AD.accuracy)
        f_list = [f_ini]
        println("initial CFI is $(f_ini)")
        if Adam == true
            gradient_CFI_Adam!(AD, Measurement, epsilon, mt, vt, beta1, beta2)
        else
            gradient_CFI!(AD, Measurement, epsilon)
        end
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                while true
                    f_now = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.tspan, AD.accuracy)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI_Adam!(AD, Measurement, epsilon, mt, vt, beta1, beta2)
                end
            else
                while true
                    f_now = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.tspan, AD.accuracy)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI!(AD, Measurement, epsilon)
                end
            end
        else
            if Adam == true
                while true
                    f_now = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.tspan, AD.accuracy)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI_Adam!(AD, Measurement, epsilon, mt, vt, beta1, beta2)
                end
            else
                while true
                    f_now = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.tspan, AD.accuracy)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI!(AD, Measurement, epsilon)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        F = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.tspan, AD.accuracy)
        f_ini = real(tr(AD.W*pinv(F)))
        f_list = [f_ini]
        println("initial value of Tr(WI^{-1}) is $(f_ini)")
        if Adam == true
            gradient_CFIM_Adam!(AD, Measurement, epsilon, mt, vt, beta1, beta2)
        else
            gradient_CFIM!(AD, Measurement, epsilon)
        end
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                while true
                    F = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(AD, Measurement, epsilon, mt, vt, beta1, beta2)
                end
            else
                while true
                    F = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM!(AD, Measurement, epsilon)
                end
            end
        else
            if Adam == true
                while true
                    F = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(AD, Measurement, epsilon, mt, vt, beta1, beta2)
                end
            else
                while true
                    F = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM!(AD, Measurement, epsilon)
                end
            end
        end
    end
end

############# time-independent Hamiltonian (noise) ################
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
    δF = gradient(x->1/(AD.W*(QFIM_TimeIndepend_AD(AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.decay_opt, AD.γ, AD.tspan) |> pinv) |> tr |>real), AD.psi) |>sum
    AD.psi += epsilon*δF
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFIM_Adam!(AD::TimeIndepend_noise{T}, epsilon, mt, vt, beta1, beta2, accuracy) where {T <: Complex}
    δF = gradient(x->1/(AD.W*(QFIM_TimeIndepend_AD(AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.decay_opt, AD.γ, AD.tspan) |> pinv) |> tr |>real), AD.psi) |>sum
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
    δI = gradient(x->1/(AD.W*(CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.decay_opt, AD.γ, AD.tspan, AD.accuracy) |> pinv) |> tr |>real), AD.psi) |>sum
    AD.psi += epsilon*δI
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFIM_Adam!(AD::TimeIndepend_noise{T}, Measurement, epsilon, mt, vt, beta1, beta2) where {T <: Complex}
    δI = gradient(x->1/(AD.W*(CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.decay_opt, AD.γ, AD.tspan, AD.accuracy) |> pinv) |> tr |>real), AD.psi) |>sum
    StateOpt_Adam!(AD, δI, epsilon, mt, vt, beta1, beta2, AD.accuracy) 
    AD.psi = AD.psi/norm(AD.psi)
end

function AD_QFIM(AD::TimeIndepend_noise{T}, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file) where {T <: Complex}
    println("state optimization")
    episodes = 1
    dim = length(AD.psi)
    if length(AD.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        f_ini = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.decay_opt, AD.γ, 
                                  AD.tspan, AD.accuracy)
        f_list = [f_ini]
        println("initial QFI is $(f_ini)")
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)', AD.decay_opt, 
                                              AD.γ, AD.tspan, AD.accuracy)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current QFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                end
            else
                gradient_QFI!(AD, epsilon)
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)', AD.decay_opt, 
                                              AD.γ, AD.tspan, AD.accuracy)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current QFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI!(AD, epsilon)
                end
            end
        else
            if Adam == true
                gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)', AD.decay_opt, 
                                              AD.γ, AD.tspan, AD.accuracy)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current QFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                end
            else
                gradient_QFI!(AD, epsilon)
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)', AD.decay_opt, 
                                              AD.γ, AD.tspan, AD.accuracy)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current QFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI!(AD, epsilon)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)', AD.decay_opt, AD.γ, 
                              AD.tspan, AD.accuracy)
        f_ini = real(tr(AD.W*pinv(F)))
        f_list = [f_ini]
        println("initial value of Tr(WF^{-1}) is $(f_ini)")
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)', AD.decay_opt,  
                                          AD.γ, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                end
            else
                gradient_QFIM!(AD, epsilon)
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)', AD.decay_opt,  
                                          AD.γ, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM!(AD, epsilon)
                end
            end
        else
            if Adam == true
                gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)', AD.decay_opt, 
                                          AD.γ, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, AD.accuracy)
                end
            else
                gradient_QFIM!(AD, epsilon)
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)', AD.decay_opt, 
                                          AD.γ, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM!(AD, epsilon)
                end
            end
        end
    end
end

function AD_CFIM(Measurement, AD::TimeIndepend_noise{T}, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file) where {T <: Complex}
    println("state optimization")
    episodes = 1
    dim = length(AD.psi)
    if length(AD.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        f_ini = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)', AD.decay_opt, 
                                  AD.γ, AD.tspan, AD.accuracy)
        f_list = [f_ini]
        println("initial CFI is $(f_ini)")
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                gradient_CFI_Adam!(AD, Measurement, epsilon, mt, vt, beta1, beta2)
                while true
                    f_now = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',
                                              AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI_Adam!(AD, Measurement, epsilon, mt, vt, beta1, beta2)
                end
            else
                gradient_CFI!(AD, Measurement, epsilon)
                while true
                    f_now = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',
                                              AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI!(AD, Measurement, epsilon)
                end
            end
        else
            if Adam == true
                gradient_CFI_Adam!(AD, Measurement, epsilon, mt, vt, beta1, beta2)
                while true
                    f_now = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',
                                              AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI_Adam!(AD, Measurement, epsilon, mt, vt, beta1, beta2)
                end
            else
                gradient_CFI!(AD, Measurement, epsilon)
                while true
                    f_now = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',
                                              AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI!(AD, Measurement, epsilon)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        F = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)', AD.decay_opt, 
                              AD.γ, AD.tspan, AD.accuracy)
        f_ini = real(tr(AD.W*pinv(F)))
        f_list = [f_ini]
        println("initial value of Tr(WI^{-1}) is $(f_ini)")
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                gradient_CFIM_Adam!(AD, Measurement, epsilon, mt, vt, beta1, beta2)
                while true
                    F = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',
                                          AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(AD, Measurement, epsilon, mt, vt, beta1, beta2)
                end
            else
                gradient_CFIM!(AD, Measurement, epsilon)
                while true
                    F = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',
                                          AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        episodes += 1
                        SaveFile_state(f_now, AD.psi)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM!(AD, Measurement, epsilon)
                end
            end
        else
            if Adam == true
                gradient_CFIM_Adam!(AD, Measurement, epsilon, mt, vt, beta1, beta2)
                while true
                    F = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',
                                          AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(AD, Measurement, epsilon, mt, vt, beta1, beta2)
                end
            else
                gradient_CFIM!(AD, Measurement, epsilon)
                while true
                    F = CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',
                                          AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM!(AD, Measurement, epsilon)
                end
            end
        end
    end
end
