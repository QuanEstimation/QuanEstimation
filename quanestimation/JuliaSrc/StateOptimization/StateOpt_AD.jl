############# time-independent Hamiltonian (noiseless) ################
function gradient_QFI!(AD::TimeIndepend_noiseless{T}, lr) where {T <: Complex}
    δF = gradient(x->QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x, AD.times), AD.psi)[1]
    AD.psi += lr*δF
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFI_Adam!(AD::TimeIndepend_noiseless{T}, lr, mt, vt, beta1, beta2, precision) where {T <: Complex}
    δF = gradient(x->QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x, AD.times), AD.psi)[1]
    StateOpt_Adam!(AD, δF, lr, mt, vt, beta1, beta2, precision) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFIM!(AD::TimeIndepend_noiseless{T}, lr) where {T <: Complex}
    δF = gradient(x->1/(AD.W*(QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, x, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    AD.psi += lr*δF
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFIM_Adam!(AD::TimeIndepend_noiseless{T}, lr, mt, vt, beta1, beta2, precision) where {T <: Complex}
    δF = gradient(x->1/(AD.W*(QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, x, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    StateOpt_Adam!(AD, δF, lr, mt, vt, beta1, beta2, precision) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFI!(AD::TimeIndepend_noiseless{T}, Measurement, lr) where {T <: Complex}
    δI = gradient(x->CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x, AD.times), AD.psi)[1]
    AD.psi += lr*δI
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFI_Adam!(AD::TimeIndepend_noiseless{T}, Measurement, lr, mt, vt, beta1, beta2, precision) where {T <: Complex}
    δI = gradient(x->CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x, AD.times), AD.psi)[1]
    StateOpt_Adam!(AD, δI, lr, mt, vt, beta1, beta2, precision) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFIM!(AD::TimeIndepend_noiseless{T}, Measurement, lr) where {T <: Complex}
    δI = gradient(x->1/(AD.W*(CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    AD.psi += lr*δI
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFIM_Adam!(AD::TimeIndepend_noiseless{T}, Measurement, lr, mt, vt, beta1, beta2, precision) where {T <: Complex}
    δI = gradient(x->1/(AD.W*(CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    StateOpt_Adam!(AD, δI, lr, mt, vt, beta1, beta2, precision) 
    AD.psi = AD.psi/norm(AD.psi)
end

function AD_QFIM(AD::TimeIndepend_noiseless{T}, precision, mt, vt, lr, beta1, beta2, max_episodes, Adam, save_file) where {T <: Complex}
    println("state optimization")
    episodes = 1
    dim = length(AD.psi)
    if length(AD.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        f_ini = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
        f_list = [f_ini]
        println("initial QFI is $(f_ini)")
        if Adam == true
            gradient_QFI_Adam!(AD, lr, mt, vt, beta1, beta2, precision)
        else
            gradient_QFI!(AD, lr)
        end
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
                    if  abs(f_now - f_ini) < precision  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile_state(f_now, AD.psi)
                        print("current QFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI_Adam!(AD, lr, mt, vt, beta1, beta2, precision)
                end
            else
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
                    if  abs(f_now - f_ini) < precision  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile_state(f_now, AD.psi)
                        print("current QFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI!(AD, lr)
                end
            end
        else
            if Adam == true
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
                    if  abs(f_now - f_ini) < precision  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current QFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI_Adam!(AD, lr, mt, vt, beta1, beta2, precision)
                end
            else
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
                    if  abs(f_now - f_ini) < precision  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current QFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI!(AD, lr)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
        f_ini = real(tr(AD.W*pinv(F)))
        f_list = [f_ini]
        println("initial value of Tr(WF^{-1}) is $(f_ini)")
        if Adam == true
            gradient_QFIM_Adam!(AD, lr, mt, vt, beta1, beta2, precision)
        else
            gradient_QFIM!(AD, lr)
        end
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  abs(f_now - f_ini) < precision || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile_state(f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(AD, lr, mt, vt, beta1, beta2, precision)
                end
            else
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  abs(f_now - f_ini) < precision || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile_state(f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM!(AD, lr)
                end
            end
        else
            if Adam == true
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  abs(f_now - f_ini) < precision || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(AD, lr, mt, vt, beta1, beta2, precision)
                end
            else
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  abs(f_now - f_ini) < precision || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM!(AD, lr)
                end
            end
        end
    end
end

function AD_CFIM(M, AD::TimeIndepend_noiseless{T}, precision, mt, vt, lr, beta1, beta2, max_episodes, Adam, save_file) where {T <: Complex}
    println("state optimization")
    episodes = 1
    dim = length(AD.psi)
    if length(AD.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        f_ini = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
        f_list = [f_ini]
        println("initial CFI is $(f_ini)")
        if Adam == true
            gradient_CFI_Adam!(AD, M, lr, mt, vt, beta1, beta2, precision)
        else
            gradient_CFI!(AD, M, lr)
        end
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                while true
                    f_now = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
                    if  abs(f_now - f_ini) < precision  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile_state(f_now, AD.psi)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI_Adam!(AD, M, lr, mt, vt, beta1, beta2, precision)
                end
            else
                while true
                    f_now = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
                    if  abs(f_now - f_ini) < precision  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile_state(f_now, AD.psi)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI!(AD, M, lr)
                end
            end
        else
            if Adam == true
                while true
                    f_now = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
                    if  abs(f_now - f_ini) < precision  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI_Adam!(AD, M, lr, mt, vt, beta1, beta2, precision)
                end
            else
                while true
                    f_now = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
                    if  abs(f_now - f_ini) < precision  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI!(AD, M, lr)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
        f_ini = real(tr(AD.W*pinv(F)))
        f_list = [f_ini]
        println("initial value of Tr(WF^{-1}) is $(f_ini)")
        if Adam == true
            gradient_CFIM_Adam!(AD, M, lr, mt, vt, beta1, beta2, precision)
        else
            gradient_CFIM!(AD, M, lr)
        end
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                while true
                    F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  abs(f_now - f_ini) < precision || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile_state(f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(AD, M, lr, mt, vt, beta1, beta2, precision)
                end
            else
                while true
                    F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  abs(f_now - f_ini) < precision || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile_state(f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM!(AD, M, lr)
                end
            end
        else
            if Adam == true
                while true
                    F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  abs(f_now - f_ini) < precision || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(AD, M, lr, mt, vt, beta1, beta2, precision)
                end
            else
                while true
                    F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  abs(f_now - f_ini) < precision || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM!(AD, M, lr)
                end
            end
        end
    end
end

############# time-independent Hamiltonian (noise) ################
function gradient_QFI!(AD::TimeIndepend_noise{T}, lr) where {T <: Complex}
    δF = gradient(x->QFIM_TimeIndepend_AD(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x*x', AD.Liouville_operator, AD.γ, AD.times), AD.psi)[1]
    AD.psi += lr*δF
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFI_Adam!(AD::TimeIndepend_noise{T}, lr, mt, vt, beta1, beta2, precision) where {T <: Complex}
    δF = gradient(x->QFIM_TimeIndepend_AD(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x*x', AD.Liouville_operator, AD.γ, AD.times), AD.psi)[1]
    StateOpt_Adam!(AD, δF, lr, mt, vt, beta1, beta2, precision) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFIM!(AD::TimeIndepend_noise{T}, lr) where {T <: Complex}
    δF = gradient(x->1/(AD.W*(QFIM_TimeIndepend_AD(AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.Liouville_operator, AD.γ, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    AD.psi += lr*δF
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFIM_Adam!(AD::TimeIndepend_noise{T}, lr, mt, vt, beta1, beta2, precision) where {T <: Complex}
    δF = gradient(x->1/(AD.W*(QFIM_TimeIndepend_AD(AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.Liouville_operator, AD.γ, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    StateOpt_Adam!(AD, δF, lr, mt, vt, beta1, beta2, precision) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFI!(AD::TimeIndepend_noise{T}, Measurement, lr) where {T <: Complex}
    δI = gradient(x->CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x*x', AD.Liouville_operator, AD.γ, AD.times), AD.psi)[1]
    AD.psi += lr*δI
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFI_Adam!(AD::TimeIndepend_noise{T}, Measurement, lr, mt, vt, beta1, beta2, precision) where {T <: Complex}
    δI = gradient(x->CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x*x', AD.Liouville_operator, AD.γ, AD.times), AD.psi)[1]
    StateOpt_Adam!(AD, δI, lr, mt, vt, beta1, beta2, precision) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFIM!(AD::TimeIndepend_noise{T}, Measurement, lr) where {T <: Complex}
    δI = gradient(x->1/(AD.W*(CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.Liouville_operator, AD.γ, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    AD.psi += lr*δI
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFIM_Adam!(AD::TimeIndepend_noise{T}, Measurement, lr, mt, vt, beta1, beta2, precision) where {T <: Complex}
    δI = gradient(x->1/(AD.W*(CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.Liouville_operator, AD.γ, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    StateOpt_Adam!(AD, δI, lr, mt, vt, beta1, beta2, precision) 
    AD.psi = AD.psi/norm(AD.psi)
end

function AD_QFIM(AD::TimeIndepend_noise{T}, precision, mt, vt, lr, beta1, beta2, max_episodes, Adam, save_file) where {T <: Complex}
    println("state optimization")
    episodes = 1
    dim = length(AD.psi)
    if length(AD.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        f_ini = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
        f_list = [f_ini]
        println("initial QFI is $(f_ini)")
        if Adam == true
            gradient_QFI_Adam!(AD, lr, mt, vt, beta1, beta2, precision)
        else
            gradient_QFI!(AD, lr)
        end
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    if  abs(f_now - f_ini) < precision  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile_state(f_now, AD.psi)
                        print("current QFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI_Adam!(AD, lr, mt, vt, beta1, beta2, precision)
                end
            else
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    if  abs(f_now - f_ini) < precision  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile_state(f_now, AD.psi)
                        print("current QFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI!(AD, lr)
                end
            end
        else
            if Adam == true
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    if  abs(f_now - f_ini) < precision  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current QFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI_Adam!(AD, lr, mt, vt, beta1, beta2, precision)
                end
            else
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    if  abs(f_now - f_ini) < precision  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current QFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFI!(AD, lr)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
        f_ini = real(tr(AD.W*pinv(F)))
        f_list = [f_ini]
        println("initial value of Tr(WF^{-1}) is $(f_ini)")
        if Adam == true
            gradient_QFIM_Adam!(AD, lr, mt, vt, beta1, beta2, precision)
        else
            gradient_QFIM!(AD, lr)
        end
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  abs(f_now - f_ini) < precision || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile_state(f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(AD, lr, mt, vt, beta1, beta2, precision)
                end
            else
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  abs(f_now - f_ini) < precision || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile_state(f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM!(AD, lr)
                end
            end
        else
            if Adam == true
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  abs(f_now - f_ini) < precision || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM_Adam!(AD, lr, mt, vt, beta1, beta2, precision)
                end
            else
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  abs(f_now - f_ini) < precision || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_QFIM!(AD, lr)
                end
            end
        end
    end
end

function AD_CFIM(M, AD::TimeIndepend_noise{T}, precision, mt, vt, lr, beta1, beta2, max_episodes, Adam, save_file) where {T <: Complex}
    println("state optimization")
    episodes = 1
    dim = length(AD.psi)
    if length(AD.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        f_ini = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
        f_list = [f_ini]
        println("initial CFI is $(f_ini)")
        if Adam == true
            gradient_CFI_Adam!(AD, M, lr, mt, vt, beta1, beta2, precision)
        else
            gradient_CFI!(AD, M, lr)
        end
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                while true
                    f_now = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    if  abs(f_now - f_ini) < precision  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile_state(f_now, AD.psi)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI_Adam!(AD, M, lr, mt, vt, beta1, beta2, precision)
                end
            else
                while true
                    f_now = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    if  abs(f_now - f_ini) < precision  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile_state(f_now, AD.psi)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI!(AD, M, lr)
                end
            end
        else
            if Adam == true
                while true
                    f_now = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    if  abs(f_now - f_ini) < precision  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI_Adam!(AD, M, lr, mt, vt, beta1, beta2, precision)
                end
            else
                while true
                    f_now = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    if  abs(f_now - f_ini) < precision  || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI!(AD, M, lr)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
        f_ini = real(tr(AD.W*pinv(F)))
        f_list = [f_ini]
        println("initial value of Tr(WF^{-1}) is $(f_ini)")
        if Adam == true
            gradient_CFIM_Adam!(AD, M, lr, mt, vt, beta1, beta2, precision)
        else
            gradient_CFIM!(AD, M, lr)
        end
        if save_file == true
            SaveFile_state(f_ini, AD.psi)
            if Adam == true
                while true
                    F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  abs(f_now - f_ini) < precision || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile_state(f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(AD, M, lr, mt, vt, beta1, beta2, precision)
                end
            else
                while true
                    F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  abs(f_now - f_ini) < precision || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile_state(f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        SaveFile_state(f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM!(AD, M, lr)
                end
            end
        else
            if Adam == true
                while true
                    F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  abs(f_now - f_ini) < precision || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(AD, M, lr, mt, vt, beta1, beta2, precision)
                end
            else
                while true
                    F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    if  abs(f_now - f_ini) < precision || episodes >= max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_state(f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM!(AD, M, lr)
                end
            end
        end
    end
end
