function gradient_CFI!(AD::projection_Mopt{T}, epsilon) where {T<:Complex}
    M_num = length(AD.Measurement)
    δI = gradient(x->CFI([x[i]*x[i]' for i in 1:M_num], AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy), AD.Measurement)[1]
    AD.Measurement[1] += epsilon*δI[1]
    AD.Measurement = gramschmidt(AD.Measurement)
end

# function gradient_CFI_Adam!(AD::projection_Mopt{T}, epsilon, mt, vt, beta1, beta2) where {T<:Complex}
#     M_num = length(AD.Measurement)
#     δI = gradient(x->CFI([x[i]*x[i]' for i in 1:M_num], AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy), AD.Measurement)[1]
#     StateOpt_Adam!(AD, δI, epsilon, mt, vt, beta1, beta2, AD.accuracy) 
#     AD.Measurement = gramschmidt(AD.Measurement)
# end

function gradient_CFIM!(AD::projection_Mopt{T}, epsilon) where {T<:Complex}
    M_num = length(AD.Measurement)
    δI = gradient(x->1/(AD.W*(CFIM([x[i]*x[i]' for i in 1:M_num], AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy) |> pinv) |> tr |>real), AD.Measurement) |>sum
    AD.Measurement[1] += epsilon*δI[1]
    AD.Measurement = gramschmidt(AD.Measurement)
end

# function gradient_CFIM_Adam!(AD::projection_Mopt{T}, epsilon, mt, vt, beta1, beta2) where {T<:Complex}
#     M_num = length(AD.Measurement)
#     δI = gradient(x->1/(AD.W*(CFIM([x[i]*x[i]' for i in 1:M_num], AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy) |> pinv) |> tr |>real), AD.Measurement) |>sum
#     StateOpt_Adam!(AD, δI, epsilon, mt, vt, beta1, beta2, AD.accuracy) 
#     AD.Measurement = gramschmidt(AD.Measurement)
# end

function CFIM_AD_Mopt(AD::projection_Mopt{T}, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file) where {T<:Complex}
    println("measurement optimization")
    dim = size(AD.ρ0)[1]
    M_num = length(AD.Measurement)
    
    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
    F_tp = CFIM(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
    f_ini = 1.0/real(tr(AD.W*pinv(F_tp)))
    f_list = [f_ini]
    F_opt = QFIM(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
    f_opt = real(tr(AD.W*pinv(F_opt)))

    episodes = 1
    if length(AD.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        println("initial CFI is $(f_ini)")
        println("QFI is $(1.0/f_opt)")

        if Adam == true
            gradient_CFI_Adam!(AD, epsilon, mt, vt, beta1, beta2)
        else
            gradient_CFI!(AD, epsilon)
        end
        if save_file == true
            SaveFile_meas(f_ini, AD.Measurement)
            if Adam == true
                while true
                    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
                    F_tp = CFIM(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
                    f_now = 1.0/real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile_meas(f_now, AD.Measurement)
                        break
                    else
                        episodes += 1
                        SaveFile_meas(f_now, AD.Measurement)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI_Adam!(AD, epsilon, mt, vt, beta1, beta2)
                end
            else
                while true
                    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
                    F_tp = CFIM(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
                    f_now = 1.0/real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile_meas(f_now, AD.Measurement)
                        break
                    else
                        episodes += 1
                        SaveFile_meas(f_now, AD.Measurement)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI!(AD, epsilon)
                end
            end
        else
            if Adam == true
                while true
                    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
                    F_tp = CFIM(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
                    f_now = 1.0/real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_meas(f_list, AD.Measurement)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI_Adam!(AD, epsilon, mt, vt, beta1, beta2)
                end
            else
                while true
                    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
                    F_tp = CFIM(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
                    f_now = 1.0/real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_meas(f_list, AD.Measurement)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current CFI is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI!(AD, epsilon)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        println("initial value of Tr(WI^{-1}) is $(1.0/f_ini)")
        println("Tr(WF^{-1}) is $(f_opt)")

        if Adam == true
            gradient_CFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2)
        else
            gradient_CFIM!(AD, epsilon)
        end
        if save_file == true
            SaveFile_meas(f_ini, AD.Measurement)
            if Adam == true
                while true
                    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
                    F_tp = CFIM(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        SaveFile_meas(f_now, AD.Measurement)
                        break
                    else
                        episodes += 1
                        SaveFile_meas(f_now, AD.Measurement)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2)
                end
            else
                while true
                    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
                    F_tp = CFIM(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        SaveFile_meas(f_now, AD.Measurement)
                        break
                    else
                        episodes += 1
                        SaveFile_meas(f_now, AD.Measurement)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM!(AD, epsilon)
                end
            end
        else
            if Adam == true
                while true
                    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
                    F_tp = CFIM(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_meas(f_list, AD.Measurement)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2)
                end
            else
                while true
                    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
                    F_tp = CFIM(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy)
                    f_now = real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WI^{-1}) is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_meas(f_list, AD.Measurement)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of Tr(WI^{-1}) is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM!(AD, epsilon)
                end
            end
        end
    end
end
