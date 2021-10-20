mutable struct StateOpt_AD{T <: Complex,M <: Real}
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
    mt::M
    vt::M
    ϵ::M
    beta1::M
    beta2::M
    precision::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    StateOpt_AD(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, psi::Vector{T},
                 times::Vector{M}, Liouville_operator::Vector{Matrix{T}},γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}},
                 control_coefficients::Vector{Vector{M}}, ctrl_bound::M, W::Matrix{M}, mt::M, vt::M, ϵ::M, beta1::M, beta2::M, precision::M, 
                 ρ=Vector{Matrix{T}}(undef, 1), ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1),∂ρ_∂V=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = 
                 new{T,M}(freeHamiltonian, Hamiltonian_derivative, psi, times, Liouville_operator, γ, control_Hamiltonian,
                          control_coefficients, ctrl_bound, W, mt, vt, ϵ, beta1, beta2, precision, ρ, ∂ρ_∂x) 
end

function gradient_QFI!(AD::StateOpt_AD{T}) where {T <: Complex}
    δF = gradient(x->QFI(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x*x', AD.Liouville_operator, AD.γ, AD.control_Hamiltonian, AD.control_coefficients, AD.times), AD.psi)[1]
    AD.psi += AD.ϵ*δF
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFI_Adam!(AD::StateOpt_AD{T}) where {T <: Complex}
    δF = gradient(x->QFI(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x*x', AD.Liouville_operator, AD.γ, AD.control_Hamiltonian, AD.control_coefficients, AD.times), AD.psi)[1]
    StateOpt_Adam!(AD, δF) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFIM!(AD::StateOpt_AD{T}) where {T <: Complex}
    δF = gradient(x->1/(AD.W*(QFIM(AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.Liouville_operator, AD.γ, AD.control_Hamiltonian, AD.control_coefficients, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    AD.psi += AD.ϵ*δF
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFIM_Adam!(AD::StateOpt_AD{T}) where {T <: Complex}
    δF = gradient(x->1/(AD.W*(QFIM(AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.Liouville_operator, AD.γ, AD.control_Hamiltonian, AD.control_coefficients, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    StateOpt_Adam!(AD, δF) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFI!(AD::StateOpt_AD{T}, Measurement) where {T <: Complex}
    δI = gradient(x->CFI(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x*x', AD.Liouville_operator, AD.γ, AD.control_Hamiltonian, AD.control_coefficients, AD.times), AD.psi)[1]
    AD.psi += AD.ϵ*δI
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFI_Adam!(AD::StateOpt_AD{T}, Measurement) where {T <: Complex}
    δI = gradient(x->CFI(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x*x', AD.Liouville_operator, AD.γ, AD.control_Hamiltonian, AD.control_coefficients, AD.times), AD.psi)[1]
    StateOpt_Adam!(AD, δI) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFIM!(AD::StateOpt_AD{T}, Measurement) where {T <: Complex}
    δI = gradient(x->1/(AD.W*(CFIM(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.Liouville_operator, AD.γ, AD.control_Hamiltonian, AD.control_coefficients, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    AD.psi += AD.ϵ*δI
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFIM_Adam!(AD::StateOpt_AD{T}, Measurement) where {T <: Complex}
    δI = gradient(x->1/(AD.W*(CFIM(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.Liouville_operator, AD.γ, AD.control_Hamiltonian, AD.control_coefficients, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    StateOpt_Adam!(AD, δI) 
    AD.psi = AD.psi/norm(AD.psi)
end

function SaveFile(Tend, f_now, control)
    open("f_auto_T$Tend.csv","a") do f
        writedlm(f, [f_now])
    end
    open("state_auto_T$Tend.csv","a") do g
        writedlm(g, control)
    end
end

function AD_QFIM(AD, epsilon, max_episodes, Adam, save_file)
    println("state optimization")
    episodes = 1
    Tend = (AD.times)[end]
    if length(AD.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: Automatic Differentiation")
        f_ini = QFI_ori(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                        AD.control_Hamiltonian, AD.control_coefficients, AD.times)
        f_list = [f_ini]
        println("initial QFI is $(f_ini)")
        if Adam == true
            gradient_QFI_Adam!(AD)
        else
            gradient_QFI!(AD)
        end
        if save_file == true
            if Adam == true
                while true
                    f_now = QFI_ori(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                                    AD.control_Hamiltonian, AD.control_coefficients, AD.times)
                    gradient_QFI_Adam!(AD)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(Tend, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(Tend, f_now, AD.psi)
                        print("current QFI is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            else
                while true
                    f_now = QFI_ori(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                                    AD.control_Hamiltonian, AD.control_coefficients, AD.times)
                    gradient_QFI!(AD)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(Tend, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(Tend, f_now, AD.psi)
                        print("current QFI is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    f_now = QFI_ori(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                                    AD.control_Hamiltonian, AD.control_coefficients, AD.times)
                    gradient_QFI_Adam!(AD)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(Tend, f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        print("current QFI is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            else
                while true
                    f_now = QFI_ori(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                                    AD.control_Hamiltonian, AD.control_coefficients, AD.times)
                    gradient_QFI!(AD)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(Tend, f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        print("current QFI is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            end
        end
    else
        println("multiparameter scenario")
        println("control algorithm: Automatic Differentiation")
        F = QFIM_ori(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                    AD.control_Hamiltonian, AD.control_coefficients, AD.times)
        f_ini = real(tr(AD.W*pinv(F)))
        f_list = [f_ini]
        println("initial value of Tr(WF^{-1}) is $(f_ini)")
        if Adam == true
            gradient_QFIM_Adam!(AD)
        else
            gradient_QFIM!(AD)
        end
        if save_file == true
            if Adam == true
                while true
                    F = QFIM_ori(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                                 AD.control_Hamiltonian, AD.control_coefficients, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_QFIM_Adam!(AD)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(Tend, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(Tend, f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            else
                while true
                    F = QFIM_ori(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                                 AD.control_Hamiltonian, AD.control_coefficients, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_QFIM!(AD)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(Tend, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(Tend, f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    F = QFIM_ori(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                                 AD.control_Hamiltonian, AD.control_coefficients, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_QFIM_Adam!(AD)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(Tend, f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            else
                while true
                    F = QFIM_ori(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                                 AD.control_Hamiltonian, AD.control_coefficients, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_QFIM!(AD)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(Tend, f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            end
        end
    end
end

function AD_CFIM(M, AD, epsilon, max_episodes, Adam, save_file)
    println("state optimization")
    episodes = 1
    Tend = (AD.times)[end]
    if length(AD.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: Automatic Differentiation")
        f_ini = CFI(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                        AD.control_Hamiltonian, AD.control_coefficients, AD.times)
        f_list = [f_ini]
        println("initial CFI is $(f_ini)")
        if Adam == true
            gradient_CFI_Adam!(AD, M)
        else
            gradient_CFI!(AD, M)
        end
        if save_file == true
            if Adam == true
                while true
                    f_now = CFI(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                                    AD.control_Hamiltonian, AD.control_coefficients, AD.times)
                    gradient_CFI_Adam!(AD, M)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile(Tend, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(Tend, f_now, AD.psi)
                        print("current CFI is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            else
                while true
                    f_now = CFI(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                                    AD.control_Hamiltonian, AD.control_coefficients, AD.times)
                    gradient_CFI!(AD, M)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile(Tend, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(Tend, f_now, AD.psi)
                        print("current CFI is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    f_now = CFI(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                                    AD.control_Hamiltonian, AD.control_coefficients, AD.times)
                    gradient_CFI_Adam!(AD, M)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile(Tend, f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        print("current CFI is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            else
                while true
                    f_now = CFI(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                                    AD.control_Hamiltonian, AD.control_coefficients, AD.times)
                    gradient_CFI!(AD, M)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile(Tend, f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        print("current CFI is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            end
        end
    else
        println("multiparameter scenario")
        println("control algorithm: Automatic Differentiation")
        F = CFIM(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                    AD.control_Hamiltonian, AD.control_coefficients, AD.times)
        f_ini = real(tr(AD.W*pinv(F)))
        f_list = [f_ini]
        println("initial value of Tr(WF^{-1}) is $(f_ini)")
        if Adam == true
            gradient_CFIM_Adam!(AD, M)
        else
            gradient_CFIM!(AD, M)
        end
        if save_file == true
            if Adam == true
                while true
                    F = CFIM(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                                 AD.control_Hamiltonian, AD.control_coefficients, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_CFIM_Adam!(AD, M)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(Tend, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(Tend, f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            else
                while true
                    F = CFIM(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                                 AD.control_Hamiltonian, AD.control_coefficients, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_CFIM!(AD, M)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(Tend, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(Tend, f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    F = CFIM(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                                 AD.control_Hamiltonian, AD.control_coefficients, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_CFIM_Adam!(AD, M)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(Tend, f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            else
                while true
                    F = CFIM(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)', AD.Liouville_operator, AD.γ, 
                                 AD.control_Hamiltonian, AD.control_coefficients, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_CFIM!(AD, M)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(Tend, f_list, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            end
        end
    end
end
