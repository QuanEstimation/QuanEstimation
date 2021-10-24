############# time-independent Hamiltonian (noiseless) ################
mutable struct StateOptAD_TimeIndepend_noiseless{T <: Complex,M <: Real}
    freeHamiltonian::Matrix{T}
    Hamiltonian_derivative::Vector{Matrix{T}}
    psi::Vector{T}
    times::Vector{M}
    W::Matrix{M}
    mt::M
    vt::M
    ϵ::M
    beta1::M
    beta2::M
    precision::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    StateOptAD_TimeIndepend_noiseless(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, psi::Vector{T},
                 times::Vector{M}, W::Matrix{M}, mt::M, vt::M, ϵ::M, beta1::M, beta2::M, precision::M, 
                 ρ=Vector{Matrix{T}}(undef, 1), ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1),∂ρ_∂V=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = 
                 new{T,M}(freeHamiltonian, Hamiltonian_derivative, psi, times, W, mt, vt, ϵ, beta1, beta2, precision, ρ, ∂ρ_∂x) 
end

function gradient_QFI!(AD::StateOptAD_TimeIndepend_noiseless{T}) where {T <: Complex}
    δF = gradient(x->QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x, AD.times), AD.psi)[1]
    AD.psi += AD.ϵ*δF
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFI_Adam!(AD::StateOptAD_TimeIndepend_noiseless{T}) where {T <: Complex}
    δF = gradient(x->QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x, AD.times), AD.psi)[1]
    StateOpt_Adam!(AD, δF) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFIM!(AD::StateOptAD_TimeIndepend_noiseless{T}) where {T <: Complex}
    δF = gradient(x->1/(AD.W*(QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, x, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    AD.psi += AD.ϵ*δF
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFIM_Adam!(AD::StateOptAD_TimeIndepend_noiseless{T}) where {T <: Complex}
    δF = gradient(x->1/(AD.W*(QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, x, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    StateOpt_Adam!(AD, δF) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFI!(AD::StateOptAD_TimeIndepend_noiseless{T}, Measurement) where {T <: Complex}
    δI = gradient(x->CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x, AD.times), AD.psi)[1]
    AD.psi += AD.ϵ*δI
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFI_Adam!(AD::StateOptAD_TimeIndepend_noiseless{T}, Measurement) where {T <: Complex}
    δI = gradient(x->CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x, AD.times), AD.psi)[1]
    StateOpt_Adam!(AD, δI) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFIM!(AD::StateOptAD_TimeIndepend_noiseless{T}, Measurement) where {T <: Complex}
    δI = gradient(x->1/(AD.W*(CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    AD.psi += AD.ϵ*δI
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFIM_Adam!(AD::StateOptAD_TimeIndepend_noiseless{T}, Measurement) where {T <: Complex}
    δI = gradient(x->1/(AD.W*(CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    StateOpt_Adam!(AD, δI) 
    AD.psi = AD.psi/norm(AD.psi)
end

function SaveFile(dim, f_now, control)
    open("f_auto_N$(dim-1).csv","a") do f
        writedlm(f, [f_now])
    end
    open("state_auto_N$(dim-1).csv","a") do g
        writedlm(g, control)
    end
end

function AD_QFIM(AD::StateOptAD_TimeIndepend_noiseless{T}, epsilon, max_episodes, Adam, save_file) where {T <: Complex}
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
            gradient_QFI_Adam!(AD)
        else
            gradient_QFI!(AD)
        end
        if save_file == true
            if Adam == true
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
                    gradient_QFI_Adam!(AD)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(dim, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(dim, f_now, AD.psi)
                        print("current QFI is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            else
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
                    gradient_QFI!(AD)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(dim, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(dim, f_now, AD.psi)
                        print("current QFI is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
                    gradient_QFI_Adam!(AD)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(dim, f_list, AD.psi)
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
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
                    gradient_QFI!(AD)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(dim, f_list, AD.psi)
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
        println("search algorithm: Automatic Differentiation (AD)")
        F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
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
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_QFIM_Adam!(AD)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(dim, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(dim, f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            else
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_QFIM!(AD)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(dim, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(dim, f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_QFIM_Adam!(AD)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(dim, f_list, AD.psi)
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
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_QFIM!(AD)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(dim, f_list, AD.psi)
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

function AD_CFIM(M, AD::StateOptAD_TimeIndepend_noiseless{T}, epsilon, max_episodes, Adam, save_file) where {T <: Complex}
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
            gradient_CFI_Adam!(AD, M)
        else
            gradient_CFI!(AD, M)
        end
        if save_file == true
            if Adam == true
                while true
                    f_now = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
                    gradient_CFI_Adam!(AD, M)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile(dim, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(dim, f_now, AD.psi)
                        print("current CFI is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            else
                while true
                    f_now = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
                    gradient_CFI!(AD, M)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile(dim, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(dim, f_now, AD.psi)
                        print("current CFI is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    f_now = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
                    gradient_CFI_Adam!(AD, M)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile(dim, f_list, AD.psi)
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
                    f_now = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi, AD.times)
                    gradient_CFI!(AD, M)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile(dim, f_list, AD.psi)
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
        println("search algorithm: Automatic Differentiation (AD)")
        F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
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
                    F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_CFIM_Adam!(AD, M)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(dim, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(dim, f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            else
                while true
                    F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_CFIM!(AD, M)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(dim, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(dim, f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_CFIM_Adam!(AD, M)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(dim, f_list, AD.psi)
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
                    F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_CFIM!(AD, M)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(dim, f_list, AD.psi)
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

############# time-independent Hamiltonian (noise) ################
mutable struct StateOptAD_TimeIndepend_noise{T <: Complex,M <: Real}
    freeHamiltonian::Matrix{T}
    Hamiltonian_derivative::Vector{Matrix{T}}
    psi::Vector{T}
    times::Vector{M}
    Liouville_operator::Vector{Matrix{T}}
    γ::Vector{M}
    W::Matrix{M}
    mt::M
    vt::M
    ϵ::M
    beta1::M
    beta2::M
    precision::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    StateOptAD_TimeIndepend_noise(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, psi::Vector{T},
                 times::Vector{M}, Liouville_operator::Vector{Matrix{T}},γ::Vector{M}, W::Matrix{M}, mt::M, vt::M, ϵ::M, beta1::M, beta2::M, precision::M, 
                 ρ=Vector{Matrix{T}}(undef, 1), ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1),∂ρ_∂V=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = 
                 new{T,M}(freeHamiltonian, Hamiltonian_derivative, psi, times, Liouville_operator, γ, W, mt, vt, ϵ, beta1, beta2, precision, ρ, ∂ρ_∂x) 
end

function gradient_QFI!(AD::StateOptAD_TimeIndepend_noise{T}) where {T <: Complex}
    δF = gradient(x->QFIM_TimeIndepend_AD(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x*x', AD.Liouville_operator, AD.γ, AD.times), AD.psi)[1]
    AD.psi += AD.ϵ*δF
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFI_Adam!(AD::StateOptAD_TimeIndepend_noise{T}) where {T <: Complex}
    δF = gradient(x->QFIM_TimeIndepend_AD(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x*x', AD.Liouville_operator, AD.γ, AD.times), AD.psi)[1]
    StateOpt_Adam!(AD, δF) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFIM!(AD::StateOptAD_TimeIndepend_noise{T}) where {T <: Complex}
    δF = gradient(x->1/(AD.W*(QFIM_TimeIndepend_AD(AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.Liouville_operator, AD.γ, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    AD.psi += AD.ϵ*δF
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_QFIM_Adam!(AD::StateOptAD_TimeIndepend_noise{T}) where {T <: Complex}
    δF = gradient(x->1/(AD.W*(QFIM_TimeIndepend_AD(AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.Liouville_operator, AD.γ, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    StateOpt_Adam!(AD, δF) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFI!(AD::StateOptAD_TimeIndepend_noise{T}, Measurement) where {T <: Complex}
    δI = gradient(x->CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x*x', AD.Liouville_operator, AD.γ, AD.times), AD.psi)[1]
    AD.psi += AD.ϵ*δI
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFI_Adam!(AD::StateOptAD_TimeIndepend_noise{T}, Measurement) where {T <: Complex}
    δI = gradient(x->CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], x*x', AD.Liouville_operator, AD.γ, AD.times), AD.psi)[1]
    StateOpt_Adam!(AD, δI) 
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFIM!(AD::StateOptAD_TimeIndepend_noise{T}, Measurement) where {T <: Complex}
    δI = gradient(x->1/(AD.W*(CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.Liouville_operator, AD.γ, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    AD.psi += AD.ϵ*δI
    AD.psi = AD.psi/norm(AD.psi)
end

function gradient_CFIM_Adam!(AD::StateOptAD_TimeIndepend_noise{T}, Measurement) where {T <: Complex}
    δI = gradient(x->1/(AD.W*(CFIM_TimeIndepend(Measurement, AD.freeHamiltonian, AD.Hamiltonian_derivative, x*x', AD.Liouville_operator, AD.γ, AD.times) |> pinv) |> tr |>real), AD.psi) |>sum
    StateOpt_Adam!(AD, δI) 
    AD.psi = AD.psi/norm(AD.psi)
end

function SaveFile(dim, f_now, control)
    open("f_auto_N$(dim-1).csv","a") do f
        writedlm(f, [f_now])
    end
    open("state_auto_N$(dim-1).csv","a") do g
        writedlm(g, control)
    end
end

function AD_QFIM(AD::StateOptAD_TimeIndepend_noise{T}, epsilon, max_episodes, Adam, save_file) where {T <: Complex}
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
            gradient_QFI_Adam!(AD)
        else
            gradient_QFI!(AD)
        end
        if save_file == true
            if Adam == true
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    gradient_QFI_Adam!(AD)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(dim, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(dim, f_now, AD.psi)
                        print("current QFI is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            else
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    gradient_QFI!(AD)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(dim, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(dim, f_now, AD.psi)
                        print("current QFI is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    gradient_QFI_Adam!(AD)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(dim, f_list, AD.psi)
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
                    f_now = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    gradient_QFI!(AD)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final QFI is ", f_now)
                        SaveFile(dim, f_list, AD.psi)
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
        println("search algorithm: Automatic Differentiation (AD)")
        F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
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
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_QFIM_Adam!(AD)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(dim, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(dim, f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            else
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_QFIM!(AD)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(dim, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(dim, f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_QFIM_Adam!(AD)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(dim, f_list, AD.psi)
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
                    F = QFIM_TimeIndepend(AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_QFIM!(AD)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(dim, f_list, AD.psi)
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

function AD_CFIM(M, AD::StateOptAD_TimeIndepend_noise{T}, epsilon, max_episodes, Adam, save_file) where {T <: Complex}
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
            gradient_CFI_Adam!(AD, M)
        else
            gradient_CFI!(AD, M)
        end
        if save_file == true
            if Adam == true
                while true
                    f_now = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    gradient_CFI_Adam!(AD, M)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile(dim, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(dim, f_now, AD.psi)
                        print("current CFI is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            else
                while true
                    f_now = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    gradient_CFI!(AD, M)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile(dim, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(dim, f_now, AD.psi)
                        print("current CFI is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    f_now = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    gradient_CFI_Adam!(AD, M)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile(dim, f_list, AD.psi)
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
                    f_now = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    gradient_CFI!(AD, M)
                    if  abs(f_now - f_ini) < epsilon  || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final CFI is ", f_now)
                        SaveFile(dim, f_list, AD.psi)
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
        println("search algorithm: Automatic Differentiation (AD)")
        F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
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
                    F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_CFIM_Adam!(AD, M)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(dim, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(dim, f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            else
                while true
                    F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_CFIM!(AD, M)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(dim, f_now, AD.psi)
                        break
                    else
                        f_ini = f_now
                        episodes += 1
                        append!(f_list,f_now)
                        SaveFile(dim, f_now, AD.psi)
                        print("current value of Tr(WF^{-1}) is ", f_now, " ($(f_list|>length) episodes)    \r")
                    end
                end
            end
        else
            if Adam == true
                while true
                    F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_CFIM_Adam!(AD, M)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(dim, f_list, AD.psi)
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
                    F = CFIM_TimeIndepend(M, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.psi*(AD.psi)',AD.Liouville_operator, AD.γ, AD.times)
                    f_now = real(tr(AD.W*pinv(F)))
                    gradient_CFIM!(AD, M)
                    if  abs(f_now - f_ini) < epsilon || episodes > max_episodes
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of Tr(WF^{-1}) is ", f_now)
                        SaveFile(dim, f_list, AD.psi)
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
