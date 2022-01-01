########## update the coefficients of the rotation matrix ##########
mutable struct RotateCoeff_Mopt{T<:Complex, M <:Real} <:ControlSystem
    freeHamiltonian
    Hamiltonian_derivative::Vector{Matrix{T}}
    ρ0::Matrix{T}
    tspan::Vector{M}
    decay_opt::Vector{Matrix{T}}
    γ::Vector{M}
    W::Matrix{M}
    accuracy::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    RotateCoeff_Mopt(freeHamiltonian, Hamiltonian_derivative::Vector{Matrix{T}}, ρ0::Matrix{T}, tspan::Vector{M}, 
    decay_opt::Vector{Matrix{T}},γ::Vector{M}, W::Matrix{M}, accuracy::M, 
    ρ=Vector{Matrix{T}}(undef, 1), ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T<:Complex, M<:Real}=
    new{T,M}(freeHamiltonian, Hamiltonian_derivative, ρ0, tspan, decay_opt, γ, W, accuracy, ρ, ∂ρ_∂x) 
end

function CFIM_AD_Mopt(AD::RotateCoeff_Mopt{T}, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_fileseed, ) where {T<:Complex}
    sym = Symbol("CFIM_noctrl")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_RotateCoeff_AD(AD, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file, seed, sym, str1, str2)
end

function gradient_CFI!(AD::RotateCoeff_Mopt{T}, epsilon, Mbasis, Mcoeff, Lambda) where {T<:Complex}
    δI = gradient(x->CFI_AD(Mbasis, x, Lambda, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy), Mcoeff)[1]
    Mcoeff += epsilon*δI
    bound!(Mcoeff)
    Mcoeff = Mcoeff/norm(Mcoeff)
    return Mcoeff
end

function gradient_CFI_Adam!(AD::RotateCoeff_Mopt{T}, epsilon, mt, vt, beta1, beta2, Mbasis, Mcoeff, Lambda) where {T<:Complex}
    δI = gradient(x->CFI_AD(Mbasis, x, Lambda, AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy), Mcoeff)[1]
    Mcoeff = MOpt_Adam!(Mcoeff, δI, epsilon, mt, vt, beta1, beta2, AD.accuracy) 
    bound!(Mcoeff)
    Mcoeff = Mcoeff/norm(Mcoeff)
    return Mcoeff
end

function gradient_CFIM!(AD::RotateCoeff_Mopt{T}, epsilon, Mbasis, Mcoeff, Lambda) where {T<:Complex}
    δI = gradient(x->1/(AD.W*(CFIM_AD(Mbasis, x, Lambda, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy) |> pinv) |> tr |>real), Mcoeff) |>sum
    Mcoeff += epsilon*δI
    bound!(Mcoeff)
    Mcoeff = Mcoeff/norm(Mcoeff)
    return Mcoeff
end

function gradient_CFIM_Adam!(AD::RotateCoeff_Mopt{T}, epsilon, mt, vt, beta1, beta2, Mbasis, Mcoeff, Lambda) where {T<:Complex}
    δI = gradient(x->1/(AD.W*(CFIM_AD(Mbasis, x, Lambda, AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy) |> pinv) |> tr |>real), Mcoeff) |>sum
    Mcoeff = MOpt_Adam!(Mcoeff, δI, epsilon, mt, vt, beta1, beta2, AD.accuracy) 
    bound!(Mcoeff)
    Mcoeff = Mcoeff/norm(Mcoeff)
    return Mcoeff
end

function info_RotateCoeff_AD(AD, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file, seed, sym, str1, str2) where {T<:Complex}
    println("measurement optimization")
    dim = size(AD.ρ0)[1]
    M_num = length(AD.Measurement)
    suN = suN_generator(dim)
    Lambda = [Matrix{ComplexF64}(I,dim,dim)]
    append!(Lambda, [suN[i] for i in 1:length(suN)])
    Random.seed!(seed)

    ## generate a set of orthonormal basis randomly ##
    Mbasis = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
    for mi in 1:M_num
        r_ini = 2*rand(dim)-ones(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        Mbasis[mi] = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end
    ## generate a rotation matrix randomly ##
    Mcoeff = rand(dim*dim)
    Mcoeff = Mcoeff/norm(Mcoeff)
    U = rotation_matrix(Mcoeff, Lambda)
    Measurement = [U*Mbasis[i]*Mbasis[i]'*U' for i in 1:M_num]

    F_tp = obj_func(Val{sym}(), AD, Measurement)
    f_ini = real(tr(AD.W*pinv(F_tp)))
    
    f_list = [f_ini]
    F_opt = obj_func(Val{:QFIM_noctrl}(), AD, Measurement)
    f_opt= real(tr(AD.W*pinv(F_opt)))

    episodes = 1
    if length(AD.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        println("initial $str1 is $(1.0/f_ini)")
        println("QFI is $(1.0/f_opt)")

        if Adam == true
            Mcoeff = gradient_CFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, Mbasis, Mcoeff, Lambda)
        else
            Mcoeff = gradient_CFI!(AD, epsilon, Mbasis, Mcoeff, Lambda)
        end
        if save_file == true
            U = rotation_matrix(Mcoeff, Lambda)
            Measurement = [U*Mbasis[i]*Mbasis[i]'*U' for i in 1:M_num]
            SaveFile_meas(f_ini, Measurement)
            if Adam == true
                while true
                    U = rotation_matrix(Mcoeff, Lambda)
                    Measurement = [U*Mbasis[i]*Mbasis[i]'*U' for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str1 is ", f_now)
                        SaveFile_meas(f_now, Measurement)
                        break
                    else
                        episodes += 1
                        SaveFile_meas(f_now, Measurement)
                        print("current $str1 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    Mcoeff = gradient_CFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, Mbasis, Mcoeff, Lambda)
                end
            else
                while true
                    U = rotation_matrix(Mcoeff, Lambda)
                    Measurement = [U*Mbasis[i]*Mbasis[i]'*U' for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str1 is ", f_now)
                        SaveFile_meas(f_now, Measurement)
                        break
                    else
                        episodes += 1
                        SaveFile_meas(f_now, Measurement)
                        print("current $str1 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    Mcoeff = gradient_CFI!(AD, epsilon, Mbasis, Mcoeff, Lambda)
                end
            end
        else
            if Adam == true
                while true
                    U = rotation_matrix(Mcoeff, Lambda)
                    Measurement = [U*Mbasis[i]*Mbasis[i]'*U' for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str1 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_meas(f_list, Measurement)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str1 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    Mcoeff = gradient_CFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, Mbasis, Mcoeff, Lambda)
                end
            else
                while true
                    U = rotation_matrix(Mcoeff, Lambda)
                    Measurement = [U*Mbasis[i]*Mbasis[i]'*U' for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str1 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_meas(f_list, Measurement)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str1 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    Mcoeff = gradient_CFI!(AD, epsilon, Mbasis, Mcoeff, Lambda)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        println("initial value of $str2 is $(f_ini)")
        println("Tr(WF^{-1}) is $(f_opt)")

        if Adam == true
            Mcoeff = gradient_CFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, Mbasis, Mcoeff, Lambda)
        else
            Mcoeff = gradient_CFIM!(AD, epsilon, Mbasis, Mcoeff, Lambda)
        end
        if save_file == true
            U = rotation_matrix(Mcoeff, Lambda)
            Measurement = [U*Mbasis[i]*Mbasis[i]'*U' for i in 1:M_num]
            SaveFile_meas(f_ini, Measurement)
            if Adam == true
                while true
                    U = rotation_matrix(Mcoeff, Lambda)
                    Measurement = [U*Mbasis[i]*Mbasis[i]'*U' for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str2 is ", f_now)
                        SaveFile_meas(f_now, Measurement)
                        break
                    else
                        episodes += 1
                        SaveFile_meas(f_now, Measurement)
                        print("current value of $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    Mcoeff = gradient_CFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, Mbasis, Mcoeff, Lambda)
                end
            else
                while true
                    U = rotation_matrix(Mcoeff, Lambda)
                    Measurement = [U*Mbasis[i]*Mbasis[i]'*U' for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str2 is ", f_now)
                        SaveFile_meas(f_now, Measurement)
                        break
                    else
                        episodes += 1
                        SaveFile_meas(f_now, Measurement)
                        print("current value of $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    Mcoeff = gradient_CFIM!(AD, epsilon, Mbasis, Mcoeff, Lambda)
                end
            end
        else
            if Adam == true
                while true
                    U = rotation_matrix(Mcoeff, Lambda)
                    Measurement = [U*Mbasis[i]*Mbasis[i]'*U' for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_meas(f_list, Measurement)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    Mcoeff = gradient_CFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, Mbasis, Mcoeff, Lambda)
                end
            else
                while true
                    U = rotation_matrix(Mcoeff, Lambda)
                    Measurement = [U*Mbasis[i]*Mbasis[i]'*U' for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_meas(f_list, Measurement)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    Mcoeff = gradient_CFIM!(AD, epsilon, Mbasis, Mcoeff, Lambda)
                end
            end
        end
    end
end

function rotation_matrix(coeff, Lambda)
    dim = size(Lambda[1])[1]
    U = Matrix{ComplexF64}(I,dim,dim)
    for i in 1:length(Lambda)
        U = U*exp(1.0im*coeff[i]*Lambda[i])
    end
    U
end


################ projection measurement ###############
function CFIM_AD_Mopt(AD::projection_Mopt{T}, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file) where {T<:Complex}
    sym = Symbol("CFIM_noctrl")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_projection_AD(AD, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file, sym, str1, str2)
end

function gradient_CFI!(AD::projection_Mopt{T}, epsilon) where {T<:Complex}
    M_num = length(AD.Measurement)
    δI = gradient(x->CFI([x[i]*x[i]' for i in 1:M_num], AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy), AD.Measurement)[1]
    AD.Measurement += epsilon*δI
    AD.Measurement = gramschmidt(AD.Measurement)
end

function gradient_CFI_Adam!(AD::projection_Mopt{T}, epsilon, mt, vt, beta1, beta2) where {T<:Complex}
    M_num = length(AD.Measurement)
    δI = gradient(x->CFI([x[i]*x[i]' for i in 1:M_num], AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy), AD.Measurement)[1]
    MOpt_Adam!(AD, δI, epsilon, mt, vt, beta1, beta2, AD.accuracy)
    AD.Measurement = gramschmidt(AD.Measurement)
end

function gradient_CFIM!(AD::projection_Mopt{T}, epsilon) where {T<:Complex}
    M_num = length(AD.Measurement)
    δI = gradient(x->1/(AD.W*(CFIM([x[i]*x[i]' for i in 1:M_num], AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy) |> pinv) |> tr |>real), AD.Measurement) |>sum
    AD.Measurement += epsilon*δI
    AD.Measurement = gramschmidt(AD.Measurement)
end

function gradient_CFIM_Adam!(AD::projection_Mopt{T}, epsilon, mt, vt, beta1, beta2) where {T<:Complex}
    M_num = length(AD.Measurement)
    δI = gradient(x->1/(AD.W*(CFIM([x[i]*x[i]' for i in 1:M_num], AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy) |> pinv) |> tr |>real), AD.Measurement) |>sum
    MOpt_Adam!(AD, δI, epsilon, mt, vt, beta1, beta2, AD.accuracy)
    AD.Measurement = gramschmidt(AD.Measurement)
end

function info_projection_AD(AD, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file, sym, str1, str2) where {T<:Complex}
    println("measurement optimization")
    dim = size(AD.ρ0)[1]
    M_num = length(AD.Measurement)

    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
    F_tp = obj_func(Val{sym}(), AD, Measurement)
    f_ini = real(tr(AD.W*pinv(F_tp)))
    
    f_list = [f_ini]
    F_opt = obj_func(Val{:QFIM_noctrl}(), AD, Measurement)
    f_opt= real(tr(AD.W*pinv(F_opt)))

    episodes = 1
    if length(AD.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        println("initial $str1 is $(1.0/f_ini)")
        println("QFI is $(1.0/f_opt)")

        if Adam == true
            gradient_CFI_Adam!(AD, epsilon, mt, vt, beta1, beta2)
        else
            gradient_CFI!(AD, epsilon)
        end
        if save_file == true
            Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
            SaveFile_meas(f_ini, Measurement)
            if Adam == true
                while true
                    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str1 is ", f_now)
                        SaveFile_meas(f_now, Measurement)
                        break
                    else
                        episodes += 1
                        SaveFile_meas(f_now, Measurement)
                        print("current $str1 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI_Adam!(AD, epsilon, mt, vt, beta1, beta2)
                end
            else
                while true
                    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str1 is ", f_now)
                        SaveFile_meas(f_now, Measurement)
                        break
                    else
                        episodes += 1
                        SaveFile_meas(f_now, Measurement)
                        print("current $str1 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI!(AD, epsilon)
                end
            end
        else
            if Adam == true
                while true
                    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str1 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_meas(f_list, Measurement)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str1 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI_Adam!(AD, epsilon, mt, vt, beta1, beta2)
                end
            else
                while true
                    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str1 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_meas(f_list, Measurement)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str1 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFI!(AD, epsilon)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        println("initial value of $str2 is $(f_ini)")
        println("Tr(WF^{-1}) is $(f_opt)")

        if Adam == true
            gradient_CFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2)
        else
            gradient_CFIM!(AD, epsilon)
        end
        if save_file == true
            Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
            SaveFile_meas(f_ini, Measurement)
            if Adam == true
                while true
                    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str2 is ", f_now)
                        SaveFile_meas(f_now, Measurement)
                        break
                    else
                        episodes += 1
                        SaveFile_meas(f_now, Measurement)
                        print("current value of $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2)
                end
            else
                while true
                    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str2 is ", f_now)
                        SaveFile_meas(f_now, Measurement)
                        break
                    else
                        episodes += 1
                        SaveFile_meas(f_now, Measurement)
                        print("current value of $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM!(AD, epsilon)
                end
            end
        else
            if Adam == true
                while true
                    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_meas(f_list, Measurement)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2)
                end
            else
                while true
                    Measurement = [AD.Measurement[i]*(AD.Measurement[i])' for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_meas(f_list, Measurement)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    gradient_CFIM!(AD, epsilon)
                end
            end
        end
    end
end


################ update the coefficients according to the given basis ###############
function CFIM_AD_Mopt(AD::givenpovm_Mopt{T}, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file, seed) where {T<:Complex}
    sym = Symbol("CFIM_noctrl")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_givenpovm_AD(AD, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file, seed, sym, str1, str2)
end

function gradient_CFI!(AD::givenpovm_Mopt{T}, epsilon, coeff, POVM_basis, M_num, dim) where {T<:Complex}
    δI = gradient(x->CFI([sum([x[i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num], AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy), coeff)[1]
    coeff += epsilon*δI
    bound!(coeff)
    return coeff/norm(coeff)
end

function gradient_CFI_Adam!(AD::givenpovm_Mopt{T}, epsilon, mt, vt, beta1, beta2, coeff, POVM_basis, M_num, dim) where {T<:Complex}
    δI = gradient(x->CFI([sum([x[i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num], AD.freeHamiltonian, AD.Hamiltonian_derivative[1], AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy), coeff)[1]
    coeff = MOpt_Adam!(coeff, δI, epsilon, mt, vt, beta1, beta2, AD.accuracy)
    bound!(coeff)
    return coeff/norm(coeff)
end

function gradient_CFIM!(AD::givenpovm_Mopt{T}, epsilon, coeff, POVM_basis, M_num, dim) where {T<:Complex}
    δI = gradient(x->1/(AD.W*(CFIM([sum([x[i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num], AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy) |> pinv) |> tr |>real), coeff) |>sum
    coeff += epsilon*δI
    bound!(coeff)
    return coeff/norm(coeff)
end

function gradient_CFIM_Adam!(AD::givenpovm_Mopt{T}, epsilon, mt, vt, beta1, beta2, coeff, POVM_basis, M_num, dim) where {T<:Complex}
    δI = gradient(x->1/(AD.W*(CFIM([sum([x[i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num], AD.freeHamiltonian, AD.Hamiltonian_derivative, AD.ρ0, AD.decay_opt, AD.γ, AD.tspan, AD.accuracy) |> pinv) |> tr |>real), coeff) |>sum
    coeff = MOpt_Adam!(coeff, δI, epsilon, mt, vt, beta1, beta2, AD.accuracy)
    bound!(coeff)
    return coeff/norm(coeff)
end

function info_givenpovm_AD(AD, mt, vt, epsilon, beta1, beta2, max_episode, Adam, save_file, seed, sym, str1, str2) where {T<:Complex}
    println("measurement optimization")
    Random.seed!(seed)
    dim = size(AD.ρ0)[1]
    M_num = AD.M_num
    POVM_basis = AD.povm_basis
    # initialize 
    coeff = generate_coeff(M_num, dim)

    Measurement = [sum([coeff[i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
    F_tp = obj_func(Val{sym}(), AD, Measurement)
    f_ini = real(tr(AD.W*pinv(F_tp)))
    
    f_list = [f_ini]
    F_opt = obj_func(Val{:QFIM_noctrl}(), AD, Measurement)
    f_opt= real(tr(AD.W*pinv(F_opt)))

    episodes = 1
    if length(AD.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        println("initial $str1 is $(1.0/f_ini)")
        println("QFI is $(1.0/f_opt)")

        if Adam == true
            coeff = gradient_CFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, coeff, POVM_basis, M_num, dim)
        else
            coeff = gradient_CFI!(AD, epsilon, coeff, POVM_basis, M_num, dim)
        end
        if save_file == true
            Measurement = [sum([coeff[i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
            SaveFile_meas(f_ini, Measurement)
            if Adam == true
                while true
                    Measurement = [sum([coeff[i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str1 is ", f_now)
                        SaveFile_meas(f_now, Measurement)
                        break
                    else
                        episodes += 1
                        SaveFile_meas(f_now, Measurement)
                        print("current $str1 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    coeff = gradient_CFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, coeff, POVM_basis, M_num, dim)
                end
            else
                while true
                    Measurement = [sum([coeff[i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str1 is ", f_now)
                        SaveFile_meas(f_now, Measurement)
                        break
                    else
                        episodes += 1
                        SaveFile_meas(f_now, Measurement)
                        print("current $str1 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    coeff = gradient_CFI!(AD, epsilon, coeff, POVM_basis, M_num, dim)
                end
            end
        else
            if Adam == true
                while true
                    Measurement = [sum([coeff[i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str1 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_meas(f_list, Measurement)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str1 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    coeff = gradient_CFI_Adam!(AD, epsilon, mt, vt, beta1, beta2, coeff, POVM_basis, M_num, dim)
                end
            else
                while true
                    Measurement = [sum([coeff[i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = 1.0/real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final $str1 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_meas(f_list, Measurement)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current $str1 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    coeff = gradient_CFI!(AD, epsilon, coeff, POVM_basis, M_num, dim)
                end
            end
        end
    else
        println("multiparameter scenario")
        println("search algorithm: Automatic Differentiation (AD)")
        println("initial value of $str2 is $(f_ini)")
        println("Tr(WF^{-1}) is $(f_opt)")

        if Adam == true
            coeff = gradient_CFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, coeff, POVM_basis, M_num, dim)
        else
            coeff = gradient_CFIM!(AD, epsilon, coeff, POVM_basis, M_num, dim)
        end
        if save_file == true
            Measurement = [sum([coeff[i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
            SaveFile_meas(f_ini, Measurement)
            if Adam == true
                while true
                    Measurement = [sum([coeff[i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str2 is ", f_now)
                        SaveFile_meas(f_now, Measurement)
                        break
                    else
                        episodes += 1
                        SaveFile_meas(f_now, Measurement)
                        print("current value of $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    coeff = gradient_CFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, coeff, POVM_basis, M_num, dim)
                end
            else
                while true
                    Measurement = [sum([coeff[i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str2 is ", f_now)
                        SaveFile_meas(f_now, Measurement)
                        break
                    else
                        episodes += 1
                        SaveFile_meas(f_now, Measurement)
                        print("current value of $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    coeff = gradient_CFIM!(AD, epsilon, coeff, POVM_basis, M_num, dim)
                end
            end
        else
            if Adam == true
                while true
                    Measurement = [sum([coeff[i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_meas(f_list, Measurement)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    coeff = gradient_CFIM_Adam!(AD, epsilon, mt, vt, beta1, beta2, coeff, POVM_basis, M_num, dim)
                end
            else
                while true
                    Measurement = [sum([coeff[i][j]*POVM_basis[j] for j in 1:dim^2]) for i in 1:M_num]
                    F_tp = obj_func(Val{sym}(), AD, Measurement)
                    f_now = real(tr(AD.W*pinv(F_tp)))
                    if  episodes >= max_episode
                        print("\e[2K")
                        println("Iteration over, data saved.")
                        println("Final value of $str2 is ", f_now)
                        append!(f_list, f_now)
                        SaveFile_meas(f_list, Measurement)
                        break
                    else
                        episodes += 1
                        append!(f_list, f_now)
                        print("current value of $str2 is ", f_now, " ($(episodes) episodes)    \r")
                    end
                    coeff = gradient_CFIM!(AD, epsilon, coeff, POVM_basis, M_num, dim)
                end
            end
        end
    end
end
