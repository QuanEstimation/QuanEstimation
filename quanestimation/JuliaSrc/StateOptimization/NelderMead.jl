mutable struct NelderMead{T <: Complex,M <: Real}
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
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    NelderMead(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, psi::Vector{T},
             times::Vector{M}, Liouville_operator::Vector{Matrix{T}},γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}},
             control_coefficients::Vector{Vector{M}}, ctrl_bound::M, W::Matrix{M}, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = new{T,M}(freeHamiltonian, 
                Hamiltonian_derivative, psi, times, Liouville_operator, γ, control_Hamiltonian, control_coefficients, ctrl_bound, W, ρ, ∂ρ_∂x) 
end

function NelderMead_QFI(NM::NelderMead{T}, state_num, ini_state, coeff_r, coeff_e, coeff_c, coeff_s, epsilon, max_episodes, seed, save_file) where {T<: Complex}
    println("state optimization")
    println("single parameter scenario")
    println("control algorithm: Nelder Mead")

    Random.seed!(seed)
    dim = length(NM.psi)
    nelder_mead = repeat(NM, state_num)

    # initialize 
    for pj in 1:length(ini_state)
        nelder_mead[pj].psi = [ini_state[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_state)+1):(state_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        nelder_mead[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    p_fit = [0.0 for i in 1:state_num] 
    for pj in 1:state_num
        rho_ini = nelder_mead[pj].psi*(nelder_mead[pj].psi)'
        p_fit[pj] = -QFI_ori(NM.freeHamiltonian, NM.Hamiltonian_derivative[1], rho_ini, NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, 
                             NM.control_coefficients, NM.times)
    end
    sort_ind = sortperm(p_fit)

    f_ini = QFI_ori(NM.freeHamiltonian, NM.Hamiltonian_derivative[1], NM.psi*(NM.psi)', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, 
                    NM.control_coefficients, NM.times)
    f_list = [f_ini]
    println("initial QFI is $(f_ini)")

    Tend = NM.times[end]
    episodes = 1
    if save_file == true
        while true
            p_fit, sort_ind = neldermead_train_QFI(nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[end]]-p_fit[sort_ind[1]] < epsilon || episodes > max_episodes
                open("state_nm_T$Tend.csv","w") do g
                    writedlm(g, nelder_mead[sort_ind[1]].psi)
                end
                open("f_nm_T$Tend.csv","w") do h
                    writedlm(h, f_list)
                end
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final QFI is ", -minimum(p_fit))
                break
            else
                episodes += 1
                append!(f_list,-minimum(p_fit))
                open("state_nm_T$Tend.csv","w") do g
                    writedlm(g, nelder_mead[sort_ind[1]].psi)
                end
                open("f_nm_T$Tend.csv","w") do h
                    writedlm(h, f_list)
                end
                print("current QFI is ", -minimum(p_fit), " ($(f_list|>length) episodes)    \r")
            end
        end
    else
        while true
            p_fit, sort_ind = neldermead_train_QFI(nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[end]]-p_fit[sort_ind[1]] < epsilon || episodes > max_episodes
                open("state_nm_T$Tend.csv","w") do g
                    writedlm(g, nelder_mead[sort_ind[1]].psi)
                end
                open("f_nm_T$Tend.csv","w") do h
                    writedlm(h, f_list)
                end
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final QFI is ", -minimum(p_fit))
                break
            else
                episodes += 1
                append!(f_list,-minimum(p_fit))
                print("current QFI is ", -minimum(p_fit), " ($(f_list|>length) episodes)    \r")
            end
        end
    end
end

function NelderMead_QFIM(NM::NelderMead{T}, state_num, ini_state, coeff_r, coeff_e, coeff_c, coeff_s, epsilon, max_episodes, seed, save_file) where {T<: Complex}
    println("state optimization")
    println("multiparameter scenario")
    println("control algorithm: Nelder Mead")

    Random.seed!(seed)
    dim = length(NM.psi)
    nelder_mead = repeat(NM, state_num)

    # initialize 
    for pj in 1:length(ini_state)
        nelder_mead[pj].psi = [ini_state[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_state)+1):(state_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        nelder_mead[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    p_fit = [0.0 for i in 1:state_num] 
    for pj in 1:state_num
        rho_ini = nelder_mead[pj].psi*(nelder_mead[pj].psi)'
        F_tp = QFIM_ori(NM.freeHamiltonian, NM.Hamiltonian_derivative, rho_ini, NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, 
                             NM.control_coefficients, NM.times)
        p_fit[pj] = -1.0/real(tr(NM.W*pinv(F_tp)))
    end
    sort_ind = sortperm(p_fit)

    F = QFIM_ori(NM.freeHamiltonian, NM.Hamiltonian_derivative, NM.psi*(NM.psi)', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, 
                    NM.control_coefficients, NM.times)
    f_ini = real(tr(NM.W*pinv(F)))
    f_list = [f_ini]
    println("initial value of Tr(WF^{-1}) is $(f_ini)")

    Tend = NM.times[end]
    episodes = 1
    if save_file == true
        while true
            p_fit, sort_ind = neldermead_train_QFIM(nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[end]]-p_fit[sort_ind[1]] < epsilon || episodes > max_episodes
                open("state_nm_T$Tend.csv","w") do g
                    writedlm(g, nelder_mead[sort_ind[1]].psi)
                end
                open("f_nm_T$Tend.csv","w") do h
                    writedlm(h, f_list)
                end
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final value of Tr(WF^{-1}) is ", -1.0/minimum(p_fit))
                break
            else
                episodes += 1
                append!(f_list,-1.0/minimum(p_fit))
                open("state_nm_T$Tend.csv","w") do g
                    writedlm(g, nelder_mead[sort_ind[1]].psi)
                end
                open("f_nm_T$Tend.csv","w") do h
                    writedlm(h, f_list)
                end
                print("current value of Tr(WF^{-1}) is ", -1.0/minimum(p_fit), " ($(f_list|>length) episodes)    \r")
            end
        end
    else
        while true
            p_fit, sort_ind = neldermead_train_QFIM(nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[end]]-p_fit[sort_ind[1]] < epsilon || episodes > max_episodes
                open("state_nm_T$Tend.csv","w") do g
                    writedlm(g, nelder_mead[sort_ind[1]].psi)
                end
                open("f_nm_T$Tend.csv","w") do h
                    writedlm(h, f_list)
                end
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final value of Tr(WF^{-1}) is ", -1.0/minimum(p_fit))
                break
            else
                episodes += 1
                append!(f_list,-1.0/minimum(p_fit))
                print("current value of Tr(WF^{-1}) is ", -1.0/minimum(p_fit), " ($(f_list|>length) episodes)    \r")
            end
        end
    end
end

function NelderMead_CFI(M, NM::NelderMead{T}, state_num, ini_state, coeff_r, coeff_e, coeff_c, coeff_s, epsilon, max_episodes, seed, save_file) where {T<: Complex}
    println("state optimization")
    println("single parameter scenario")
    println("control algorithm: Nelder Mead")

    Random.seed!(seed)
    dim = length(NM.psi)
    nelder_mead = repeat(NM, state_num)

    # initialize 
    for pj in 1:length(ini_state)
        nelder_mead[pj].psi = [ini_state[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_state)+1):(state_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        nelder_mead[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    p_fit = [0.0 for i in 1:state_num] 
    for pj in 1:state_num
        rho_ini = nelder_mead[pj].psi*(nelder_mead[pj].psi)'
        p_fit[pj] = -CFI(M, NM.freeHamiltonian, NM.Hamiltonian_derivative[1], rho_ini, NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, 
                             NM.control_coefficients, NM.times)
    end
    sort_ind = sortperm(p_fit)

    f_ini = CFI(M, NM.freeHamiltonian, NM.Hamiltonian_derivative[1], NM.psi*(NM.psi)', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, 
                    NM.control_coefficients, NM.times)
    f_list = [f_ini]
    println("initial CFI is $(f_ini)")

    Tend = NM.times[end]
    episodes = 1
    if save_file == true
        while true
            p_fit, sort_ind = neldermead_train_CFI(M, nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[end]]-p_fit[sort_ind[1]] < epsilon || episodes > max_episodes
                open("state_nm_T$Tend.csv","w") do g
                    writedlm(g, nelder_mead[sort_ind[1]].psi)
                end
                open("f_nm_T$Tend.csv","w") do h
                    writedlm(h, f_list)
                end
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final CFI is ", -minimum(p_fit))
                break
            else
                episodes += 1
                append!(f_list,-minimum(p_fit))
                open("state_nm_T$Tend.csv","w") do g
                    writedlm(g, nelder_mead[sort_ind[1]].psi)
                end
                open("f_nm_T$Tend.csv","w") do h
                    writedlm(h, f_list)
                end
                print("current CFI is ", -minimum(p_fit), " ($(f_list|>length) episodes)    \r")
            end
        end
    else
        while true
            p_fit, sort_ind = neldermead_train_CFI(M, nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[end]]-p_fit[sort_ind[1]] < epsilon || episodes > max_episodes
                open("state_nm_T$Tend.csv","w") do g
                    writedlm(g, nelder_mead[sort_ind[1]].psi)
                end
                open("f_nm_T$Tend.csv","w") do h
                    writedlm(h, f_list)
                end
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final CFI is ", -minimum(p_fit))
                break
            else
                episodes += 1
                append!(f_list,-minimum(p_fit))
                print("current CFI is ", -minimum(p_fit), " ($(f_list|>length) episodes)    \r")
            end
        end
    end
end


function NelderMead_CFIM(M, NM::NelderMead{T}, state_num, ini_state, coeff_r, coeff_e, coeff_c, coeff_s, epsilon, max_episodes, seed, save_file) where {T<: Complex}
    println("state optimization")
    println("multiparameter scenario")
    println("control algorithm: Nelder Mead")

    Random.seed!(seed)
    dim = length(NM.psi)
    nelder_mead = repeat(NM, state_num)

    # initialize 
    for pj in 1:length(ini_state)
        nelder_mead[pj].psi = [ini_state[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_state)+1):(state_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        nelder_mead[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    p_fit = [0.0 for i in 1:state_num] 
    for pj in 1:state_num
        rho_ini = nelder_mead[pj].psi*(nelder_mead[pj].psi)'
        F_tp = CFIM(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, rho_ini, NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, 
                             NM.control_coefficients, NM.times)
        p_fit[pj] = -1.0/real(tr(NM.W*pinv(F_tp)))
    end
    sort_ind = sortperm(p_fit)

    F = CFIM(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, NM.psi*(NM.psi)', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, 
                    NM.control_coefficients, NM.times)
    f_ini = real(tr(NM.W*pinv(F)))
    f_list = [f_ini]
    println("initial value of Tr(WF^{-1}) is $(f_ini)")

    Tend = NM.times[end]
    episodes = 1
    if save_file == true
        while true
            p_fit, sort_ind = neldermead_train_CFIM(M, nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[end]]-p_fit[sort_ind[1]] < epsilon || episodes > max_episodes
                open("state_nm_T$Tend.csv","w") do g
                    writedlm(g, nelder_mead[sort_ind[1]].psi)
                end
                open("f_nm_T$Tend.csv","w") do h
                    writedlm(h, f_list)
                end
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final value of Tr(WF^{-1}) is ", -1.0/minimum(p_fit))
                break
            else
                episodes += 1
                append!(f_list,-1.0/minimum(p_fit))
                open("state_nm_T$Tend.csv","w") do g
                    writedlm(g, nelder_mead[sort_ind[1]].psi)
                end
                open("f_nm_T$Tend.csv","w") do h
                    writedlm(h, f_list)
                end
                print("current value of Tr(WF^{-1}) is ", -1.0/minimum(p_fit), " ($(f_list|>length) episodes)    \r")
            end
        end
    else
        while true
            p_fit, sort_ind = neldermead_train_CFIM(M, nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[end]]-p_fit[sort_ind[1]] < epsilon || episodes > max_episodes
                open("state_nm_T$Tend.csv","w") do g
                    writedlm(g, nelder_mead[sort_ind[1]].psi)
                end
                open("f_nm_T$Tend.csv","w") do h
                    writedlm(h, f_list)
                end
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final value of Tr(WF^{-1}) is ", -1.0/minimum(p_fit))
                break
            else
                episodes += 1
                append!(f_list,-1.0/minimum(p_fit))
                print("current value of Tr(WF^{-1}) is ", -1.0/minimum(p_fit), " ($(f_list|>length) episodes)    \r")
            end
        end
    end
end

function neldermead_train_QFI(nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
    # calculate the average vector
    vec_ave = zeros(ComplexF64, dim)
    for ni in 1:dim
        vec_ave[ni] = [nelder_mead[pk].psi[ni] for pk in 1:state_num] |>sum
        vec_ave[ni] = vec_ave[ni]/state_num
    end
    vec_ave = vec_ave/norm(vec_ave)

    # reflection
    vec_ref = zeros(ComplexF64, dim)
    for nj in 1:dim
        vec_ref[nj] = vec_ave[nj] + coeff_r*(vec_ave[nj]-nelder_mead[sort_ind[end]].psi[nj])
    end
    vec_ref = vec_ref/norm(vec_ref)

    fr = -QFI_ori(NM.freeHamiltonian, NM.Hamiltonian_derivative[1], vec_ref*vec_ref', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)

    if fr < p_fit[sort_ind[1]]
        # expansion
        vec_exp = zeros(ComplexF64, dim)
        for nk in 1:dim
            vec_exp[nk] = vec_ave[nk] + coeff_e*(vec_ref[nk]-vec_ave[nk])
        end
        vec_exp = vec_exp/norm(vec_exp)

        fe = -QFI_ori(NM.freeHamiltonian, NM.Hamiltonian_derivative[1], vec_exp*vec_exp', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)

        if fe >= fr
            for np in 1:dim
                nelder_mead[sort_ind[end]].psi[np] = vec_ref[np]
            end
            p_fit[sort_ind[end]] = fr
            sort_ind = sortperm(p_fit)
        else
            for np in 1:dim
                nelder_mead[sort_ind[end]].psi[np] = vec_exp[np]
            end
            p_fit[sort_ind[end]] = fe
            sort_ind = sortperm(p_fit)
        end
    elseif fr > p_fit[sort_ind[end-1]]
        # constraction
        if fr >= p_fit[sort_ind[end]]
            # inside constraction
            vec_ic = zeros(ComplexF64, dim)
            for nl in 1:dim
                vec_ic[nl] = vec_ave[nl] - coeff_c*(vec_ave[nl]-nelder_mead[sort_ind[end]].psi[nl])
            end
            vec_ic = vec_ic/norm(vec_ic)

            fic = -QFI_ori(NM.freeHamiltonian, NM.Hamiltonian_derivative[1], vec_ic*vec_ic', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
        
            if fic < p_fit[sort_ind[end]]
                for np in 1:dim
                    nelder_mead[sort_ind[end]].psi[np] = vec_ic[np]
                end
                p_fit[sort_ind[end]] = fic
                sort_ind = sortperm(p_fit)
            else
                # shrink
                vec_first = [nelder_mead[sort_ind[1]].psi[i] for i in 1:dim]
                for pk in 1:state_num
                    for nq in 1:dim
                        nelder_mead[pk].psi[nq] = vec_first[nq] + coeff_s*(nelder_mead[pk].psi[nq]-vec_first[nq])
                    end
                    nelder_mead[pk].psi = nelder_mead[pk].psi/norm(nelder_mead[pk].psi)

                    rho = nelder_mead[pk].psi*(nelder_mead[pk].psi)'
                    p_fit[pk] = -QFI_ori(NM.freeHamiltonian, NM.Hamiltonian_derivative[1], rho, NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
                end
                sort_ind = sortperm(p_fit)
            end
        else
            # outside constraction
            vec_oc = zeros(ComplexF64, dim)
            for nn in 1:dim
                vec_oc[nn] = vec_ave[nn] + coeff_c*(vec_ref[nn]-vec_ave[nn])
            end
            vec_oc = vec_oc/norm(vec_oc)

            foc = -QFI_ori(NM.freeHamiltonian, NM.Hamiltonian_derivative[1], vec_oc*vec_oc', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)

            if foc <= fr
                for np in 1:dim
                    nelder_mead[sort_ind[end]].psi[np] = vec_oc[np]
                end
                p_fit[sort_ind[end]] = foc
                sort_ind = sortperm(p_fit)
            else
                # shrink
                vec_first = [nelder_mead[sort_ind[1]].psi[i] for i in 1:dim]
                for pk in 1:state_num
                    for nq in 1:dim
                        nelder_mead[pk].psi[nq] = vec_first[nq] + coeff_s*(nelder_mead[pk].psi[nq]-vec_first[nq])
                    end
                    nelder_mead[pk].psi = nelder_mead[pk].psi/norm(nelder_mead[pk].psi)

                    rho = nelder_mead[pk].psi*(nelder_mead[pk].psi)'
                    p_fit[pk] = -QFI_ori(NM.freeHamiltonian, NM.Hamiltonian_derivative[1], rho, NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
                end
                sort_ind = sortperm(p_fit)
            end
        end
    else
        for np in 1:dim
            nelder_mead[sort_ind[end]].psi[np] = vec_ref[np]
        end
        p_fit[sort_ind[end]] = fr
        sort_ind = sortperm(p_fit)
    end
    return p_fit, sort_ind
end

function neldermead_train_QFIM(nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
    # calculate the average vector
    vec_ave = zeros(ComplexF64, dim)
    for ni in 1:dim
        vec_ave[ni] = [nelder_mead[pk].psi[ni] for pk in 1:state_num] |>sum
        vec_ave[ni] = vec_ave[ni]/state_num
    end
    vec_ave = vec_ave/norm(vec_ave)

    # reflection
    vec_ref = zeros(ComplexF64, dim)
    for nj in 1:dim
        vec_ref[nj] = vec_ave[nj] + coeff_r*(vec_ave[nj]-nelder_mead[sort_ind[end]].psi[nj])
    end
    vec_ref = vec_ref/norm(vec_ref)

    F_r = QFIM_ori(NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_ref*vec_ref', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
    fr = -1.0/real(tr(NM.W*pinv(F_r)))

    if fr < p_fit[sort_ind[1]]
        # expansion
        vec_exp = zeros(ComplexF64, dim)
        for nk in 1:dim
            vec_exp[nk] = vec_ave[nk] + coeff_e*(vec_ref[nk]-vec_ave[nk])
        end
        vec_exp = vec_exp/norm(vec_exp)

        F_e = QFIM_ori(NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_exp*vec_exp', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
        fe = -1.0/real(tr(NM.W*pinv(F_e)))

        if fe >= fr
            for np in 1:dim
                nelder_mead[sort_ind[end]].psi[np] = vec_ref[np]
            end
            p_fit[sort_ind[end]] = fr
            sort_ind = sortperm(p_fit)
        else
            for np in 1:dim
                nelder_mead[sort_ind[end]].psi[np] = vec_exp[np]
            end
            p_fit[sort_ind[end]] = fe
            sort_ind = sortperm(p_fit)
        end
    elseif fr > p_fit[sort_ind[end-1]]
        # constraction
        if fr >= p_fit[sort_ind[end]]
            # inside constraction
            vec_ic = zeros(ComplexF64, dim)
            for nl in 1:dim
                vec_ic[nl] = vec_ave[nl] - coeff_c*(vec_ave[nl]-nelder_mead[sort_ind[end]].psi[nl])
            end
            vec_ic = vec_ic/norm(vec_ic)

            F_ic = QFIM_ori(NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_ic*vec_ic', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
            fic = -1.0/real(tr(NM.W*pinv(F_ic)))

            if fic < p_fit[sort_ind[end]]
                for np in 1:dim
                    nelder_mead[sort_ind[end]].psi[np] = vec_ic[np]
                end
                p_fit[sort_ind[end]] = fic
                sort_ind = sortperm(p_fit)
            else
                # shrink
                vec_first = [nelder_mead[sort_ind[1]].psi[i] for i in 1:dim]
                for pk in 1:state_num
                    for nq in 1:dim
                        nelder_mead[pk].psi[nq] = vec_first[nq] + coeff_s*(nelder_mead[pk].psi[nq]-vec_first[nq])
                    end
                    nelder_mead[pk].psi = nelder_mead[pk].psi/norm(nelder_mead[pk].psi)

                    rho = nelder_mead[pk].psi*(nelder_mead[pk].psi)'
                    F_tp = QFIM_ori(NM.freeHamiltonian, NM.Hamiltonian_derivative, rho, NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
                    p_fit[pk] = -1.0/real(tr(NM.W*pinv(F_tp)))
                end
                sort_ind = sortperm(p_fit)
            end
        else
            # outside constraction
            vec_oc = zeros(ComplexF64, dim)
            for nn in 1:dim
                vec_oc[nn] = vec_ave[nn] + coeff_c*(vec_ref[nn]-vec_ave[nn])
            end
            vec_oc = vec_oc/norm(vec_oc)

            F_oc = QFIM_ori(NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_oc*vec_oc', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
            foc = -1.0/real(tr(NM.W*pinv(F_oc)))

            if foc <= fr
                for np in 1:dim
                    nelder_mead[sort_ind[end]].psi[np] = vec_oc[np]
                end
                p_fit[sort_ind[end]] = foc
                sort_ind = sortperm(p_fit)
            else
                # shrink
                vec_first = [nelder_mead[sort_ind[1]].psi[i] for i in 1:dim]
                for pk in 1:state_num
                    for nq in 1:dim
                        nelder_mead[pk].psi[nq] = vec_first[nq] + coeff_s*(nelder_mead[pk].psi[nq]-vec_first[nq])
                    end
                    nelder_mead[pk].psi = nelder_mead[pk].psi/norm(nelder_mead[pk].psi)

                    rho = nelder_mead[pk].psi*(nelder_mead[pk].psi)'
                    F_tp = QFIM_ori(NM.freeHamiltonian, NM.Hamiltonian_derivative, rho, NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
                    p_fit[pk] = -1.0/real(tr(NM.W*pinv(F_tp)))
                end
                sort_ind = sortperm(p_fit)
            end
        end
    else
        for np in 1:dim
            nelder_mead[sort_ind[end]].psi[np] = vec_ref[np]
        end
        p_fit[sort_ind[end]] = fr
        sort_ind = sortperm(p_fit)
    end
    return p_fit, sort_ind
end

function neldermead_train_CFI(M, nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
    # calculate the average vector
    vec_ave = zeros(ComplexF64, dim)
    for ni in 1:dim
        vec_ave[ni] = [nelder_mead[pk].psi[ni] for pk in 1:state_num] |>sum
        vec_ave[ni] = vec_ave[ni]/state_num
    end
    vec_ave = vec_ave/norm(vec_ave)

    # reflection
    vec_ref = zeros(ComplexF64, dim)
    for nj in 1:dim
        vec_ref[nj] = vec_ave[nj] + coeff_r*(vec_ave[nj]-nelder_mead[sort_ind[end]].psi[nj])
    end
    vec_ref = vec_ref/norm(vec_ref)

    fr = -CFI(M, NM.freeHamiltonian, NM.Hamiltonian_derivative[1], vec_ref*vec_ref', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)

    if fr < p_fit[sort_ind[1]]
        # expansion
        vec_exp = zeros(ComplexF64, dim)
        for nk in 1:dim
            vec_exp[nk] = vec_ave[nk] + coeff_e*(vec_ref[nk]-vec_ave[nk])
        end
        vec_exp = vec_exp/norm(vec_exp)

        fe = -CFI(M, NM.freeHamiltonian, NM.Hamiltonian_derivative[1], vec_exp*vec_exp', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)

        if fe >= fr
            for np in 1:dim
                nelder_mead[sort_ind[end]].psi[np] = vec_ref[np]
            end
            p_fit[sort_ind[end]] = fr
            sort_ind = sortperm(p_fit)
        else
            for np in 1:dim
                nelder_mead[sort_ind[end]].psi[np] = vec_exp[np]
            end
            p_fit[sort_ind[end]] = fe
            sort_ind = sortperm(p_fit)
        end
    elseif fr > p_fit[sort_ind[end-1]]
        # constraction
        if fr >= p_fit[sort_ind[end]]
            # inside constraction
            vec_ic = zeros(ComplexF64, dim)
            for nl in 1:dim
                vec_ic[nl] = vec_ave[nl] - coeff_c*(vec_ave[nl]-nelder_mead[sort_ind[end]].psi[nl])
            end
            vec_ic = vec_ic/norm(vec_ic)

            fic = -CFI(M, NM.freeHamiltonian, NM.Hamiltonian_derivative[1], vec_ic*vec_ic', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
        
            if fic < p_fit[sort_ind[end]]
                for np in 1:dim
                    nelder_mead[sort_ind[end]].psi[np] = vec_ic[np]
                end
                p_fit[sort_ind[end]] = fic
                sort_ind = sortperm(p_fit)
            else
                # shrink
                vec_first = [nelder_mead[sort_ind[1]].psi[i] for i in 1:dim]
                for pk in 1:state_num
                    for nq in 1:dim
                        nelder_mead[pk].psi[nq] = vec_first[nq] + coeff_s*(nelder_mead[pk].psi[nq]-vec_first[nq])
                    end
                    nelder_mead[pk].psi = nelder_mead[pk].psi/norm(nelder_mead[pk].psi)

                    rho = nelder_mead[pk].psi*(nelder_mead[pk].psi)'
                    p_fit[pk] = -CFI(M, NM.freeHamiltonian, NM.Hamiltonian_derivative[1], rho, NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
                end
                sort_ind = sortperm(p_fit)
            end
        else
            # outside constraction
            vec_oc = zeros(ComplexF64, dim)
            for nn in 1:dim
                vec_oc[nn] = vec_ave[nn] + coeff_c*(vec_ref[nn]-vec_ave[nn])
            end
            vec_oc = vec_oc/norm(vec_oc)

            foc = -CFI(M, NM.freeHamiltonian, NM.Hamiltonian_derivative[1], vec_oc*vec_oc', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)

            if foc <= fr
                for np in 1:dim
                    nelder_mead[sort_ind[end]].psi[np] = vec_oc[np]
                end
                p_fit[sort_ind[end]] = foc
                sort_ind = sortperm(p_fit)
            else
                # shrink
                vec_first = [nelder_mead[sort_ind[1]].psi[i] for i in 1:dim]
                for pk in 1:state_num
                    for nq in 1:dim
                        nelder_mead[pk].psi[nq] = vec_first[nq] + coeff_s*(nelder_mead[pk].psi[nq]-vec_first[nq])
                    end
                    nelder_mead[pk].psi = nelder_mead[pk].psi/norm(nelder_mead[pk].psi)

                    rho = nelder_mead[pk].psi*(nelder_mead[pk].psi)'
                    p_fit[pk] = -CFI(M, NM.freeHamiltonian, NM.Hamiltonian_derivative[1], rho, NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
                end
                sort_ind = sortperm(p_fit)
            end
        end
    else
        for np in 1:dim
            nelder_mead[sort_ind[end]].psi[np] = vec_ref[np]
        end
        p_fit[sort_ind[end]] = fr
        sort_ind = sortperm(p_fit)
    end
    return p_fit, sort_ind
end

function neldermead_train_CFIM(M, nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
    # calculate the average vector
    vec_ave = zeros(ComplexF64, dim)
    for ni in 1:dim
        vec_ave[ni] = [nelder_mead[pk].psi[ni] for pk in 1:state_num] |>sum
        vec_ave[ni] = vec_ave[ni]/state_num
    end
    vec_ave = vec_ave/norm(vec_ave)

    # reflection
    vec_ref = zeros(ComplexF64, dim)
    for nj in 1:dim
        vec_ref[nj] = vec_ave[nj] + coeff_r*(vec_ave[nj]-nelder_mead[sort_ind[end]].psi[nj])
    end
    vec_ref = vec_ref/norm(vec_ref)

    F_r = CFIM(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_ref*vec_ref', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
    fr = -1.0/real(tr(NM.W*pinv(F_r)))

    if fr < p_fit[sort_ind[1]]
        # expansion
        vec_exp = zeros(ComplexF64, dim)
        for nk in 1:dim
            vec_exp[nk] = vec_ave[nk] + coeff_e*(vec_ref[nk]-vec_ave[nk])
        end
        vec_exp = vec_exp/norm(vec_exp)

        F_e = CFIM(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_exp*vec_exp', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
        fe = -1.0/real(tr(NM.W*pinv(F_e)))

        if fe >= fr
            for np in 1:dim
                nelder_mead[sort_ind[end]].psi[np] = vec_ref[np]
            end
            p_fit[sort_ind[end]] = fr
            sort_ind = sortperm(p_fit)
        else
            for np in 1:dim
                nelder_mead[sort_ind[end]].psi[np] = vec_exp[np]
            end
            p_fit[sort_ind[end]] = fe
            sort_ind = sortperm(p_fit)
        end
    elseif fr > p_fit[sort_ind[end-1]]
        # constraction
        if fr >= p_fit[sort_ind[end]]
            # inside constraction
            vec_ic = zeros(ComplexF64, dim)
            for nl in 1:dim
                vec_ic[nl] = vec_ave[nl] - coeff_c*(vec_ave[nl]-nelder_mead[sort_ind[end]].psi[nl])
            end
            vec_ic = vec_ic/norm(vec_ic)

            F_ic = CFIM(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_ic*vec_ic', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
            fic = -1.0/real(tr(NM.W*pinv(F_ic)))

            if fic < p_fit[sort_ind[end]]
                for np in 1:dim
                    nelder_mead[sort_ind[end]].psi[np] = vec_ic[np]
                end
                p_fit[sort_ind[end]] = fic
                sort_ind = sortperm(p_fit)
            else
                # shrink
                vec_first = [nelder_mead[sort_ind[1]].psi[i] for i in 1:dim]
                for pk in 1:state_num
                    for nq in 1:dim
                        nelder_mead[pk].psi[nq] = vec_first[nq] + coeff_s*(nelder_mead[pk].psi[nq]-vec_first[nq])
                    end
                    nelder_mead[pk].psi = nelder_mead[pk].psi/norm(nelder_mead[pk].psi)

                    rho = nelder_mead[pk].psi*(nelder_mead[pk].psi)'
                    F_tp = CFIM(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, rho, NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
                    p_fit[pk] = -1.0/real(tr(NM.W*pinv(F_tp)))
                end
                sort_ind = sortperm(p_fit)
            end
        else
            # outside constraction
            vec_oc = zeros(ComplexF64, dim)
            for nn in 1:dim
                vec_oc[nn] = vec_ave[nn] + coeff_c*(vec_ref[nn]-vec_ave[nn])
            end
            vec_oc = vec_oc/norm(vec_oc)

            F_oc = CFIM(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_oc*vec_oc', NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
            foc = -1.0/real(tr(NM.W*pinv(F_oc)))

            if foc <= fr
                for np in 1:dim
                    nelder_mead[sort_ind[end]].psi[np] = vec_oc[np]
                end
                p_fit[sort_ind[end]] = foc
                sort_ind = sortperm(p_fit)
            else
                # shrink
                vec_first = [nelder_mead[sort_ind[1]].psi[i] for i in 1:dim]
                for pk in 1:state_num
                    for nq in 1:dim
                        nelder_mead[pk].psi[nq] = vec_first[nq] + coeff_s*(nelder_mead[pk].psi[nq]-vec_first[nq])
                    end
                    nelder_mead[pk].psi = nelder_mead[pk].psi/norm(nelder_mead[pk].psi)

                    rho = nelder_mead[pk].psi*(nelder_mead[pk].psi)'
                    F_tp = CFIM(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, rho, NM.Liouville_operator, NM.γ, NM.control_Hamiltonian, NM.control_coefficients, NM.times)
                    p_fit[pk] = -1.0/real(tr(NM.W*pinv(F_tp)))
                end
                sort_ind = sortperm(p_fit)
            end
        end
    else
        for np in 1:dim
            nelder_mead[sort_ind[end]].psi[np] = vec_ref[np]
        end
        p_fit[sort_ind[end]] = fr
        sort_ind = sortperm(p_fit)
    end
    return p_fit, sort_ind
end