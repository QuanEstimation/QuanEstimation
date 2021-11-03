############# time-independent Hamiltonian (noiseless) ################
mutable struct TimeIndepend_noiseless{T <: Complex,M <: Real}
    freeHamiltonian::Matrix{T}
    Hamiltonian_derivative::Vector{Matrix{T}}
    psi::Vector{T}
    times::Vector{M}
    W::Matrix{M}
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    TimeIndepend_noiseless(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, psi::Vector{T},
             times::Vector{M}, W::Matrix{M}, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = new{T,M}(freeHamiltonian, 
                Hamiltonian_derivative, psi, times, W, ρ, ∂ρ_∂x) 
end

function NelderMead_QFI(NM::TimeIndepend_noiseless{T}, state_num, ini_state, coeff_r, coeff_e, coeff_c, coeff_s, epsilon, max_episodes, seed, save_file) where {T<: Complex}
    println("state optimization")
    println("single parameter scenario")
    println("search algorithm: Nelder-Mead method (NM)")

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
        p_fit[pj] = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative[1], nelder_mead[pj].psi, NM.times)
    end
    sort_ind = sortperm(p_fit, rev=true)

    f_ini = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative[1], NM.psi, NM.times)
    f_list = [f_ini]
    println("initial QFI is $(f_ini)")

    episodes = 1
    if save_file == true
        SaveFile_nm(dim, f_list, NM.psi)
        while true
            p_fit, sort_ind = train_QFIM_noiseless(nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[1]]-p_fit[sort_ind[end]] < epsilon || episodes >= max_episodes
                append!(f_list, maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final QFI is ", maximum(p_fit))
                break
            else
                episodes += 1
                append!(f_list, maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("current QFI is ", maximum(p_fit), " ($(episodes-1) episodes)    \r")
            end
        end
    else
        while true
            p_fit, sort_ind = train_QFIM_noiseless(nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[1]]-p_fit[sort_ind[end]] < epsilon || episodes >= max_episodes
                append!(f_list, maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final QFI is ", maximum(p_fit))
                break
            else
                episodes += 1
                append!(f_list, maximum(p_fit))
                print("current QFI is ", maximum(p_fit), " ($(episodes-1) episodes)    \r")
            end
        end
    end
end

function NelderMead_QFIM(NM::TimeIndepend_noiseless{T}, state_num, ini_state, coeff_r, coeff_e, coeff_c, coeff_s, epsilon, max_episodes, seed, save_file) where {T<: Complex}
    println("state optimization")
    println("multiparameter scenario")
    println("search algorithm: Nelder-Mead method (NM)")

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
        F_tp = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative, nelder_mead[pj].psi, NM.times)
        p_fit[pj] = 1.0/real(tr(NM.W*pinv(F_tp)))
    end
    sort_ind = sortperm(p_fit, rev=true)

    F = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative, NM.psi, NM.times)
    f_ini = real(tr(NM.W*pinv(F)))
    f_list = [f_ini]
    println("initial value of Tr(WF^{-1}) is $(f_ini)")

    episodes = 1
    if save_file == true
        SaveFile_nm(dim, f_list, NM.psi)
        while true
            p_fit, sort_ind = train_QFIM_noiseless(nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[1]]-p_fit[sort_ind[end]] < epsilon || episodes >= max_episodes
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
                break
            else
                episodes += 1
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($(episodes-1) episodes)    \r")
            end
        end
    else
        while true
            p_fit, sort_ind = train_QFIM_noiseless(nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[1]]-p_fit[sort_ind[end]] < epsilon || episodes >= max_episodes
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
                break
            else
                episodes += 1
                append!(f_list, 1.0/maximum(p_fit))
                print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($(episodes-1) episodes)    \r")
            end
        end
    end
end

function NelderMead_CFI(M, NM::TimeIndepend_noiseless{T}, state_num, ini_state, coeff_r, coeff_e, coeff_c, coeff_s, epsilon, max_episodes, seed, save_file) where {T<: Complex}
    println("state optimization")
    println("single parameter scenario")
    println("search algorithm: Nelder-Mead method (NM)")

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
        p_fit[pj] = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative[1], nelder_mead[pj].psi, NM.times)
    end
    sort_ind = sortperm(p_fit, rev=true)

    f_ini = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative[1], NM.psi, NM.times)
    f_list = [f_ini]
    println("initial CFI is $(f_ini)")

    episodes = 1
    if save_file == true
        SaveFile_nm(dim, f_list, NM.psi)
        while true
            p_fit, sort_ind = train_CFIM_noiseless(M, nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[1]]-p_fit[sort_ind[end]] < epsilon || episodes >= max_episodes
                append!(f_list, maximum(p_fit))    
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final CFI is ", maximum(p_fit))
                break
            else
                episodes += 1
                append!(f_list, maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("current CFI is ", maximum(p_fit), " ($(episodes-1) episodes)    \r")
            end
        end
    else
        while true
            p_fit, sort_ind = train_CFIM_noiseless(M, nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[1]]-p_fit[sort_ind[end]] < epsilon || episodes >= max_episodes
                append!(f_list, maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final CFI is ", maximum(p_fit))
                break
            else
                episodes += 1
                append!(f_list, maximum(p_fit))
                print("current CFI is ", maximum(p_fit), " ($(episodes-1) episodes)    \r")
            end
        end
    end
end

function NelderMead_CFIM(M, NM::TimeIndepend_noiseless{T}, state_num, ini_state, coeff_r, coeff_e, coeff_c, coeff_s, epsilon, max_episodes, seed, save_file) where {T<: Complex}
    println("state optimization")
    println("multiparameter scenario")
    println("search algorithm: Nelder-Mead method (NM)")

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
        F_tp = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, nelder_mead[pj].psi, NM.times)
        p_fit[pj] = 1.0/real(tr(NM.W*pinv(F_tp)))
    end
    sort_ind = sortperm(p_fit, rev=true)

    F = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, NM.psi, NM.times)
    f_ini = real(tr(NM.W*pinv(F)))
    f_list = [f_ini]
    println("initial value of Tr(WF^{-1}) is $(f_ini)")

    episodes = 1
    if save_file == true
        SaveFile_nm(dim, f_list, NM.psi)
        while true
            p_fit, sort_ind = train_CFIM_noiseless(M, nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[1]]-p_fit[sort_ind[end]] < epsilon || episodes >= max_episodes
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
                break
            else
                episodes += 1
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($(episodes-1) episodes)    \r")
            end
        end
    else
        while true
            p_fit, sort_ind = train_CFIM_noiseless(M, nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[1]]-p_fit[sort_ind[end]] < epsilon || episodes >= max_episodes
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
                break
            else
                episodes += 1
                append!(f_list, 1.0/maximum(p_fit))
                print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($(episodes-1) episodes)    \r")
            end
        end
    end
end

function train_QFIM_noiseless(nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
    # calculate the average vector
    vec_ave = zeros(ComplexF64, dim)
    for ni in 1:dim
        vec_ave[ni] = [nelder_mead[pk].psi[ni] for pk in 1:(state_num-1)] |>sum
        vec_ave[ni] = vec_ave[ni]/(state_num-1)
    end
    vec_ave = vec_ave/norm(vec_ave)

    # reflection
    vec_ref = zeros(ComplexF64, dim)
    for nj in 1:dim
        vec_ref[nj] = vec_ave[nj] + coeff_r*(vec_ave[nj]-nelder_mead[sort_ind[end]].psi[nj])
    end
    vec_ref = vec_ref/norm(vec_ref)

    F_r = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_ref, NM.times)
    fr = 1.0/real(tr(NM.W*pinv(F_r)))

    if fr > p_fit[sort_ind[1]]
        # expansion
        vec_exp = zeros(ComplexF64, dim)
        for nk in 1:dim
            vec_exp[nk] = vec_ave[nk] + coeff_e*(vec_ref[nk]-vec_ave[nk])
        end
        vec_exp = vec_exp/norm(vec_exp)

        F_e = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_exp, NM.times)
        fe = 1.0/real(tr(NM.W*pinv(F_e)))

        if fe <= fr
            for np in 1:dim
                nelder_mead[sort_ind[end]].psi[np] = vec_ref[np]
            end
            p_fit[sort_ind[end]] = fr
            sort_ind = sortperm(p_fit, rev=true)
        else
            for np in 1:dim
                nelder_mead[sort_ind[end]].psi[np] = vec_exp[np]
            end
            p_fit[sort_ind[end]] = fe
            sort_ind = sortperm(p_fit, rev=true)
        end
    elseif fr < p_fit[sort_ind[end-1]]
        # constraction
        if fr <= p_fit[sort_ind[end]]
            # inside constraction
            vec_ic = zeros(ComplexF64, dim)
            for nl in 1:dim
                vec_ic[nl] = vec_ave[nl] - coeff_c*(vec_ave[nl]-nelder_mead[sort_ind[end]].psi[nl])
            end
            vec_ic = vec_ic/norm(vec_ic)

            F_ic = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_ic, NM.times)
            fic = 1.0/real(tr(NM.W*pinv(F_ic)))

            if fic > p_fit[sort_ind[end]]
                for np in 1:dim
                    nelder_mead[sort_ind[end]].psi[np] = vec_ic[np]
                end
                p_fit[sort_ind[end]] = fic
                sort_ind = sortperm(p_fit, rev=true)
            else
                # shrink
                vec_first = [nelder_mead[sort_ind[1]].psi[i] for i in 1:dim]
                for pk in 1:state_num
                    for nq in 1:dim
                        nelder_mead[pk].psi[nq] = vec_first[nq] + coeff_s*(nelder_mead[pk].psi[nq]-vec_first[nq])
                    end
                    nelder_mead[pk].psi = nelder_mead[pk].psi/norm(nelder_mead[pk].psi)

                    F_tp = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative, nelder_mead[pk].psi, NM.times)
                    p_fit[pk] = 1.0/real(tr(NM.W*pinv(F_tp)))
                end
                sort_ind = sortperm(p_fit, rev=true)
            end
        else
            # outside constraction
            vec_oc = zeros(ComplexF64, dim)
            for nn in 1:dim
                vec_oc[nn] = vec_ave[nn] + coeff_c*(vec_ref[nn]-vec_ave[nn])
            end
            vec_oc = vec_oc/norm(vec_oc)

            F_oc = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_oc, NM.times)
            foc = 1.0/real(tr(NM.W*pinv(F_oc)))

            if foc >= fr
                for np in 1:dim
                    nelder_mead[sort_ind[end]].psi[np] = vec_oc[np]
                end
                p_fit[sort_ind[end]] = foc
                sort_ind = sortperm(p_fit, rev=true)
            else
                # shrink
                vec_first = [nelder_mead[sort_ind[1]].psi[i] for i in 1:dim]
                for pk in 1:state_num
                    for nq in 1:dim
                        nelder_mead[pk].psi[nq] = vec_first[nq] + coeff_s*(nelder_mead[pk].psi[nq]-vec_first[nq])
                    end
                    nelder_mead[pk].psi = nelder_mead[pk].psi/norm(nelder_mead[pk].psi)

                    F_tp = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative, nelder_mead[pk].psi, NM.times)
                    p_fit[pk] = 1.0/real(tr(NM.W*pinv(F_tp)))
                end
                sort_ind = sortperm(p_fit, rev=true)
            end
        end
    else
        for np in 1:dim
            nelder_mead[sort_ind[end]].psi[np] = vec_ref[np]
        end
        p_fit[sort_ind[end]] = fr
        sort_ind = sortperm(p_fit, rev=true)
    end
    return p_fit, sort_ind
end

function train_CFIM_noiseless(M, nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
    # calculate the average vector
    vec_ave = zeros(ComplexF64, dim)
    for ni in 1:dim
        vec_ave[ni] = [nelder_mead[pk].psi[ni] for pk in 1:(state_num-1)] |>sum
        vec_ave[ni] = vec_ave[ni]/(state_num-1)
    end
    vec_ave = vec_ave/norm(vec_ave)

    # reflection
    vec_ref = zeros(ComplexF64, dim)
    for nj in 1:dim
        vec_ref[nj] = vec_ave[nj] + coeff_r*(vec_ave[nj]-nelder_mead[sort_ind[end]].psi[nj])
    end
    vec_ref = vec_ref/norm(vec_ref)

    F_r = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_ref, NM.times)
    fr = 1.0/real(tr(NM.W*pinv(F_r)))

    if fr > p_fit[sort_ind[1]]
        # expansion
        vec_exp = zeros(ComplexF64, dim)
        for nk in 1:dim
            vec_exp[nk] = vec_ave[nk] + coeff_e*(vec_ref[nk]-vec_ave[nk])
        end
        vec_exp = vec_exp/norm(vec_exp)

        F_e = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_exp, NM.times)
        fe = 1.0/real(tr(NM.W*pinv(F_e)))

        if fe <= fr
            for np in 1:dim
                nelder_mead[sort_ind[end]].psi[np] = vec_ref[np]
            end
            p_fit[sort_ind[end]] = fr
            sort_ind = sortperm(p_fit, rev=true)
        else
            for np in 1:dim
                nelder_mead[sort_ind[end]].psi[np] = vec_exp[np]
            end
            p_fit[sort_ind[end]] = fe
            sort_ind = sortperm(p_fit, rev=true)
        end
    elseif fr < p_fit[sort_ind[end-1]]
        # constraction
        if fr <= p_fit[sort_ind[end]]
            # inside constraction
            vec_ic = zeros(ComplexF64, dim)
            for nl in 1:dim
                vec_ic[nl] = vec_ave[nl] - coeff_c*(vec_ave[nl]-nelder_mead[sort_ind[end]].psi[nl])
            end
            vec_ic = vec_ic/norm(vec_ic)

            F_ic = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_ic, NM.times)
            fic = 1.0/real(tr(NM.W*pinv(F_ic)))

            if fic > p_fit[sort_ind[end]]
                for np in 1:dim
                    nelder_mead[sort_ind[end]].psi[np] = vec_ic[np]
                end
                p_fit[sort_ind[end]] = fic
                sort_ind = sortperm(p_fit, rev=true)
            else
                # shrink
                vec_first = [nelder_mead[sort_ind[1]].psi[i] for i in 1:dim]
                for pk in 1:state_num
                    for nq in 1:dim
                        nelder_mead[pk].psi[nq] = vec_first[nq] + coeff_s*(nelder_mead[pk].psi[nq]-vec_first[nq])
                    end
                    nelder_mead[pk].psi = nelder_mead[pk].psi/norm(nelder_mead[pk].psi)

                    F_tp = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, nelder_mead[pk].psi, NM.times)
                    p_fit[pk] = 1.0/real(tr(NM.W*pinv(F_tp)))
                end
                sort_ind = sortperm(p_fit, rev=true)
            end
        else
            # outside constraction
            vec_oc = zeros(ComplexF64, dim)
            for nn in 1:dim
                vec_oc[nn] = vec_ave[nn] + coeff_c*(vec_ref[nn]-vec_ave[nn])
            end
            vec_oc = vec_oc/norm(vec_oc)

            F_oc = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_oc, NM.times)
            foc = 1.0/real(tr(NM.W*pinv(F_oc)))

            if foc >= fr
                for np in 1:dim
                    nelder_mead[sort_ind[end]].psi[np] = vec_oc[np]
                end
                p_fit[sort_ind[end]] = foc
                sort_ind = sortperm(p_fit, rev=true)
            else
                # shrink
                vec_first = [nelder_mead[sort_ind[1]].psi[i] for i in 1:dim]
                for pk in 1:state_num
                    for nq in 1:dim
                        nelder_mead[pk].psi[nq] = vec_first[nq] + coeff_s*(nelder_mead[pk].psi[nq]-vec_first[nq])
                    end
                    nelder_mead[pk].psi = nelder_mead[pk].psi/norm(nelder_mead[pk].psi)

                    F_tp = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, nelder_mead[pk].psi, NM.times)
                    p_fit[pk] = 1.0/real(tr(NM.W*pinv(F_tp)))
                end
                sort_ind = sortperm(p_fit, rev=true)
            end
        end
    else
        for np in 1:dim
            nelder_mead[sort_ind[end]].psi[np] = vec_ref[np]
        end
        p_fit[sort_ind[end]] = fr
        sort_ind = sortperm(p_fit, rev=true)
    end
    return p_fit, sort_ind
end

############# time-independent Hamiltonian (noise) ################
mutable struct TimeIndepend_noise{T <: Complex,M <: Real}
    freeHamiltonian::Matrix{T}
    Hamiltonian_derivative::Vector{Matrix{T}}
    psi::Vector{T}
    times::Vector{M}
    Liouville_operator::Vector{Matrix{T}}
    γ::Vector{M}
    W::Matrix{M}
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    TimeIndepend_noise(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, psi::Vector{T},
             times::Vector{M}, Liouville_operator::Vector{Matrix{T}}, γ::Vector{M}, W::Matrix{M}, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = new{T,M}(freeHamiltonian, 
                Hamiltonian_derivative, psi, times, Liouville_operator, γ, W, ρ, ∂ρ_∂x) 
end

function NelderMead_QFI(NM::TimeIndepend_noise{T}, state_num, ini_state, coeff_r, coeff_e, coeff_c, coeff_s, epsilon, max_episodes, seed, save_file) where {T<: Complex}
    println("state optimization")
    println("single parameter scenario")
    println("search algorithm: Nelder-Mead method (NM)")

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
        rho = nelder_mead[pj].psi*(nelder_mead[pj].psi)'
        p_fit[pj] = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative[1], rho, NM.Liouville_operator, NM.γ, NM.times)
    end
    sort_ind = sortperm(p_fit, rev=true)

    f_ini = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative[1], NM.psi*(NM.psi)', NM.Liouville_operator, NM.γ, NM.times)
    f_list = [f_ini]
    println("initial QFI is $(f_ini)")

    episodes = 1
    if save_file == true
        SaveFile_nm(dim, f_list, NM.psi)
        while true
            p_fit, sort_ind = train_QFIM_noise(nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[1]]-p_fit[sort_ind[end]] < epsilon || episodes >= max_episodes
                append!(f_list, maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final QFI is ", maximum(p_fit))
                break
            else
                episodes += 1
                append!(f_list, maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("current QFI is ", maximum(p_fit), " ($(episodes-1) episodes)    \r")
            end
        end
    else
        while true
            p_fit, sort_ind = train_QFIM_noise(nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[1]]-p_fit[sort_ind[end]] < epsilon || episodes >= max_episodes
                append!(f_list, maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)    
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final QFI is ", maximum(p_fit))
                break
            else
                episodes += 1
                append!(f_list, maximum(p_fit))
                print("current QFI is ", maximum(p_fit), " ($(episodes-1) episodes)    \r")
            end
        end
    end
end

function NelderMead_QFIM(NM::TimeIndepend_noise{T}, state_num, ini_state, coeff_r, coeff_e, coeff_c, coeff_s, epsilon, max_episodes, seed, save_file) where {T<: Complex}
    println("state optimization")
    println("multiparameter scenario")
    println("search algorithm: Nelder-Mead method (NM)")

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
        rho = nelder_mead[pj].psi*(nelder_mead[pj].psi)'
        F_tp = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative, rho, NM.Liouville_operator, NM.γ, NM.times)
        p_fit[pj] = 1.0/real(tr(NM.W*pinv(F_tp)))
    end
    sort_ind = sortperm(p_fit, rev=true)
    F = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative, NM.psi*(NM.psi)', NM.Liouville_operator, NM.γ, NM.times)
    f_ini = real(tr(NM.W*pinv(F)))
    f_list = [f_ini]
    println("initial value of Tr(WF^{-1}) is $(f_ini)")

    episodes = 1
    if save_file == true
        SaveFile_nm(dim, f_list, NM.psi)
        while true
            p_fit, sort_ind = train_QFIM_noise(nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[1]]-p_fit[sort_ind[end]] < epsilon || episodes >= max_episodes
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
                break
            else
                episodes += 1
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($(episodes-1) episodes)    \r")
            end
        end
    else
        while true
            p_fit, sort_ind = train_QFIM_noise(nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[1]]-p_fit[sort_ind[end]] < epsilon || episodes >= max_episodes
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
                break
            else
                episodes += 1
                append!(f_list, 1.0/maximum(p_fit))
                print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($(episodes-1) episodes)    \r")
            end
        end
    end
end

function NelderMead_CFI(M, NM::TimeIndepend_noise{T}, state_num, ini_state, coeff_r, coeff_e, coeff_c, coeff_s, epsilon, max_episodes, seed, save_file) where {T<: Complex}
    println("state optimization")
    println("single parameter scenario")
    println("search algorithm: Nelder-Mead method (NM)")

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
        rho = nelder_mead[pj].psi*(nelder_mead[pj].psi)'
        p_fit[pj] = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative[1], rho, NM.Liouville_operator, NM.γ, NM.times)
    end
    sort_ind = sortperm(p_fit, rev=true)

    f_ini = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative[1], NM.psi*(NM.psi)', NM.Liouville_operator, NM.γ, NM.times)
    f_list = [f_ini]
    println("initial CFI is $(f_ini)")

    episodes = 1
    if save_file == true
        SaveFile_nm(dim, f_list, NM.psi)
        while true
            p_fit, sort_ind = train_CFIM_noise(M, nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[1]]-p_fit[sort_ind[end]] < epsilon || episodes >= max_episodes
                append!(f_list, maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final CFI is ", maximum(p_fit))
                break
            else
                episodes += 1
                append!(f_list, maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("current CFI is ", maximum(p_fit), " ($(episodes-1) episodes)    \r")
            end
        end
    else
        while true
            p_fit, sort_ind = train_CFIM_noise(M, nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[1]]-p_fit[sort_ind[end]] < epsilon || episodes >= max_episodes
                append!(f_list, maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final CFI is ", maximum(p_fit))
                break
            else
                episodes += 1
                append!(f_list, maximum(p_fit))
                print("current CFI is ", maximum(p_fit), " ($(episodes-1) episodes)    \r")
            end
        end
    end
end


function NelderMead_CFIM(M, NM::TimeIndepend_noise{T}, state_num, ini_state, coeff_r, coeff_e, coeff_c, coeff_s, epsilon, max_episodes, seed, save_file) where {T<: Complex}
    println("state optimization")
    println("multiparameter scenario")
    println("search algorithm: Nelder-Mead method (NM)")

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
        rho = nelder_mead[pj].psi*(nelder_mead[pj].psi)'
        F_tp = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, rho, NM.Liouville_operator, NM.γ, NM.times)
        p_fit[pj] = 1.0/real(tr(NM.W*pinv(F_tp)))
    end
    sort_ind = sortperm(p_fit, rev=true)

    F = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, NM.psi*(NM.psi)', NM.Liouville_operator, NM.γ, NM.times)
    f_ini = real(tr(NM.W*pinv(F)))
    f_list = [f_ini]
    println("initial value of Tr(WF^{-1}) is $(f_ini)")

    episodes = 1
    if save_file == true
        SaveFile_nm(dim, f_list, NM.psi)
        while true
            p_fit, sort_ind = train_CFIM_noise(M, nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[1]]-p_fit[sort_ind[end]] < epsilon || episodes >= max_episodes
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
                break
            else
                episodes += 1
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($(episodes-1) episodes)    \r")
            end
        end
    else
        while true
            p_fit, sort_ind = train_CFIM_noise(M, nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
            if p_fit[sort_ind[1]]-p_fit[sort_ind[end]] < epsilon || episodes >= max_episodes
                append!(f_list, 1.0/maximum(p_fit))
                SaveFile_nm(dim, f_list, nelder_mead[sort_ind[1]].psi)
                print("\e[2K")
                println("Iteration over, data saved.")
                println("Final value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit))
                break
            else
                episodes += 1
                append!(f_list, 1.0/maximum(p_fit))
                print("current value of Tr(WF^{-1}) is ", 1.0/maximum(p_fit), " ($(episodes-1) episodes)    \r")
            end
        end
    end
end

function train_QFIM_noise(nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
    # calculate the average vector
    ####test
    vec_ave = zeros(ComplexF64, dim)
    for ni in 1:dim
        vec_ave[ni] = [nelder_mead[pk].psi[ni] for pk in 1:(state_num-1)] |>sum
        vec_ave[ni] = vec_ave[ni]/(state_num-1)
    end
    vec_ave = vec_ave/norm(vec_ave)

    # reflection
    vec_ref = zeros(ComplexF64, dim)
    for nj in 1:dim
        vec_ref[nj] = vec_ave[nj] + coeff_r*(vec_ave[nj]-nelder_mead[sort_ind[end]].psi[nj])
    end
    vec_ref = vec_ref/norm(vec_ref)

    F_r = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_ref*vec_ref', NM.Liouville_operator, NM.γ, NM.times)
    fr = 1.0/real(tr(NM.W*pinv(F_r)))

    if fr > p_fit[sort_ind[1]]
        # expansion
        vec_exp = zeros(ComplexF64, dim)
        for nk in 1:dim
            vec_exp[nk] = vec_ave[nk] + coeff_e*(vec_ref[nk]-vec_ave[nk])
        end
        vec_exp = vec_exp/norm(vec_exp)

        F_e = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_exp*vec_exp', NM.Liouville_operator, NM.γ, NM.times)
        fe = 1.0/real(tr(NM.W*pinv(F_e)))

        if fe <= fr
            for np in 1:dim
                nelder_mead[sort_ind[end]].psi[np] = vec_ref[np]
            end
            p_fit[sort_ind[end]] = fr
            sort_ind = sortperm(p_fit, rev=true)
        else
            for np in 1:dim
                nelder_mead[sort_ind[end]].psi[np] = vec_exp[np]
            end
            p_fit[sort_ind[end]] = fe
            sort_ind = sortperm(p_fit, rev=true)
        end
    elseif fr < p_fit[sort_ind[end-1]]
        # constraction
        if fr <= p_fit[sort_ind[end]]
            # inside constraction
            vec_ic = zeros(ComplexF64, dim)
            for nl in 1:dim
                vec_ic[nl] = vec_ave[nl] - coeff_c*(vec_ave[nl]-nelder_mead[sort_ind[end]].psi[nl])
            end
            vec_ic = vec_ic/norm(vec_ic)

            F_ic = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_ic*vec_ic', NM.Liouville_operator, NM.γ, NM.times)
            fic = 1.0/real(tr(NM.W*pinv(F_ic)))

            if fic > p_fit[sort_ind[end]]
                for np in 1:dim
                    nelder_mead[sort_ind[end]].psi[np] = vec_ic[np]
                end
                p_fit[sort_ind[end]] = fic
                sort_ind = sortperm(p_fit, rev=true)
            else
                # shrink
                vec_first = [nelder_mead[sort_ind[1]].psi[i] for i in 1:dim]
                for pk in 1:state_num
                    for nq in 1:dim
                        nelder_mead[pk].psi[nq] = vec_first[nq] + coeff_s*(nelder_mead[pk].psi[nq]-vec_first[nq])
                    end
                    nelder_mead[pk].psi = nelder_mead[pk].psi/norm(nelder_mead[pk].psi)

                    rho = nelder_mead[pk].psi*(nelder_mead[pk].psi)'
                    F_tp = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative, rho, NM.Liouville_operator, NM.γ, NM.times)
                    p_fit[pk] = 1.0/real(tr(NM.W*pinv(F_tp)))
                end
                sort_ind = sortperm(p_fit, rev=true)
            end
        else
            # outside constraction
            vec_oc = zeros(ComplexF64, dim)
            for nn in 1:dim
                vec_oc[nn] = vec_ave[nn] + coeff_c*(vec_ref[nn]-vec_ave[nn])
            end
            vec_oc = vec_oc/norm(vec_oc)

            F_oc = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_oc*vec_oc', NM.Liouville_operator, NM.γ, NM.times)
            foc = 1.0/real(tr(NM.W*pinv(F_oc)))

            if foc >= fr
                for np in 1:dim
                    nelder_mead[sort_ind[end]].psi[np] = vec_oc[np]
                end
                p_fit[sort_ind[end]] = foc
                sort_ind = sortperm(p_fit, rev=true)
            else
                # shrink
                vec_first = [nelder_mead[sort_ind[1]].psi[i] for i in 1:dim]
                for pk in 1:state_num
                    for nq in 1:dim
                        nelder_mead[pk].psi[nq] = vec_first[nq] + coeff_s*(nelder_mead[pk].psi[nq]-vec_first[nq])
                    end
                    nelder_mead[pk].psi = nelder_mead[pk].psi/norm(nelder_mead[pk].psi)

                    rho = nelder_mead[pk].psi*(nelder_mead[pk].psi)'
                    F_tp = QFIM_TimeIndepend(NM.freeHamiltonian, NM.Hamiltonian_derivative, rho, NM.Liouville_operator, NM.γ, NM.times)
                    p_fit[pk] = 1.0/real(tr(NM.W*pinv(F_tp)))
                end
                sort_ind = sortperm(p_fit, rev=true)
            end
        end
    else
        for np in 1:dim
            nelder_mead[sort_ind[end]].psi[np] = vec_ref[np]
        end
        p_fit[sort_ind[end]] = fr
        sort_ind = sortperm(p_fit, rev=true)
    end
    return p_fit, sort_ind
end

function train_CFIM_noise(M, nelder_mead, NM, p_fit, sort_ind, dim, state_num, coeff_r, coeff_e, coeff_c, coeff_s)
    # calculate the average vector
    vec_ave = zeros(ComplexF64, dim)
    for ni in 1:dim
        vec_ave[ni] = [nelder_mead[pk].psi[ni] for pk in 1:(state_num-1)] |>sum
        vec_ave[ni] = vec_ave[ni]/(state_num-1)
    end
    vec_ave = vec_ave/norm(vec_ave)

    # reflection
    vec_ref = zeros(ComplexF64, dim)
    for nj in 1:dim
        vec_ref[nj] = vec_ave[nj] + coeff_r*(vec_ave[nj]-nelder_mead[sort_ind[end]].psi[nj])
    end
    vec_ref = vec_ref/norm(vec_ref)

    F_r = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_ref*vec_ref', NM.Liouville_operator, NM.γ, NM.times)
    fr = 1.0/real(tr(NM.W*pinv(F_r)))

    if fr > p_fit[sort_ind[1]]
        # expansion
        vec_exp = zeros(ComplexF64, dim)
        for nk in 1:dim
            vec_exp[nk] = vec_ave[nk] + coeff_e*(vec_ref[nk]-vec_ave[nk])
        end
        vec_exp = vec_exp/norm(vec_exp)

        F_e = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_exp*vec_exp', NM.Liouville_operator, NM.γ, NM.times)
        fe = 1.0/real(tr(NM.W*pinv(F_e)))

        if fe <= fr
            for np in 1:dim
                nelder_mead[sort_ind[end]].psi[np] = vec_ref[np]
            end
            p_fit[sort_ind[end]] = fr
            sort_ind = sortperm(p_fit, rev=true)
        else
            for np in 1:dim
                nelder_mead[sort_ind[end]].psi[np] = vec_exp[np]
            end
            p_fit[sort_ind[end]] = fe
            sort_ind = sortperm(p_fit, rev=true)
        end
    elseif fr < p_fit[sort_ind[end-1]]
        # constraction
        if fr <= p_fit[sort_ind[end]]
            # inside constraction
            vec_ic = zeros(ComplexF64, dim)
            for nl in 1:dim
                vec_ic[nl] = vec_ave[nl] - coeff_c*(vec_ave[nl]-nelder_mead[sort_ind[end]].psi[nl])
            end
            vec_ic = vec_ic/norm(vec_ic)

            F_ic = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_ic*vec_ic', NM.Liouville_operator, NM.γ, NM.times)
            fic = 1.0/real(tr(NM.W*pinv(F_ic)))

            if fic > p_fit[sort_ind[end]]
                for np in 1:dim
                    nelder_mead[sort_ind[end]].psi[np] = vec_ic[np]
                end
                p_fit[sort_ind[end]] = fic
                sort_ind = sortperm(p_fit, rev=true)
            else
                # shrink
                vec_first = [nelder_mead[sort_ind[1]].psi[i] for i in 1:dim]
                for pk in 1:state_num
                    for nq in 1:dim
                        nelder_mead[pk].psi[nq] = vec_first[nq] + coeff_s*(nelder_mead[pk].psi[nq]-vec_first[nq])
                    end
                    nelder_mead[pk].psi = nelder_mead[pk].psi/norm(nelder_mead[pk].psi)

                    rho = nelder_mead[pk].psi*(nelder_mead[pk].psi)'
                    F_tp = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, rho, NM.Liouville_operator, NM.γ, NM.times)
                    p_fit[pk] = 1.0/real(tr(NM.W*pinv(F_tp)))
                end
                sort_ind = sortperm(p_fit, rev=true)
            end
        else
            # outside constraction
            vec_oc = zeros(ComplexF64, dim)
            for nn in 1:dim
                vec_oc[nn] = vec_ave[nn] + coeff_c*(vec_ref[nn]-vec_ave[nn])
            end
            vec_oc = vec_oc/norm(vec_oc)

            F_oc = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, vec_oc*vec_oc', NM.Liouville_operator, NM.γ, NM.times)
            foc = 1.0/real(tr(NM.W*pinv(F_oc)))

            if foc >= fr
                for np in 1:dim
                    nelder_mead[sort_ind[end]].psi[np] = vec_oc[np]
                end
                p_fit[sort_ind[end]] = foc
                sort_ind = sortperm(p_fit, rev=true)
            else
                # shrink
                vec_first = [nelder_mead[sort_ind[1]].psi[i] for i in 1:dim]
                for pk in 1:state_num
                    for nq in 1:dim
                        nelder_mead[pk].psi[nq] = vec_first[nq] + coeff_s*(nelder_mead[pk].psi[nq]-vec_first[nq])
                    end
                    nelder_mead[pk].psi = nelder_mead[pk].psi/norm(nelder_mead[pk].psi)

                    rho = nelder_mead[pk].psi*(nelder_mead[pk].psi)'
                    F_tp = CFIM_TimeIndepend(M, NM.freeHamiltonian, NM.Hamiltonian_derivative, rho, NM.Liouville_operator, NM.γ, NM.times)
                    p_fit[pk] = 1.0/real(tr(NM.W*pinv(F_tp)))
                end
                sort_ind = sortperm(p_fit, rev=true)
            end
        end
    else
        for np in 1:dim
            nelder_mead[sort_ind[end]].psi[np] = vec_ref[np]
        end
        p_fit[sort_ind[end]] = fr
        sort_ind = sortperm(p_fit, rev=true)
    end
    return p_fit, sort_ind
end
