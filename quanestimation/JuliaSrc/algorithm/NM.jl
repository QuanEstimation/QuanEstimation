function update!(opt::StateOpt, alg::NM, obj, dynamics, output)
    (; max_episode, state_num, ini_state, ar, ae, ac, as0, rng) = alg
    if ismissing(ini_state)
        ini_state = [opt.ψ₀]
    end
    dim = length(dynamics.data.ψ0)
    nelder_mead = repeat(dynamics, state_num)

    # initialize 
    if length(ini_state) > state_num
        ini_state = [ini_state[i] for i in 1:state_num]
    end 
    for pj in 1:length(ini_state)
        nelder_mead[pj].data.ψ0 = [ini_state[pj][i] for i in 1:dim]
    end
    for pj in (length(ini_state)+1):state_num
        r_ini = 2*rand(rng, dim)-ones(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(rng, dim)
        nelder_mead[pj].data.ψ0 = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end
    
    p_fit, p_out = zeros(state_num), zeros(state_num)
    for pj in 1:state_num
        p_out[pj], p_fit[pj] = objective(obj, nelder_mead[pj])
    end
    sort_ind = sortperm(p_fit, rev=true)

    set_f!(output, p_out[1])
    set_buffer!(output, transpose(dynamics.data.ψ0))
    set_io!(output, p_out[1])
    show(opt, output, obj)

    f_list = [p_out[1]]
    idx = 0
    for ei in 1:(max_episode-1)
        # calculate the average vector
        vec_ave = zeros(ComplexF64, dim)
        for ni in 1:dim
            vec_ave[ni] = [nelder_mead[pk].data.ψ0[ni] for pk in 1:(state_num-1)] |> sum
            vec_ave[ni] = vec_ave[ni]/(state_num-1)
        end
        vec_ave = vec_ave/norm(vec_ave)

        # reflection
        vec_ref = zeros(ComplexF64, dim)
        for nj in 1:dim
            vec_ref[nj] = vec_ave[nj] + ar*(vec_ave[nj]-nelder_mead[sort_ind[end]].data.ψ0[nj])
        end
        vec_ref = vec_ref/norm(vec_ref)
        dynamics_copy = set_state(dynamics, vec_ref)
        fr_out, fr = objective(obj, dynamics_copy)

        if fr > p_fit[sort_ind[1]]
            # expansion
            vec_exp = zeros(ComplexF64, dim)
            for nk in 1:dim
                vec_exp[nk] = vec_ave[nk] + ae*(vec_ref[nk]-vec_ave[nk])
            end
            vec_exp = vec_exp/norm(vec_exp)
            dynamics_copy = set_state(dynamics, vec_exp)
            fe_out, fe = objective(obj, dynamics_copy)
            if fe <= fr
                for np in 1:dim
                    nelder_mead[sort_ind[end]].data.ψ0[np] = vec_ref[np]
                end
                p_fit[sort_ind[end]] = fr
                p_out[sort_ind[end]] = fr_out
                sort_ind = sortperm(p_fit, rev=true)
            else
                for np in 1:dim
                    nelder_mead[sort_ind[end]].data.ψ0[np] = vec_exp[np]
                end
                p_fit[sort_ind[end]] = fe
                p_out[sort_ind[end]] = fe_out
                sort_ind = sortperm(p_fit, rev=true)
            end
        elseif fr < p_fit[sort_ind[end-1]]
            # constraction
            if fr <= p_fit[sort_ind[end]]
                # inside constraction
                vec_ic = zeros(ComplexF64, dim)
                for nl in 1:dim
                    vec_ic[nl] = vec_ave[nl] - ac*(vec_ave[nl]-nelder_mead[sort_ind[end]].data.ψ0[nl])
                end
                vec_ic = vec_ic/norm(vec_ic)
                dynamics_copy = set_state(dynamics, vec_ic)
                fic_out, fic = objective(obj, dynamics_copy)
                if fic > p_fit[sort_ind[end]]
                    for np in 1:dim
                        nelder_mead[sort_ind[end]].data.ψ0[np] = vec_ic[np]
                    end
                    p_fit[sort_ind[end]] = fic
                    p_out[sort_ind[end]] = fic_out
                    sort_ind = sortperm(p_fit, rev=true)
                else
                    # shrink
                    vec_first = [nelder_mead[sort_ind[1]].data.ψ0[i] for i in 1:dim]
                    for pk in 1:state_num
                        for nq in 1:dim
                            nelder_mead[pk].data.ψ0[nq] = vec_first[nq] + as0*(nelder_mead[pk].data.ψ0[nq]-vec_first[nq])
                        end
                        nelder_mead[pk].data.ψ0 = nelder_mead[pk].data.ψ0/norm(nelder_mead[pk].data.ψ0)
                        p_out[pk], p_fit[pk] = objective(obj, nelder_mead[pk])
                    end
                    sort_ind = sortperm(p_fit, rev=true)
                end
            else
                # outside constraction
                vec_oc = zeros(ComplexF64, dim)
                for nn in 1:dim
                    vec_oc[nn] = vec_ave[nn] + ac*(vec_ref[nn]-vec_ave[nn])
                end
                vec_oc = vec_oc/norm(vec_oc)
                dynamics_copy = set_state(dynamics, vec_oc)
                foc_out, foc = objective(obj, dynamics_copy)
                if foc >= fr
                    for np in 1:dim
                        nelder_mead[sort_ind[end]].data.ψ0[np] = vec_oc[np]
                    end
                    p_fit[sort_ind[end]] = foc
                    p_out[sort_ind[end]] = foc_out
                    sort_ind = sortperm(p_fit, rev=true)
                else
                    # shrink
                    vec_first = [nelder_mead[sort_ind[1]].data.ψ0[i] for i in 1:dim]
                    for pk in 1:state_num
                        for nq in 1:dim
                            nelder_mead[pk].data.ψ0[nq] = vec_first[nq] + as0*(nelder_mead[pk].data.ψ0[nq]-vec_first[nq])
                        end
                        nelder_mead[pk].data.ψ0 = nelder_mead[pk].data.ψ0/norm(nelder_mead[pk].data.ψ0)
                        p_out[pk], p_fit[pk] = objective(obj, nelder_mead[pk])
                    end
                    sort_ind = sortperm(p_fit, rev=true)
                end
            end
        else
            for np in 1:dim
                nelder_mead[sort_ind[end]].data.ψ0[np] = vec_ref[np]
            end
            p_fit[sort_ind[end]] = fr
            p_out[sort_ind[end]] = fr_out
            sort_ind = sortperm(p_fit, rev=true)
        end
        idx = findmax(p_fit)[2]
        set_f!(output, p_out[idx])
        set_buffer!(output, transpose(nelder_mead[sort_ind[1]].data.ψ0))
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, p_out[idx])
end
