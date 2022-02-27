mutable struct adapt{T<:Complex, M<:Real}
    x::Vector{Vector{M}}
    H
    dH
    ρ0::Matrix{T}
    tspan::Vector{M}
    decay_opt::Vector{Matrix{T}}
    γ::Vector{M}
    W::Matrix{M}
    eps::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    adapt(x::Vector{Vector{M}}, H, dH, ρ0::Matrix{T}, tspan::Vector{M}, decay_opt::Vector{Matrix{T}}, 
        γ::Vector{M}, W::Matrix{M}, eps::M, ρ=Vector{Matrix{T}}(undef, 1), ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where 
        {T<:Complex, M<:Real} = new{T,M}(x, H, dH, ρ0, tspan, decay_opt, γ, W, eps, ρ, ∂ρ_∂x) 
end

function adaptive(apt::adapt{T}, M, p, max_episode, save_file) where {T<:Complex}
    para_num = length(apt.x)
    dim = size(apt.ρ0)[1]
    
    if para_num == 1
        #### singleparameter senario ####
        p_indx = findall(x -> x > apt.eps, p)
        p = p[p_indx]
        xspan = apt.x[1][p_indx]
        p_num = length(p)

        F = zeros(length(apt.x[1]))
        for hi in 1:length(apt.x[1])
            rho_tp, drho_tp = dynamics_TimeIndepend(apt.H[hi], apt.dH[hi], apt.ρ0, apt.decay_opt, apt.γ, apt.tspan)
            F[hi] = CFI(rho_tp, drho_tp, M; eps=apt.eps)
        end
        indx = findmax(F)[2]
        x_opt = apt.x[1][indx]
        println("The optimal parameter is $x_opt")

        u = 0.0
        if save_file == false
            x_out = 0.0
            for ei in 1:max_episode
                rho = [zeros(ComplexF64, dim, dim) for i in 1:p_num]
                for hj in 1:p_num
                    x_indx = findmin(abs.(apt.x[1] .- (xspan[hj]+u)))
                    H_tp = apt.H[x_indx[2]]
                    dH_tp = apt.dH[x_indx[2]]
                    rho_tp, drho_tp = dynamics_TimeIndepend(H_tp, dH_tp, apt.ρ0, apt.decay_opt, apt.γ, apt.tspan)
                    rho[hj] = rho_tp
                end

                println("Please enter the experimental result")
                enter = readline()
                res_exp = parse(Int64, enter)

                pyx = zeros(p_num)
                for xi in 1:p_num
                    pyx[xi] = real(tr(rho[xi]*M[res_exp]))
                end
                arr = [pyx[m]*p[m] for m in 1:p_num]
                py = trapz(xspan, arr)
                p_update = pyx.*p/py
                p = p_update
                indx_p = findmax(p)[2]
                x_out = xspan[indx_p]
                println("The estimator is $x_out")
                u = x_opt - x_out

                if mod(ei, 50) == 0
                    if (x_out+u) > apt.x[1][end] || (x_out+u) < apt.x[1][1]
                        throw("please increase the regime of the parameters.")
                    end
                end
            end
            return p, x_out
        else
            pout, xout = [], []
            x_out = 0.0
            for ei in 1:max_episode
                rho = [zeros(ComplexF64, dim, dim) for i in 1:p_num]
                for hj in 1:p_num
                    x_indx = findmin(abs.(apt.x[1] .- (xspan[hj]+u)))
                    H_tp = apt.H[x_indx[2]]
                    dH_tp = apt.dH[x_indx[2]]
                    rho_tp, drho_tp = dynamics_TimeIndepend(H_tp, dH_tp, apt.ρ0, apt.decay_opt, apt.γ, apt.tspan)
                    rho[hj] = rho_tp 
                end

                println("Please enter the experimental result")
                enter = readline()
                res_exp = parse(Int64, enter)

                pyx = zeros(p_num)
                for xi in 1:p_num
                    pyx[xi] = real(tr(rho[xi]*M[res_exp]))
                end
                arr = [pyx[m]*p[m] for m in 1:p_num]
                py = trapz(xspan, arr)
                p_update = pyx.*p/py
                p = p_update
                indx_p = findmax(p)[2]
                x_out = xspan[indx_p]
                println("The estimator is $x_out")
                u = x_opt - x_out

                if mod(ei, 50) == 0
                    if (x_out+u) > apt.x[1][end] || (x_out+u) < apt.x[1][1]
                        throw("please increase the regime of the parameters.")
                    end
                end
                append!(xout, x_out)
                append!(pout, [p])
            end
            open("pout.csv","w") do f
                writedlm(f, pout)
            end
            open("xout.csv","w") do m
                writedlm(m, xout)
            end
            return p, x_out
        end
    else
        #### multiparameter senario ####
        x_list = [(Iterators.product(apt.x...))...]
        # xspan = reshape(x_list, size(p))[findall(x-> abs(x) > apt.eps, p)]
        # max_value = [maximum([xspan[i][j] for i in 1:[size(p)...][1]]) for j in 1:para_num]
        # min_value = [minimum([xspan[i][j] for i in 1:[size(p)...][1]]) for j in 1:para_num]

        H_tp = [[h...] for h in apt.H]
        dH_tp = [[dh...] for dh in apt.dH]

        F_all = zeros(length(p |> vec))
        for hi in 1:length(p |> vec)
            H_hi = H_tp[hi][1]
            dH_hi = [dH_tp[j][1][hi] for j in 1:para_num]
            rho_tp, drho_tp = dynamics_TimeIndepend(H_hi, dH_hi, apt.ρ0, apt.decay_opt, apt.γ, apt.tspan) 
            F_all[hi] = tr(apt.W*pinv(QFIM(rho_tp, drho_tp, apt.eps)))
        end
        F = reshape(F_all, reverse(size(p))) |> transpose
        indx = findmax(F)[2]
        x_opt = [apt.x[i][indx[i]] for i in 1:para_num]
        println("The optimal parameter is $x_opt")

        u = [0.0 for i in 1:para_num]
        if save_file == false
            x_out = [0.0 for i in 1:para_num]
            for ei in 1:max_episode
                rho = Array{Matrix{ComplexF64}}(undef, length(p|>vec))
                for hj in 1:length(p|>vec)
                    x_indx = [findmin(abs.(apt.x[k] .- (x_list[hj][k]+u[k]))) for k in 1:para_num]
                    x_indx = reverse(x_indx)

                    hi = sum([(x_indx[j][2]-1)*length(apt.x[j]) for j in para_num:2]) + x_indx[1][2]
                    H_hj = H_tp[hi][1]
                    dH_hj = [dH_tp[j][1][hi] for j in 1:para_num]
                    rho_tp, drho_tp = dynamics_TimeIndepend(H_hj, dH_hj, apt.ρ0, apt.decay_opt, apt.γ, apt.tspan)
                    rho[hj] = rho_tp
                end
                rho = reshape(rho, reverse(size(p))) |> transpose

                println("Please enter the experimental result")
                enter = readline()
                res_exp = parse(Int64, enter)

                pyx = real.(tr.(rho.*[M[res_exp]]))
                arr = p.*pyx
                py = trapz(tuple(apt.x...), arr)
                p_update = p.*pyx/py
                p = p_update

                indx_p = findmax(p)[2]
                x_out = [apt.x[i][indx_p[i]] for i in 1:para_num]
                println("The estimator is $x_out")
                u = x_out .- x_opt

                if mod(ei, 50) == 0
                    for un in 1:para_num
                        if (x_out[un]+u[un]) > apt.x[un][end] || (x_out[un]+u[un]) < apt.x[un][1]
                            throw("Please increase the regime of the parameters.")
                        end
                    end
                end
            end
            return p, x_out
        else
            pout, xout = [], []
            x_out = [0.0 for i in 1:para_num]
            for ei in 1:max_episode
                rho = Array{Matrix{ComplexF64}}(undef, length(p|>vec))
                for hj in 1:length(p|>vec)
                    x_indx = [findmin(abs.(apt.x[k] .- (x_list[hj][k]+u[k]))) for k in 1:para_num]
                    x_indx = reverse(x_indx)
                    hi = sum([(x_indx[j][2]-1)*length(apt.x[j]) for j in para_num:2]) + x_indx[1][2]
                    H_hj = H_tp[hi][1]
                    dH_hj = [dH_tp[j][1][hi] for j in 1:para_num]
                    rho_tp, drho_tp = dynamics_TimeIndepend(H_hj, dH_hj, apt.ρ0, apt.decay_opt, apt.γ, apt.tspan)
                    rho[hj] = rho_tp
                end
                rho = reshape(rho, reverse(size(p))) |> transpose

                println("Please enter the experimental result")
                enter = readline()
                res_exp = parse(Int64, enter)

                pyx = real.(tr.(rho.*[M[res_exp]]))
                arr = p.*pyx
                py = trapz(tuple(apt.x...), arr)
                p_update = p.*pyx/py
                p = p_update

                indx_p = findmax(p)[2]
                x_out = [apt.x[i][indx_p[i]] for i in 1:para_num]
                println("The estimator is $x_out")

                u = x_out .- x_opt
                if mod(ei, 50) == 0
                    for un in 1:para_num
                        if (x_out[un]+u[un]) > apt.x[un][end] || (x_out[un]+u[un]) < apt.x[un][1]
                            throw("Please increase the regime of the parameters.")
                        end
                    end
                end
                append!(pout, [p])
                append(xout, x_out)
            end
            open("pout.csv","w") do f
                writedlm(f, pout)
            end
            open("xout.csv","w") do m
                writedlm(m, xout)
            end
            return p, x_out
        end
    end
end
