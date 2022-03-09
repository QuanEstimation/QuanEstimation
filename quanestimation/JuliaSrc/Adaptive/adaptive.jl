function adaptive(x::AbstractVector, p, rho0::AbstractMatrix, tspan, H, dH; decay::Union{Vector,Nothing}=nothing, 
                Hc::Union{Vector,Nothing}=nothing, ctrl::Union{Vector,Nothing}=nothing, M::Union{AbstractVector,Nothing}=nothing, 
                W::Union{Matrix,Nothing}=nothing, max_episode::Int=1000, eps::Float64=1e-8, save_file=false)
    dim = size(rho0)[1]
    para_num = length(x)
    if M == nothing
        M = SIC(size(rho0)[1])
    end
    if decay == nothing
        decay_opt = [zeros(ComplexF64, dim, dim)]
        gamma = [0.0]
    else
        decay_opt = [decay[i][1] for i in 1:len(decay)]
        gamma = [decay[i][2] for i in 1:len(decay)]
    end
    if Hc == nothing
        Hc = [zeros(ComplexF64, dim, dim)]
        ctrl = [zeros(length(tspan)-1)]
    elseif ctrl == nothing
        ctrl = [zeros(length(tspan)-1) for j in range(length(Hc))]
    else
        ctrl_length = length(ctrl)
        ctrlnum = length(Hc)
        if ctrlnum < ctrl_length
            throw("There are $ctrlnum control Hamiltonians but $ctrl_length coefficients sequences: too many coefficients sequences")
        elseif ctrlnum > ctrl_length 
            println("Not enough coefficients sequences: there are $ctrlnum control Hamiltonians 
                     but $ctrl_length coefficients sequences. The rest of the control sequences are set to be 0.")
            number = ceil((length(tspan)-1)/length(ctrl[1]))
            if mod(length(tspan)-1, length(ctrl[1])) != 0
                tnum = number*length(ctrl[1])
                tspan = arange(tspan[1], stop=tspan[end], length=tnum+1) |> collect
            end
        end
    end
    if W == nothing
        W = zeros(para_num, para_num)
    end    
    
    if para_num == 1
        #### singleparameter senario ####
        p_num = length(p)

        F = zeros(p_num)
        for hi in 1:p_num
            rho_tp, drho_tp = dynamics(H[hi], dH[hi][1], rho0, decay_opt, gamma, Hc, ctrl, tspan)
            F[hi] = CFI(rho_tp, drho_tp, M, eps)
        end
        idx = findmax(F)[2]
        x_opt = x[1][idx]
        println("The optimal parameter is $x_opt")

        u = 0.0
        y, xout, pout = [], [], []
        for ei in 1:max_episode
            rho = [zeros(ComplexF64, dim, dim) for i in 1:p_num]
            for hj in 1:p_num
                x_idx = findmin(abs.(x[1] .- (x[1][hj]+u)))[2]
                H_tp = H[x_idx]
                dH_tp = dH[x_idx]
                rho_tp, drho_tp = dynamics(H_tp, dH_tp[1], rho0, decay_opt, gamma, Hc, ctrl, tspan)
                rho[hj] = rho_tp
            end
            println("The tunable parameter is $u")
            print("Please enter the experimental result: ")
            enter = readline()
            res_exp = parse(Int64, enter)
            res_exp = Int(res_exp+1)

            pyx = real.(tr.(rho .* [M[res_exp]]))

            py = trapz(x[1], pyx.*p)
            p_update = pyx.*p/py
            p = p_update 
            p_idx = findmax(p)[2]
            x_out = x[1][p_idx]
            println("The estimator is $x_out ($ei episodes)")
            u = x_opt - x_out

            if mod(ei, 50) == 0
                if (x_out+u) > x[1][end] || (x_out+u) < x[1][1]
                    throw("please increase the regime of the parameters.")
                end
            end
            append!(xout, x_out)
            append!(y, Int(res_exp-1))
            append!(pout, [p])
        end
        if save_file == false
            open("pout.csv","w") do f
                writedlm(f, [p])
            end
            open("xout.csv","w") do m
                writedlm(m, xout)
            end
            open("y.csv","w") do n
                writedlm(n, y)
            end
        else
            open("pout.csv","w") do f
                writedlm(f, pout)
            end
            open("xout.csv","w") do m
                writedlm(m, xout)
            end
            open("y.csv","w") do n
                writedlm(n, y)
            end
        end
    else
        #### multiparameter senario ####
        p_num = length(p|>vec)
        x_list = [(Iterators.product(x...))...]

        dynamics_res = [dynamics(H_tp, dH_tp, rho0, decay_opt, gamma, Hc, ctrl, tspan) for (H_tp, dH_tp) in zip(H, dH)]
        F_all = zeros(p_num)
        for hi in 1:p_num
            F_tp = CFIM(dynamics_res[hi][1], dynamics_res[hi][2], M, eps)
            F_all[hi] = abs(det(F_tp)) < eps ? eps : 1.0/real(tr(W*inv(F_tp)))
        end
        F = reshape(F_all, size(p))
        idx = findmax(F)[2]
        x_opt = [x[i][idx[i]] for i in 1:para_num]
        println("The optimal parameter are $x_opt")

        u = [0.0 for i in 1:para_num]
        y, xout, pout = [], [], []
        for ei in 1:max_episode
            rho = Array{Matrix{ComplexF64}}(undef, p_num)
            for hj in 1:p_num
                x_idx = [findmin(abs.(x[k] .- (x_list[hj][k]+u[k])))[2] for k in 1:para_num]
                H_tp = H[x_idx...]
                dH_tp = dH[x_idx...]
                rho_tp, drho_tp = dynamics(H_tp, dH_tp, rho0, decay_opt, gamma, Hc, ctrl, tspan)
                rho[hj] = rho_tp
            end

            println("The tunable parameter are $u")
            print("Please enter the experimental result: ")
            enter = readline()
            res_exp = parse(Int64, enter)
            res_exp = Int(res_exp+1)

            pyx_list = real.(tr.(rho.*[M[res_exp]]))
            pyx = reshape(pyx_list, size(p))

            arr = p.*pyx
            py = trapz(tuple(x...), arr)
            p_update = p.*pyx/py
            p = p_update

            p_idx = findmax(p)[2]
            x_out = [x[i][p_idx[i]] for i in 1:para_num]
            println("The estimator are $x_out ($ei episodes)")
            u = x_out .- x_opt

            if mod(ei, 50) == 0
                for un in 1:para_num
                    if (x_out[un]+u[un]) > x[un][end] || (x_out[un]+u[un]) < x[un][1]
                        throw("Please increase the regime of the parameters.")
                    end
                end
            end
            append!(xout, [x_out])
            append!(y, Int(res_exp-1))
            append!(pout, [p])
        end
        if save_file == false
            open("pout.csv","w") do f
                writedlm(f, [p])
            end
            open("xout.csv","w") do m
                writedlm(m, xout)
            end
            open("y.csv","w") do n
                writedlm(n, y)
            end
        else
            open("pout.csv","w") do f
                writedlm(f, pout)
            end
            open("xout.csv","w") do m
                writedlm(m, xout)
            end
            open("y.csv","w") do n
                writedlm(n, y)
            end
        end
    end
end
