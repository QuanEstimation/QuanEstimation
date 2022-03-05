function adaptive(x::AbstractVector, p, rho0::AbstractMatrix, M::AbstractVector, K, 
    dK; W::Union{Matrix,Nothing}=nothing, max_episode::Int=1000, eps::Float64=1e-8, save_file=false)
    dim = size(rho0)[1]
    para_num = length(x)

    if W == nothing
        W = zeros(para_num, para_num)
    end
    
    if para_num == 1
        #### singleparameter senario ####
        p_num = length(p)
        
        F = zeros(length(p_num))
        for hi in 1:length(p_num)
            rho_tp = sum([Ki*rho0*Ki' for Ki in K[hi]]) 
            drho_tp = [sum([dKi*rho0*Ki' + Ki*rho0*dKi' for (Ki,dKi) in zip(K[hi],dKj)]) for dKj in dK[hi]]
            F[hi] = CFI(rho_tp, drho_tp[1], M; eps=eps)
        end
        indx = findmax(F)[2]
        x_opt = x[1][indx]
        println("The optimal parameter is $x_opt")

        u = 0.0
        y, xout, pout = [], [], []
        for ei in 1:max_episode
            rho = [zeros(ComplexF64, dim, dim) for i in 1:p_num]
            for hj in 1:p_num
                x_idx = findmin(abs.(x[1] .- (x[1][hj]+u)))[2]
                rho[hj] = sum([Ki*rho0*Ki' for Ki in K[x_idx]])
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
        x_list = [(Iterators.product(x...))...]

        rho_tp = [sum([Ki*rho0*Ki' for Ki in K_tp]) for K_tp in K]
        
        drho_tp = [[sum([dKi*rho0*Ki' + Ki*rho0*dKi' for (Ki,dKi) in zip(K_tp,dKj)]) for dKj in dK_tp] for (K_tp,dK_tp) in zip(K,dK)]
        #### data format for dK
        F_all = zeros(length(p |> vec))
        for hi in 1:length(p |> vec) 
            F_tp = QFIM(rho_tp[hi], drho_tp[hi], eps)
            F_all[hi] = abs(det(F_tp)) < eps ? eps : 1.0/real(tr(W*inv(F_tp)))
        end

        F = reshape(F_all, size(p))
        idx = findmax(F)[2]
        x_opt = [x[i][idx[i]] for i in 1:para_num]
        println("The optimal parameter are $x_opt")

        u = [0.0 for i in 1:para_num]
        y, xout, pout = [], [], []
        for ei in 1:max_episode
            rho = Array{Matrix{ComplexF64}}(undef, length(p|>vec))
            for hj in 1:length(p|>vec)
                x_indx = [findmin(abs.(x[k] .- (x_list[hj][k]+u[k])))[2] for k in 1:para_num]
                rho_tp = sum([Ki*rho0*Ki' for Ki in K[x_indx...]]) 
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
            append!(y, Int(res_exp+1))
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
