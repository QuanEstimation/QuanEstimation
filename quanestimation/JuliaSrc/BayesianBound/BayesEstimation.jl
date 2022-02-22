function Bayes(x, p, rho, M, y; save_file=false)
    y = y .+ 1
    para_num = length(x)
    max_episode = length(y)
    if para_num == 1
        #### singleparameter senario ####
        if save_file == false
            for mi in 1:max_episode
                res_exp = y[mi]
                pyx = zeros(length(x[1]))
                for xi in 1:length(x[1])
                    pyx[xi] = real(tr(rho[xi]*M[res_exp]))
                end
                arr = [pyx[m]*p[m] for m in 1:length(x[1])]
                py = trapz(x[1], arr)
                p_update = pyx.*p/py
                p = p_update
            end
            indx = findmax(p)[2]
            x_out = x[1][indx]
            return p, x_out
        else
            p_out, x_out = [], []
            for mi in 1:max_episode
                res_exp = y[mi]
                pyx = zeros(length(x[1]))
                for xi in 1:length(x[1])
                    pyx[xi] = real(tr(rho[xi]*M[res_exp]))
                end
                arr = [pyx[m]*p[m] for m in 1:length(x[1])]
                py = trapz(x[1], arr)
                p_update = pyx.*p/py
                p = p_update
                
                indx = findmax(p)[2]
                append!(p_out, [p])
                append!(x_out, x[1][indx])
            end
            
            open("p_out.csv","w") do f
                writedlm(f, p_out)
            end
            open("x_out.csv","w") do m
                writedlm(m, x_out)
            end
            return p, x_out[end]
        end
    else
        #### multiparameter senario ####
        if save_file == false
            for mi in 1:max_episode
                res_exp = y[mi]
                pyx = real.(tr.(rho.*[M[res_exp]]))
                arr = p.*pyx
                py = trapz(tuple(x...), arr)

                p_update = p.*pyx/py
                p = p_update
            end
            indx = findmax(p)[2]
            x_out = [x[i][indx[i]] for i in 1:para_num]
            return p, x_out
        else
            p_out, x_out = [], []
            for mi in 1:max_episode
                res_exp = y[mi]
                pyx = real.(tr.(rho.*[M[res_exp]]))
                arr = p.*pyx
                py = trapz(tuple(x...), arr)
                p_update = p.*pyx/py
                p = p_update
                
                indx = findmax(p)[2]
                append!(p_out, [p])
                append!(x_out, [[x[i][indx[i]] for i in 1:para_num]])
            end
            
            open("p_out.csv","w") do f
                writedlm(f, p_out)
            end
            open("x_out.csv","w") do m
                writedlm(m, x_out)
            end
            return p, x_out[end]
        end
    end
end

function MLE(x, rho, M, y; save_file=false)
    y = y .+ 1
    para_num = length(x)
    max_episode = length(y)
    if para_num == 1
        if save_file == false
            L_out = ones(length(x[1]))
            for mi in 1:max_episode
                res_exp = y[mi]
                for xi in 1:length(x[1])
                    p_tp = real(tr(rho[xi]*M[res_exp]))
                    L_out[xi] = L_out[xi]*p_tp
                end
            end
            indx = findmax(L_out)[2]
            x_out = x[1][indx] 
            return L_out, x_out
        else
            L_out, x_out = [], []
            L_tp = ones(length(x[1]))
            for mi in 1:max_episode
                res_exp = y[mi]
                for xi in 1:length(x[1])
                    p_tp = real(tr(rho[xi]*M[res_exp]))
                    L_tp[xi] = L_tp[xi]*p_tp
                end
                indx = findmax(L_tp)[2]
                append!(L_out, [L_tp])
                append!(x_out, x[1][indx])
            end
            
            open("L_out.csv","w") do f
                writedlm(f, L_out)
            end
            open("x_out.csv","w") do m
                writedlm(m, x_out)
            end
            return L_tp, x_out[end]
        end
    else
        #### multiparameter senario ####
        p_shape = []
        for i in 1:para_num
            append!(p_shape,length(x[i]))
        end

        if save_file == false
            L_out = ones(p_shape...)
            for mi in 1:max_episode
                res_exp = y[mi]
                p_tp = real.(tr.(rho.*[M[res_exp]]))
                L_out = L_out.*p_tp
            end
            indx = findmax(L_out)[2]
            x_out = [x[i][indx[i]] for i in 1:para_num]
            return L_out, x_out
        else
            L_out, x_out = [], []
            L_tp = ones(p_shape...)
            for mi in 1:max_episode
                res_exp = y[mi]
                p_tp = real.(tr.(rho.*[M[res_exp]]))
                L_tp = L_tp.*p_tp
                indx = findmax(L_tp)[2]
                append!(L_out, [L_tp])
                append!(x_out, [[x[i][indx[i]] for i in 1:para_num]])
            end

            open("L_out.csv","w") do f
                writedlm(f, L_out)
            end
            open("x_out.csv","w") do m
                writedlm(m, x_out)
            end
            return L_tp, x_out[end]
        end
    end
end
