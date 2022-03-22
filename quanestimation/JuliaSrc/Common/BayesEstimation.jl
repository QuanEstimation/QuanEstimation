function Bayes(x, p, rho, y; M=nothing, savefile=false)
    y = y .+ 1
    para_num = length(x)
    max_episode = length(y)
    if para_num == 1
        #### singleparameter senario ####
        if M==nothing
            M = SIC(size(rho[1])[1])
        end
        if savefile == false
            for mi in 1:max_episode
                res_exp = y[mi] |> Int
                pyx = real.(tr.(rho.*[M[res_exp]]))
                py = trapz(x[1], pyx.*p)
                p_update = pyx.*p/py
                p = p_update
            end
            indx = findmax(p)[2]
            x_out = x[1][indx]
            return p, x_out
        else
            p_out, x_out = [], []
            for mi in 1:max_episode
                res_exp = y[mi] |> Int
                pyx = real.(tr.(rho.*[M[res_exp]]))
                py = trapz(x[1], pyx.*p)
                p_update = pyx.*p/py
                p = p_update
                
                indx = findmax(p)[2]
                append!(p_out, [p])
                append!(x_out, x[1][indx])
            end
            
            open("pout.csv","w") do f
                writedlm(f, p_out)
            end
            open("xout.csv","w") do m
                writedlm(m, x_out)
            end
            return p, x_out[end]
        end
    else 
        #### multiparameter senario ####
        if M==nothing
            M = SIC(size(vec(rho)[1])[1])
        end
        if savefile == false
            for mi in 1:max_episode
                res_exp = y[mi] |> Int
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
                res_exp = y[mi] |> Int
                pyx = real.(tr.(rho.*[M[res_exp]]))
                arr = p.*pyx
                py = trapz(tuple(x...), arr)
                p_update = p.*pyx/py
                p = p_update
                
                indx = findmax(p)[2]
                append!(p_out, [p])
                append!(x_out, [[x[i][indx[i]] for i in 1:para_num]])
            end
            
            open("pout.csv","w") do f
                writedlm(f, p_out)
            end
            open("xout.csv","w") do m
                writedlm(m, x_out)
            end
            return p, x_out[end]
        end
    end
end

function MLE(x, rho, y; M::Union{AbstractVector,Nothing}=nothing, savefile=false)
    y = y .+ 1
    para_num = length(x)
    max_episode = length(y)
    if para_num == 1
        if M==nothing
            M = SIC(size(rho[1])[1])
        end
        if savefile == false
            L_out = ones(length(x[1]))
            for mi in 1:max_episode
                res_exp = y[mi] |> Int
                p_tp = real.(tr.(rho.*[M[res_exp]]))
                L_out = L_out.*p_tp
            end
            indx = findmax(L_out)[2]
            x_out = x[1][indx] 
            return L_out, x_out
        else
            L_out, x_out = [], []
            L_tp = ones(length(x[1]))
            for mi in 1:max_episode
                res_exp = y[mi] |> Int
                p_tp = real.(tr.(rho.*[M[res_exp]]))
                L_tp = L_tp.*p_tp

                indx = findmax(L_tp)[2]
                append!(L_out, [L_tp])
                append!(x_out, x[1][indx])
            end
            
            open("Lout.csv","w") do f
                writedlm(f, L_out)
            end
            open("xout.csv","w") do m
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

        if M==nothing
            M = SIC(size(vec(rho)[1])[1])
        end

        if savefile == false
            L_out = ones(p_shape...)
            for mi in 1:max_episode
                res_exp = y[mi] |> Int
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
                res_exp = y[mi] |> Int
                p_tp = real.(tr.(rho.*[M[res_exp]]))
                L_tp = L_tp.*p_tp
                indx = findmax(L_tp)[2]
                append!(L_out, [L_tp])
                append!(x_out, [[x[i][indx[i]] for i in 1:para_num]])
            end

            open("Lout.csv","w") do f
                writedlm(f, L_out)
            end
            open("xout.csv","w") do m
                writedlm(m, x_out)
            end
            return L_tp, x_out[end]
        end
    end
end
