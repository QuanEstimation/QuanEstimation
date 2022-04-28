"""

    Bayes(x, p, rho, y; M=nothing, savefile=false)

Bayesian estimation. The prior distribution is updated via the posterior distribution obtained by the Bayes’ rule and the estimated value of parameters obtained via the maximum a posteriori probability (MAP).
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `rho`: Parameterized density matrix.
- `y`: The experimental results obtained in practice.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `savefile`: Whether or not to save all the posterior distributions. 
"""
function Bayes(x, p, rho, y; M=nothing, estimator="mean", savefile=false)
    y = y .+ 1
    para_num = length(x)
    max_episode = length(y)
    if para_num == 1
        #### singleparameter senario ####
        if M==nothing
            M = SIC(size(rho[1])[1])
        end
        if savefile == false
            x_out = []
            if estimator == "mean"
                for mi in 1:max_episode
                    res_exp = y[mi] |> Int
                    pyx = real.(tr.(rho.*[M[res_exp]]))
                    py = trapz(x[1], pyx.*p)
                    p_update = pyx.*p/py
                    p = p_update
                    append!(x_out, trapz(x[1], p.*x[1]))
                end
            elseif estimator == "MAP"
                for mi in 1:max_episode
                    res_exp = y[mi] |> Int
                    pyx = real.(tr.(rho.*[M[res_exp]]))
                    py = trapz(x[1], pyx.*p)
                    p_update = pyx.*p/py
                    p = p_update
    
                    indx = findmax(p)[2]
                    append!(x_out, x[1][indx])
                end
            else
               println(
                "The input is not a valid value for estimator, supported values are 'mean' and 'MAP'.")
            end
            
            open("pout.csv","w") do f
                writedlm(f, [p])
            end
            open("xout.csv","w") do m
                writedlm(m, x_out)
            end
            return p, x_out[end]
        else
            p_out, x_out = [], []
            if estimator == "mean"
                for mi in 1:max_episode
                    res_exp = y[mi] |> Int
                    pyx = real.(tr.(rho.*[M[res_exp]]))
                    py = trapz(x[1], pyx.*p)
                    p_update = pyx.*p/py
                    p = p_update
                    
                    append!(p_out, [p])
                    append!(x_out, trapz(x[1], p.*x[1]))
                end
            elseif estimator == "MAP"
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
            else
               println(
                "The input is not a valid value for estimator, supported values are 'mean' and 'MAP'.")
            end
            
            open("pout.csv","w") do f
                writedlm(f, [p_out])
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
            x_out = []
            if estimator == "mean"
                for mi in 1:max_episode
                    res_exp = y[mi] |> Int
                    pyx = real.(tr.(rho.*[M[res_exp]]))
                    arr = p.*pyx
                    py = trapz(tuple(x...), arr)
    
                    p_update = p.*pyx/py
                    p = p_update
                    append!(x_out, [integ(x, p)])
                end
            elseif estimator == "MAP"
                for mi in 1:max_episode
                    res_exp = y[mi] |> Int
                    pyx = real.(tr.(rho.*[M[res_exp]]))
                    arr = p.*pyx
                    py = trapz(tuple(x...), arr)
    
                    p_update = p.*pyx/py
                    p = p_update
    
                    indx = findmax(p)[2]
                    append!(x_out, [[x[i][indx[i]] for i in 1:para_num]])
                end
            else
               println(
                "The input is not a valid value for estimator, supported values are 'mean' and 'MAP'.")
            end
            
            open("pout.csv","w") do f
                writedlm(f, [p])
            end
            open("xout.csv","w") do m
                writedlm(m, x_out)
            end
            return p, x_out[end]
        else
            p_out, x_out = [], []
            if estimator == "mean"
                for mi in 1:max_episode
                    res_exp = y[mi] |> Int
                    pyx = real.(tr.(rho.*[M[res_exp]]))
                    arr = p.*pyx
                    py = trapz(tuple(x...), arr)
                    p_update = p.*pyx/py
                    p = p_update
                    
                    append!(p_out, [p])
                    append!(x_out, [integ(x,p)])
                end
            elseif estimator == "MAP"
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
            else
               println(
                "The input is not a valid value for estimator, supported values are 'mean' and 'MAP'.")
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

"""

    Bayes(x, p, rho, y; M=nothing, savefile=false)

Bayesian estimation. The estimated value of parameters obtained via the maximum likelihood estimation (MLE).
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `rho`: Parameterized density matrix.
- `y`: The experimental results obtained in practice.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `savefile`: Whether or not to save all the posterior distributions. 
"""
function MLE(x, rho, y; M::Union{AbstractVector,Nothing}=nothing, savefile=false)
    y = y .+ 1
    para_num = length(x)
    max_episode = length(y)
    if para_num == 1
        if M==nothing
            M = SIC(size(rho[1])[1])
        end
        if savefile == false
            x_out = []
            L_out = ones(length(x[1]))
            for mi in 1:max_episode
                res_exp = y[mi] |> Int
                p_tp = real.(tr.(rho.*[M[res_exp]]))
                L_out = L_out.*p_tp
                indx = findmax(L_out)[2]
                append!(x_out, x[1][indx])
            end
             
            open("Lout.csv","w") do f
                writedlm(f, [L_out])
            end
            open("xout.csv","w") do m
                writedlm(m, x_out)
            end
            return L_out, x_out[end]
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
            x_out = []
            L_out = ones(p_shape...)
            for mi in 1:max_episode
                res_exp = y[mi] |> Int
                p_tp = real.(tr.(rho.*[M[res_exp]]))
                L_out = L_out.*p_tp
                indx = findmax(L_out)[2]
                append!(x_out, [[x[i][indx[i]] for i in 1:para_num]])
            end
            open("Lout.csv","w") do f
                writedlm(f, [L_out])
            end
            open("xout.csv","w") do m
                writedlm(m, x_out)
            end
            return L_out, x_out[end]
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

function integ(x, p)
    para_num = length(x)
    mean = [0.0 for i in 1:para_num]
    for i in 1:para_num
        p_tp = p
        if i == para_num
            for si = 1:para_num-1
                p_tp = trapz(x[si], p_tp, Val(1))
            end
        
        elseif i == 1
            for si = para_num:2 
                p_tp = trapz(x[si], p_tp)
            end
        else
            p_tp = trapz(x[end], p_tp)
            for si = 1:para_num-1
                p_tp = trapz(x[si], p_tp, Val(1))
            end
        end
        mean[i] = trapz(x[i], x[i].*p_tp)
    end
    mean
end
        