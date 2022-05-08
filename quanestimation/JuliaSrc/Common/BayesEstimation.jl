"""

    Bayes(x, p, rho, y; M=missing, savefile=false)

Bayesian estimation. The prior distribution is updated via the posterior distribution obtained by the Bayes’ rule and the estimated value of parameters obtained via the maximum a posteriori probability (MAP).
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `rho`: Parameterized density matrix.
- `y`: The experimental results obtained in practice.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `savefile`: Whether or not to save all the posterior distributions. 
"""
function Bayes(x, p, rho, y; M=missing, estimator="mean", savefile=false)
    y = y .+ 1
    para_num = length(x)
    max_episode = length(y)
    if para_num == 1
        #### singleparameter senario ####
        if ismissing(M)
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
        if ismissing(M)
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

    MLE(x, rho, y; M=missing, savefile=false)

Bayesian estimation. The estimated value of parameters obtained via the maximum likelihood estimation (MLE).
- `x`: The regimes of the parameters for the integral.
- `rho`: Parameterized density matrix.
- `y`: The experimental results obtained in practice.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `savefile`: Whether or not to save all the posterior distributions. 
"""
function MLE(x, rho, y; M=missing, savefile=false)
    y = y .+ 1
    para_num = length(x)
    max_episode = length(y)
    if para_num == 1
        if ismissing(M)
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

        if ismissing(M)
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

"""

    BayesCost(x, p, xest, rho, M; W=missing, eps=GLOBAL_EPS)

Calculation of the average Bayesian cost with a quadratic cost function.
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `xest`: The estimators.
- `rho`: Parameterized density matrix.
- `M`: A set of POVM.
- `W`: Weight matrix.
- `eps`: Machine epsilon.
"""
function BayesCost(x, p, xest, rho, M; W=missing, eps=GLOBAL_EPS)
    para_num = length(x)
    trapzm(x, integrands, slice_dim) = [trapz(tuple(x...), I) for I in [reshape(hcat(integrands...)[i,:], length.(x)...) for i in 1:slice_dim]] 

    if para_num == 1
        # single-parameter scenario
        if ismissing(M)
            M = SIC(size(rho[1])[1])
        end

        p_num = length(x[1])
        value = [p[i]*sum([tr(rho[i]*M[mi])*(x[1][i]-xest[mi][1])^2 for mi in 1:length(M)]) for i in 1:p_num]
        return real(trapz(x[1], value))
    else
        # multi-parameter scenario
        if ismissing(W) 
            W = Matrix(I, para_num, para_num)
        end

        if ismissing(M)
            M = SIC(size(vec(rho)[1])[1])
        end

        x_list = Iterators.product(x...)
        p_num = length(x_list)
        xCx = [sum([tr(rho_i*M[mi])*([xi...]-xest[mi])'*W*([xi...]-xest[mi]) for mi in 1:length(M)]) for (xi,rho_i) in zip(x_list,rho)]
        xCx = reshape(xCx, size(p))

        value = p.*xCx
        for si in reverse(1:para_num)
            value = trapz(x[si], value)
        end
        return real(value)
    end
end

"""

    BCB(x, p, rho; W=missing, eps=GLOBAL_EPS)

Calculation of the minimum Bayesian cost with a quadratic cost function.
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `rho`: Parameterized density matrix.
- `W`: Weight matrix.
- `eps`: Machine epsilon.
"""
function BCB(x, p, rho; W=missing, eps=GLOBAL_EPS)
    para_num = length(x)
    trapzm(x, integrands, slice_dim) = [trapz(tuple(x...), I) for I in [reshape(hcat(integrands...)[i,:], length.(x)...) for i in 1:slice_dim]] 

    if para_num == 1
        # single-parameter scenario
        dim = size(rho[1])[1]
        p_num = length(x[1])
        value = [p[i]*x[1][i]^2 for i in 1:p_num]
        delta2_x = trapz(x[1], value)
        rho_avg = zeros(ComplexF64, dim, dim)
        rho_pri = zeros(ComplexF64, dim, dim)

        for di in 1:dim
            for dj in 1:dim
                rho_avg_arr = [p[m]*rho[m][di,dj] for m in 1:p_num]
                rho_pri_arr = [p[n]*x[1][n]*rho[n][di,dj] for n in 1:p_num]
                rho_avg[di,dj] = trapz(x[1], rho_avg_arr)
                rho_pri[di,dj] = trapz(x[1], rho_pri_arr)
            end
        end
        
        Lambda = Lambda_avg(rho_avg, rho_pri, eps=eps)
        minBC = delta2_x-real(tr(rho_avg*Lambda*Lambda))
        return minBC
    else
        # multi-parameter scenario
        if ismissing(W) 
            W = Matrix(I, para_num, para_num)
        end
        x_list = Iterators.product(x...)
        p_num = length(x_list)
        xCx = [[xi...]'*W*[xi...] for xi in x_list]
        xCx = reshape(xCx, size(p))

        delta2_x = p.*xCx
        for si in reverse(1:para_num)
            delta2_x = trapz(x[si], delta2_x)
        end

        term_tp = [pp*rho_i|>vec for (pp, rho_i) in zip(p, rho)]
        rho_avg = trapzm(x, term_tp, para_num^2) |> I->reshape(I,para_num,para_num)

        x_re = Vector{Float64}[]
        for xj in x_list
            append!(x_re, [[_ for _ in xj]])
        end

        rho_pri = [trapzm(x, term_tp.*reshape([x_re[i][para_i] for i in 1:p_num], size(p)), para_num^2) |> I->reshape(I,para_num,para_num) for para_i in 1:para_num]

        Lambda = Lambda_avg(rho_avg, rho_pri, eps=eps)

        Mat = zeros(ComplexF64, para_num, para_num)
        for para_m in 1:para_num
            for para_n in 1:para_num
                Mat += W[para_m, para_n]*(Lambda[para_m]*Lambda[para_n])
            end
        end

        minBC = delta2_x-real(tr(rho_avg*Mat))
        return minBC
    end
end


function Lambda_avg(rho_avg::Matrix{T}, rho_pri::Matrix{T}; eps=GLOBAL_EPS) where {T<:Complex}
    dim = size(rho_avg)[1]
    Lambda = Matrix{ComplexF64}(undef, dim, dim)

    val, vec = eigen(rho_avg)
    val = val |> real
    Lambda_eig = zeros(T, dim, dim)
    for fi = 1:dim
        for fj = 1:dim
             if abs(val[fi] + val[fj]) > eps
                Lambda_eig[fi, fj] = 2 * (vec[:, fi]' * rho_pri * vec[:, fj]) / (val[fi] + val[fj])
            end
        end
    end
    Lambda_eig[findall(Lambda_eig == Inf)] .= 0.0

    Lambda = vec * (Lambda_eig * vec')
    return Lambda
end

function Lambda_avg(rho_avg::Matrix{T}, rho_pri::Vector{Matrix{T}}; eps = GLOBAL_EPS) where {T<:Complex}
    (x -> SLD(rho_avg, x; eps = eps)).(rho_pri)
end