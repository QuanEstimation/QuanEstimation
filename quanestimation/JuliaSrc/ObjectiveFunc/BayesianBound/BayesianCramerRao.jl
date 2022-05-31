########## Bayesian version of CFIM ##########
@doc raw"""

    BCFIM(x::AbstractVector, p, rho, drho; M=missing, eps=GLOBAL_EPS)

Calculation of the Bayesian classical Fisher information (BCFI) and the Bayesian classical Fisher information matrix (BCFIM) of the form
``\mathcal{I}_{\mathrm{Bayes}}=\int p(\textbf{x})\mathcal{I}\mathrm{d}\textbf{x}`` with ``\mathcal{I}`` the CFIM and ``p(\textbf{x})`` the prior distribution.
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `rho`: Parameterized density matrix.
- `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `eps`: Machine epsilon.
"""
function BCFIM(x::AbstractVector, p, rho, drho; M=missing, eps=GLOBAL_EPS)
    para_num = length(x)
    if para_num == 1
        #### singleparameter scenario ####
        p_num = length(p)
        if ismissing(M)
            M = SIC(size(rho[1])[1])
        end
        F_tp = zeros(p_num)
        for i in 1:p_num
            F_tp[i] = CFIM(rho[i], drho[i][1], M; eps=eps)
        end
        F = 0.0
        arr = [p[i]*F_tp[i] for i in 1:p_num]
        F = trapz(x[1], arr)
    else
        #### multiparameter scenario #### 
        if ismissing(M)
            M = SIC(size(vec(rho)[1])[1])
        end

        xnum = length(x)
        trapzm(x, integrands, slice_dim) = [trapz(tuple(x...), I) for I in [reshape(hcat(integrands...)[i,:], length.(x)...) for i in 1:slice_dim]] 
        Fs = [p*CFIM(rho,drho,M;eps=eps)|>vec for (p,rho,drho) in zip(p,rho,drho)]
        F = trapzm(x, Fs, xnum^2) |> I->reshape(I,xnum,xnum)
    end    
end

########## Bayesian version of QFIM ##########
@doc raw"""

    BQFIM(x::AbstractVector, p, rho, drho; LDtype=:SLD, eps=GLOBAL_EPS)

Calculation of the Bayesian quantum Fisher information (BQFI) and the Bayesian quantum Fisher information matrix (BQFIM) of the form
``\mathcal{F}_{\mathrm{Bayes}}=\int p(\textbf{x})\mathcal{F}\mathrm{d}\textbf{x}`` with ``\mathcal{F}`` the QFIM of all types and ``p(\textbf{x})`` the prior distribution.
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `rho`: Parameterized density matrix.
- `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
- `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are "SLD" (default), "RLD" and "LLD".
- `eps`: Machine epsilon.
"""
function BQFIM(x::AbstractVector, p, rho, drho; LDtype=:SLD, eps=GLOBAL_EPS) 
    para_num = length(x)
    if para_num == 1
        #### singleparameter scenario ####
        p_num = length(p)
        F_tp = zeros(p_num)
        for i in 1:p_num
            F_tp[i] = QFIM(rho[i], drho[i][1]; LDtype=LDtype, eps=eps)
        end
        F = 0.0
        arr = [p[i]*F_tp[i] for i in 1:p_num]
        F = trapz(x[1], arr)
    else
        #### multiparameter scenario #### 
        xnum = length(x)
        trapzm(x, integrands, slice_dim) =  [trapz(tuple(x...), I) for I in [reshape(hcat(integrands...)[i,:], length.(x)...) for i in 1:slice_dim]] 
        Fs = [p*QFIM(rho,drho;LDtype=LDtype,eps=eps)|>vec for (p,rho,drho) in zip(p,rho,drho)]
        F = trapzm(x, Fs, xnum^2) |> I->reshape(I,xnum,xnum)
    end    
end

########## Bayesian quantum Cramer-Rao bound ##########
@doc raw"""

    BQCRB(x::AbstractVector, p, dp, rho, drho; b=missing, db=missing, LDtype=:SLD, btype=1, eps=GLOBAL_EPS)

Calculation of the Bayesian quantum Cramer-Rao bound (BQCRB).
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `dp`: Derivatives of the prior distribution with respect to the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
- `rho`: Parameterized density matrix.
- `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
- `b`: Vector of biases of the form ``\textbf{b}=(b(x_0),b(x_1),\dots)^{\mathrm{T}}``.
- `db`: Derivatives of b on the unknown parameters to be estimated, It should be expressed as ``\textbf{b}'=(\partial_0 b(x_0),\partial_1 b(x_1),\dots)^{\mathrm{T}}``.
- `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are "SLD" (default), "RLD" and "LLD".
- `btype`: Types of the BCRB. Options are 1, 2 and 3.
- `eps`: Machine epsilon.
"""
function BQCRB(x::AbstractVector, p, dp, rho, drho; b=missing, db=missing, LDtype=:SLD, btype=1, eps=GLOBAL_EPS)
    para_num = length(x)

    if ismissing(b)
        b =  [zero(x) for x in x]
        db =  [zero(x) for x in x]
    end
    if ismissing(db)
        db = [zero(x) for x in x]
    end
    
    if para_num == 1    
        
        #### singleparameter scenario ####
        p_num = length(p)

        if typeof(drho[1]) == Vector{Matrix{ComplexF64}}
            drho = [drho[i][1] for i in 1:p_num]
        end
        if typeof(b[1]) == Vector{Float64} || typeof(b[1]) == Vector{Int64}
            b = b[1]
        end
        if typeof(db[1]) == Vector{Float64} || typeof(db[1]) == Vector{Int64}
            db = db[1]
        end
        F_tp = zeros(p_num)
        for i in 1:p_num
            f = QFIM(rho[i], drho[i]; LDtype=LDtype, eps=eps)
            F_tp[i] = f
        end
        F = 0.0
        if btype == 1
            arr = [p[i]*((1+db[i])^2/F_tp[i]+b[i]^2) for i in 1:p_num]
            F = trapz(x[1], arr)
        elseif btype == 2
            arr = [p[i]*F_tp[i] for i in 1:p_num]
            F1 = trapz(x[1], arr)
            arr2 = [p[j]*(1+db[j]) for j in 1:p_num]
            B = trapz(x[1], arr2)
            arr3 = [p[k]*b[k]^2 for k in 1:p_num]
            bb = trapz(x[1], arr3)
            F = B^2/F1+bb
        elseif btype == 3
            I_tp = [real(dp[i]*dp[i]/p[i]^2) for i in 1:p_num]
            arr = [p[j]*(dp[j]*b[j]/p[j]+(1 + db[j]))^2 / (I_tp[j] + F_tp[j]) for j in 1:p_num]
            F = trapz(x[1], arr)
        else
            println("NameError: btype should be choosen in {1, 2, 3}.")
        end  
        return F
    else
        #### multiparameter scenario ####
        xnum = length(x)
        bs  =  Iterators.product(b...)
        dbs =  Iterators.product(db...)
        trapzm(x, integrands, slice_dim) = [trapz(tuple(x...), I) for I in [reshape(hcat(integrands...)[i,:], length.(x)...) for i in 1:slice_dim]] 

        if btype == 1 
            integrand1(p,rho,drho,b,db)=p*diagm(1 .+db)*pinv(QFIM(rho,drho;LDtype=LDtype,eps=eps))*diagm(1 .+db)+b*b'
            integrands = [integrand1(p,rho,drho,[b...],[db...])|>vec for (p,rho,drho,b,db) in zip(p,rho,drho,bs,dbs)]
            I = trapzm(x, integrands, xnum^2) |> I->reshape(I,xnum,xnum)
        elseif btype == 2
            Bs = [p*(1 .+[db...]) for (p,db) in zip(p,dbs)]
            B = trapzm(x, Bs, xnum)|> diagm
            Fs = [p*QFIM(rho,drho;LDtype=LDtype,eps=eps)|>vec for (p,rho,drho) in zip(p,rho,drho)]
            F = trapzm(x, Fs, xnum^2) |> I->reshape(I,xnum,xnum)
            bbts = [p*[b...]*[b...]'|>vec for (p,b) in zip(p,bs)]
            I = B*pinv(F)*B + (trapzm(x, bbts, xnum^2) |> I->reshape(I,xnum,xnum))
        elseif btype == 3
            Ip(p,dp) = dp*dp'/p^2
            G = [G_mat(p,dp,[b...],[db...]) for (p,dp,b,db) in zip(p,dp,bs,dbs)]
            integrand3(p,dp,rho,drho,G_tp)=p*G_tp*pinv(Ip(p,dp)+QFIM(rho,drho;LDtype=LDtype,eps=eps))*G_tp'
            integrands = [integrand3(p,dp,rho,drho,G_tp)|>vec for (p,dp,rho,drho,G_tp) in zip(p,dp,rho,drho,G)]
            I = trapzm(x, integrands, xnum^2) |> I->reshape(I,xnum,xnum)
        else
            println("NameError: btype should be choosen in {1, 2, 3}.")
        end  
        return I
    end
end

@doc raw"""

    BCRB(x::AbstractVector, p, dp, rho, drho; M=missing, b=missing, db=missing, btype=1, eps=GLOBAL_EPS)

Calculation of the Bayesian Cramer-Rao bound (BCRB).
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `dp`: Derivatives of the prior distribution with respect to the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
- `rho`: Parameterized density matrix.
- `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `b`: Vector of biases of the form ``\textbf{b}=(b(x_0),b(x_1),\dots)^{\mathrm{T}}``.
- `db`: Derivatives of b on the unknown parameters to be estimated, It should be expressed as ``\textbf{b}'=(\partial_0 b(x_0),\partial_1 b(x_1),\dots)^{\mathrm{T}}``.
- `btype`: Types of the BCRB. Options are 1, 2 and 3.
- `eps`: Machine epsilon.
"""
function BCRB(x::AbstractVector, p, dp, rho, drho; M=missing, b=missing, db=missing, btype=1, eps=GLOBAL_EPS)
    para_num = length(x)
     
    if ismissing(b)
        b =  [zero(x) for x in x]
        db =  [zero(x) for x in x]
    end
    if ismissing(db)
        db = [zero(x) for x in x]
    end

    if para_num == 1
        #### singleparameter scenario ####
        p_num = length(p)
        if ismissing(M)
            M = SIC(size(rho[1])[1])
        end
        if typeof(drho[1]) == Vector{Matrix{ComplexF64}}
            drho = [drho[i][1] for i in 1:p_num]
        end
        if typeof(b[1]) == Vector{Float64} || typeof(b[1]) == Vector{Int64}
            b = b[1]
        end
        if typeof(db[1]) == Vector{Float64} || typeof(db[1]) == Vector{Int64}
            db = db[1]
        end
        F_tp = zeros(p_num)

        for i in 1:p_num
            f = CFIM(rho[i], drho[i], M; eps=eps)
            F_tp[i] = f
        end
        F = 0.0
        if btype == 1
            arr = [p[i]*((1+db[i])^2/F_tp[i]+b[i]^2) for i in 1:p_num]
            F = trapz(x[1], arr)
        elseif btype == 2
            arr = [p[i]*F_tp[i] for i in 1:p_num]
            F1 = trapz(x[1], arr)
            arr2 = [p[j]*(1+db[j]) for j in 1:p_num]
            B = trapz(x[1], arr2)
            arr3 = [p[k]*b[k]^2 for k in 1:p_num]
            bb = trapz(x[1], arr3)
            F = B^2/F1+bb
        elseif btype == 3
            I_tp = [real(dp[i]*dp[i]/p[i]^2) for i in 1:p_num]
            arr = [p[j]*(dp[j]*b[j]/p[j]+(1 + db[j]))^2 / (I_tp[j] + F_tp[j]) for j in 1:p_num]
            F = trapz(x[1], arr)
        else
            println("NameError: btype should be choosen in {1, 2, 3}")
        end
        return F
    else
        #### multiparameter scenario #### 
        if ismissing(M)
            M = SIC(size(vec(rho)[1])[1])
        end

        xnum = length(x)
        bs  =  Iterators.product(b...)
        dbs =  Iterators.product(db...)
        trapzm(x, integrands, slice_dim) =  [trapz(tuple(x...), I) for I in [reshape(hcat(integrands...)[i,:], length.(x)...) for i in 1:slice_dim]] 

        if btype == 1 
            integrand1(p,rho,drho,b,db)=p*diagm(1 .+db)*pinv(CFIM(rho,drho,M;eps=eps))*diagm(1 .+db)+b*b'
            integrands = [integrand1(p,rho,drho,[b...],[db...])|>vec for (p,rho,drho,b,db) in zip(p,rho,drho,bs,dbs)]
            I = trapzm(x, integrands, xnum^2) |> I->reshape(I,xnum,xnum)
        elseif btype == 2
            Bs = [p*(1 .+[db...]) for (p,db) in zip(p,dbs)]
            B = trapzm(x, Bs, xnum)|> diagm
            Fs = [p*CFIM(rho,drho,M;eps=eps)|>vec for (p,rho,drho) in zip(p,rho,drho)]
            F = trapzm(x, Fs, xnum^2) |> I->reshape(I,xnum,xnum)
            bbts = [p*[b...]*[b...]'|>vec for (p,b) in zip(p,bs)]
            I = B*pinv(F)*B + (trapzm(x, bbts, xnum^2) |> I->reshape(I,xnum,xnum))
        elseif btype == 3
            Ip(p,dp) = dp*dp'/p^2
            G = [G_mat(p,dp,[b...],[db...]) for (p,dp,b,db) in zip(p,dp,bs,dbs)]
            integrand3(p,dp,rho,drho,G_tp)=p*G_tp*pinv(Ip(p,dp)+CFIM(rho,drho,M;eps=eps))*G_tp'
            integrands = [integrand3(p,dp,rho,drho,G_tp)|>vec for (p,dp,rho,drho,G_tp) in zip(p,dp,rho,drho,G)]
            I = trapzm(x, integrands, xnum^2) |> I->reshape(I,xnum,xnum)
        else
            println("NameError: btype should be choosen in {1, 2, 3}.")
        end    
        return I
    end
end

function G_mat(p, dp, b, db)
    para_num = length(db)
    G_tp = zeros(para_num, para_num)
    for i in 1:para_num
        for j in 1:para_num
            if i == j
                G_tp[i,j] = dp[j]*b[i]/p + (1+db[i])
            else
                G_tp[i,j] = dp[j]*b[i]/p
            end
        end
    end
    return G_tp
end

"""

    QVTB(x::AbstractVector, p, dp, rho, drho; LDtype=:SLD, eps=GLOBAL_EPS)

Calculation of the Bayesian version of Cramer-Rao bound in troduced by Van Trees (VTB).
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `dp`: Derivatives of the prior distribution with respect to the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
- `rho`: Parameterized density matrix.
- `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
- `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are "SLD" (default), "RLD" and "LLD".
- `eps`: Machine epsilon.
"""
function QVTB(x::AbstractVector, p, dp, rho, drho; LDtype=:SLD, eps=GLOBAL_EPS)
    para_num = length(x)
    if para_num == 1
        #### singleparameter scenario ####
        p_num = length(p)
        if typeof(drho[1]) == Vector{Matrix{ComplexF64}}
            drho = [drho[i][1] for i in 1:p_num]
        end
        if typeof(dp[1]) == Vector{Float64}
            dp = [dp[i][1] for i in 1:p_num]
        end
        F_tp = zeros(p_num)
        for m in 1:p_num
            F_tp[m] = QFIM(rho[m], drho[m]; LDtype=LDtype, eps=eps)
        end

        arr1 = [real(dp[i]*dp[i]/p[i]) for i in 1:p_num]  
        I = trapz(x[1], arr1)
        arr2 = [real(F_tp[j]*p[j]) for j in 1:p_num]
        F = trapz(x[1], arr2)
        I = 1.0/(I+F)
        return I
    else
        #### multiparameter scenario ####
        xnum = length(x)
        trapzm(x, integrands, slice_dim) =  [trapz(tuple(x...), I) for I in [reshape(hcat(integrands...)[i,:], length.(x)...) for i in 1:slice_dim]] 
        Ip(p,dp) = dp*dp'/p^2

        Iprs = [p*Ip(p,dp)|>vec for (p,dp) in zip(p,dp)]
        Ipr = trapzm(x, Iprs, xnum^2)|> I->reshape(I,xnum,xnum)
        Fs = [p*QFIM(rho,drho;eps=eps)|>vec for (p,rho,drho) in zip(p,rho,drho)]
        F = trapzm(x, Fs, xnum^2) |> I->reshape(I,xnum,xnum)
        I = pinv(Ipr+F)
        return I
    end
end

"""

    VTB(x::AbstractVector, p, dp, rho, drho; M=missing, eps=GLOBAL_EPS)

Calculation of the Bayesian version of Cramer-Rao bound introduced by Van Trees (VTB).
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `dp`: Derivatives of the prior distribution with respect to the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
- `rho`: Parameterized density matrix.
- `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `eps`: Machine epsilon.
"""
function VTB(x::AbstractVector, p, dp, rho, drho; M=missing, eps=GLOBAL_EPS)
    para_num = length(x)
    if para_num == 1
        #### singleparameter scenario ####
        p_num = length(p)
        if ismissing(M)
            M = SIC(size(rho[1])[1])
        end
        if typeof(drho[1]) == Vector{Matrix{ComplexF64}}
            drho = [drho[i][1] for i in 1:p_num]
        end
        if typeof(dp[1]) == Vector{Float64}
            dp = [dp[i][1] for i in 1:p_num]
        end
        F_tp = zeros(p_num)
        for m in 1:p_num
            F_tp[m] = CFIM(rho[m], drho[m], M; eps=eps)
        end

        arr1 = [real(dp[i]*dp[i]/p[i]) for i in 1:p_num]  
        I = trapz(x[1], arr1)
        arr2 = [real(F_tp[j]*p[j]) for j in 1:p_num]
        F = trapz(x[1], arr2)
        res = 1.0/(I+F)
        return res
    else
        #### multiparameter scenario #### 
        if ismissing(M)
            M = SIC(size(vec(rho)[1])[1])
        end
        xnum = length(x)
        trapzm(x, integrands, slice_dim) =  [trapz(tuple(x...), I) for I in [reshape(hcat(integrands...)[i,:], length.(x)...) for i in 1:slice_dim]] 
        Ip(p,dp) = dp*dp'/p^2

        Iprs = [p*Ip(p,dp)|>vec for (p,dp) in zip(p,dp)]
        Ipr = trapzm(x, Iprs, xnum^2)|> I->reshape(I,xnum,xnum)
        Fs = [p*CFIM(rho,drho,M;eps=eps)|>vec for (p,rho,drho) in zip(p,rho,drho)]
        F = trapzm(x, Fs, xnum^2) |> I->reshape(I,xnum,xnum)
        I = pinv(Ipr+F) 
        return I
    end
end

########## optimal biased bound ##########
function OBB_func(du, u, para, t)
    F, J, x = para
    J_tp = interp1(x, J, t)
    F_tp = interp1(x, F, t)
    bias = u[1]
    dbias = u[2]
    du[1] = dbias
    du[2] = -J_tp*dbias + F_tp*bias - J_tp
end

function boundary_condition(residual, u, p, t)
    residual[1] = u[1][2] + 1.0
    residual[2] = u[end][2] + 1.0
end

function interp1(xspan, yspan, x)
    idx = (x .>= xspan[1]) .& (x .<= xspan[end])
    intf = interpolate((xspan,), yspan, Gridded(Linear()))
    y = intf[x[idx]]
    return y
end

"""

    OBB(x::AbstractVector, p, dp, rho, drho, d2rho; LDtype=:SLD, eps=GLOBAL_EPS)

Calculation of the Bayesian version of Cramer-Rao bound introduced by Van Trees (VTB).
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `dp`: Derivatives of the prior distribution with respect to the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
- `rho`: Parameterized density matrix.
- `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
- `d2rho`: Second order Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
- `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are "SLD" (default), "RLD" and "LLD".
- `eps`: Machine epsilon.
"""
function OBB(x::AbstractVector, p, dp, rho, drho, d2rho; LDtype=:SLD, eps=GLOBAL_EPS)
    p_num = length(p)
    
    if typeof(drho[1]) == Vector{Matrix{ComplexF64}}
        drho = [drho[i][1] for i in 1:p_num]
    end
    if typeof(d2rho[1]) == Vector{Matrix{ComplexF64}}
        d2rho = [d2rho[i][1] for i in 1:p_num]
    end
    if typeof(dp[1]) == Vector{Float64}
        dp = [dp[i][1] for i in 1:p_num]
    end
    if typeof(x[1]) != Float64 || typeof(x[1]) != Int64
        x = x[1]
    end

    delta = x[2] - x[1] 
    F, J = zeros(p_num), zeros(p_num)
    for m in 1:p_num
        f, LD = QFIM(rho[m], drho[m], LDtype=LDtype, exportLD=true)
        dF = real(tr(2*d2rho[m]*d2rho[m]*LD - LD*LD*drho[m]))
        J[m] = dp[m]/p[m] - dF/f 
        F[m] = f
    end

    prob = BVProblem(OBB_func, boundary_condition, [0.0, 0.0], (x[1], x[end]), (F, J, x))
    sol = solve(prob, GeneralMIRK4(), dt=delta)

    bias = [sol.u[i][1] for i in 1:p_num] 
    dbias = [sol.u[i][2] for i in 1:p_num]

    value = [p[i]*((1+dbias[i])^2/F[i] + bias[i]^2) for i in 1:p_num]
    return trapz(x, value)
end
