
########## Bayesian quantum Cramer-Rao bound ##########
function BCRB(x, p, rho, drho; M=[], b=[], db=[], btype=1, eps=1e-8)
    para_num = length(x)
    p_num = length(p)

    if b==[]
        b = [zeros(p_num) for i in 1:para_num]
        db = [zeros(p_num) for i in 1:para_num]
    end
    if b!=[] && db==[] 
        db = [zeros(p_num) for i in 1:para_num]
    end

    if para_num == 1
        #### singleparameter senario ####
        if M==[]
            M = load_M(size(rho[1])[1])
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
            f = CFI(rho[i], drho[i], M; eps=eps)
            F_tp[i] = f
        end
        F = 0.0
        if btype == 1
            arr = [p[i]*((1+db)^2/F_tp[i]+b^2) for i in 1:p_num]
            F = trapz(x[1], arr)
        elseif btype == 2
            arr = [p[i]*F_tp[i] for i in 1:p_num]
            F1 = trapz(x[1], arr)
            arr2 = [p[j]*(1+db[j]) for j in 1:p_num]
            B = trapz(x[1], arr2)
            arr3 = [p[k]*b[k]^2 for k in 1:p_num]
            B1 = trapz(x[1], arr3)
            F = B^2*(1.0/F1)+B1^2
        end
        return F
    else
        #### multiparameter senario ####
        println("pass")
    end
end

function BQCRB(x, p, rho, drho; b=[], db=[], dtype="SLD", btype=1, eps=1e-8)
    para_num = length(x)
    p_num = length(p)

    if b==[]
        b = [zeros(p_num) for i in 1:para_num]
        db = [zeros(p_num) for i in 1:para_num]
    end
    if b!=[] && db==[] 
        db = [zeros(p_num) for i in 1:para_num]
    end
    if para_num == 1
        #### singleparameter senario ####
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
            f = QFIM(rho[i], drho[i]; dtype=dtype, eps=eps)
            F_tp[i] = f
        end
        F = 0.0
        if btype == 1
            arr = [p[i]*((1+db)^2/F_tp[i]+b^2) for i in 1:p_num]
            F = trapz(x[1], arr)
        elseif btype == 2
            arr = [p[i]*F_tp[i] for i in 1:p_num]
            F1 = trapz(x[1], arr)
            arr2 = [p[j]*(1+db[j]) for j in 1:p_num]
            B = trapz(x[1], arr2)
            arr3 = [p[k]*b[k]^2 for k in 1:p_num]
            B1 = trapz(x[1], arr3)
            F = B^2*(1.0/F1)+B1^2
        end
        return F
    else
        #### multiparameter senario ####
        println("pass")
    end
end

function VTB(x, p, dp, rho, drho; M=[], btype=1, eps=1e-8)
    para_num = length(x)
    p_num = length(p)
    if para_num == 1
        #### singleparameter senario ####
        if M==[]
            M = load_M(size(rho[1])[1])
        end
        if typeof(drho[1]) == Vector{Matrix{ComplexF64}}
            drho = [drho[i][1] for i in 1:p_num]
        end
        if typeof(dp[1]) == Vector{Float64}
            dp = [dp[i][1] for i in 1:p_num]
        end
        F_tp = zeros(p_num)
        for m in 1:p_num
            F_tp[m] = CFI(rho[m], drho[m], M; eps=eps)
        end
        res = 0.0
        if btype==1
            I_tp = [real(dp[i]*dp[i]/p[i]^2) for i in 1:p_num]
            arr = [p[j]/(I_tp[j]+F_tp[j]) for j in 1:p_num]
            res = trapz(x[1], arr)
        elseif btype==2
            arr1 = [real(dp[i]*dp[i]/p[i]) for i in 1:p_num]  
            I = trapz(x[1], arr1)
            arr2 = [real(F_tp[j]*p[j]) for j in 1:p_num]
            F = trapz(x[1], arr2)
            res = 1.0/(I+F)
        end
        return res
    else
        #### multiparameter senario ####
        println("pass")
    end
end

function QVTB(x, p, dp, rho, drho; dtype="SLD", btype=1, eps=1e-8)
    para_num = length(x)
    p_num = length(p)
    if para_num == 1
        #### singleparameter senario ####
        if typeof(drho[1]) == Vector{Matrix{ComplexF64}}
            drho = [drho[i][1] for i in 1:p_num]
        end
        if typeof(dp[1]) == Vector{Float64}
            dp = [dp[i][1] for i in 1:p_num]
        end
        F_tp = zeros(p_num)
        for m in 1:p_num
            F_tp[m] = QFIM(rho[m], drho[m]; dtype=dtype, eps=eps)
        end
        res = 0.0
        if btype==1
            I_tp = [real(dp[i]*dp[i]/p[i]^2) for i in 1:p_num]
            arr = [p[j]/(I_tp[j]+F_tp[j]) for j in 1:p_num]
            res = trapz(x[1], arr)
        elseif btype==2
            I, F = 0.0, 0.0
            arr1 = [real(dp[i]*dp[i]/p[i]) for i in 1:p_num]  
            I = trapz(x[1], arr1)
            arr2 = [real(F_tp[j]*p[j]) for j in 1:p_num]
            F = trapz(x[1], arr2)
            res = 1.0/(I+F)
        end
        return res
    else
        #### multiparameter senario ####
        println("pass")
    end
end

function TWCB(x, p, dp, rho, drho; eps=1e-8)
    para_num = length(dp)
    p_num = length(p)
    CFIM_res = zeros(para_num,para_num)
    QFIM_res = zeros(para_num,para_num)
    
    deltax = 1.0/p_num
    LD = [SLD(rho[i], drho[i], eps) for i in 1:p_num]
    for para_i in 1:para_num
        for para_j in para_i:para_num
            arr1 = [real(dp[para_i][i]*dp[para_j][i]/p[i]) for i in 1:p_num]
            S1 = trapz(x, arr1)
            CFIM_res[para_i,para_j] = S1
            CFIM_res[para_j,para_i] = S1
            
            F_tp = zeros(p_num)
            for j in 1:p_num
                SLD_ac = LD[j][para_i]*LD[j][para_j]+LD[j][para_j]*LD[j][para_i]
                F_tp[j] = real(0.5*tr(rho[j]*SLD_ac))
            end

            arr2 = [real(F_tp[i]*p[i]) for i in 1:p_num]
            S2 = trapz(x, arr2)
            QFIM_res[para_i,para_j] = S2
            QFIM_res[para_j,para_i] = S2
        end
    end
            
    F_total = CFIM_res + QFIM_res
    return pinv(F_total)
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

function OBB(x, p, dp, rho, drho, d2rho; dtype="SLD", eps=1e-8)
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
        f, LD = QFIM(rho[m], drho[m]; dtype=dtype, exportLD=true, eps=eps)
        dF = real(tr(2*d2rho[m]*d2rho[m]*LD - LD*LD*drho[m]))
        J[m] = dp[m]/p[m] - dF/f 
        F[m] = f
    end

    prob = BVProblem(OBB_func, boundary_condition, [0.0, 0.0], (x1, x2), (F, J, x))
    sol = solve(prob, GeneralMIRK4(), dt=delta)

    bias = [sol.u[i][1] for i in 1:p_num] 
    dbias = [sol.u[i][2] for i in 1:p_num]

    value = [p[i]*((1+dbias[i])^2/F[i] + bias[i]^2) for i in 1:p_num]
    return trapz(x, value)
end
