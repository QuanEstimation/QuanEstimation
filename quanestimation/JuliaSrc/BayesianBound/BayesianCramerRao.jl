
########## Bayesian quantum Cramer-Rao bound ##########
function BQCRB(x, p, rho, drho; accuracy=1e-8)
    x1, x2 = x[1], x[end]
    xrange = Vector(range(x1, stop=x2, length=length(p)))

    F = zeros(length(p))
    for i in 1:length(p)
        f = QFI(rho[i], drho[i][1], accuracy)
        F[i] = f
    end
    value = [p[i]/F[i] for i in 1:(length(p))]
    return trapz(xrange, value)
end

function TWCB(x, p, dp, ρ, dρ; accuracy=1e-8)
    x1, x2 = x[1], x[end]
    xrange = range(x1, stop=x2, length=length(p))
    para_num = length(dp)
    CFIM_res = zeros(para_num,para_num)
    QFIM_res = zeros(para_num,para_num)
    
    deltax = 1.0/length(p)
    LD = [SLD(ρ[i], dρ[i], accuracy) for i in 1:length(p)]
    for para_i in 1:para_num
        for para_j in para_i:para_num
            arr1 = [real(dp[para_i][i]*dp[para_j][i]/p[i]) for i in 1:length(p)]
            S1 = trapz(xrange, arr1)
            CFIM_res[para_i,para_j] = S1
            CFIM_res[para_j,para_i] = S1
            
            F_tp = zeros(length(p))
            for j in 1:length(p)
                SLD_ac = LD[j][para_i]*LD[j][para_j]+LD[j][para_j]*LD[j][para_i]
                F_tp[j] = real(0.5*tr(ρ[j]*SLD_ac))
            end

            arr2 = [real(F_tp[i]*p[i]) for i in 1:length(p)]
            S2 = trapz(xrange, arr2)
            QFIM_res[para_i,para_j] = S2
            QFIM_res[para_j,para_i] = S2
        end
    end
            
    F_total = CFIM_res + QFIM_res
    return pinv(F_total)
end


########## optimal biased bound ##########
function OBB_func(du, u, para, t)
    F, J, xrange = para
    J_tp = interp1(xrange, J, t)
    F_tp = interp1(xrange, F, t)
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

function OBB(x, p, dp, rho, drho, d2rho; accuracy=1e-8)
    x1, x2 = x[1], x[end]
    xrange = Vector(range(x1, stop=x2, length=length(p)))
    delta = xrange[2] - xrange[1] 
    F, J = zeros(length(p)), zeros(length(p))
    for i in 1:length(p)
        f = QFI(rho[i], drho[i][1], accuracy)
        F[i] = f
        LD = SLD(rho[i], drho[i][1], accuracy)
        dF = real(tr(2*d2rho[i][1]*d2rho[i][1]*LD - LD*LD*drho[i][1]))
        J[i] = dp[1][i]/p[i] - dF/f 
    end

    prob = BVProblem(OBB_func, boundary_condition, [0.0, 0.0], (x1, x2), (F, J, xrange))
    sol = solve(prob, GeneralMIRK4(), dt=delta)

    bias = [sol.u[i][1] for i in 1:length(p)] 
    dbias = [sol.u[i][2] for i in 1:length(p)]

    value = [p[i]*((1+dbias[i])^2/F[i] + bias[i]^2) for i in 1:(length(p))]
    return trapz(xrange, value)
end
