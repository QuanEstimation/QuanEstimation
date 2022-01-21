
########## Bayesian quantum Cramer-Rao bound ##########
function BQCRB(ρ, dρ, p, dp, x1, x2)
    xrange = range(x1, stop=x2, length=length(p))
    para_num = length(dρ)
    CFIM_res = zeros(para_num,para_num)
    QFIM_res = zeros(para_num,para_num)
    LD = SLD(ρ, dρ)
    deltax = 1.0/length(p)
    for para_i in 1:para_num
        for para_j in para_i:para_num
            arr1 = [real(dp[para_i][i]*dp[para_j][i]/p[i]) for i in 1:length(p)]
            
            S1 = trapz(xrange, arr1)
            CFIM_res[para_i,para_j] = S1
            CFIM_res[para_j,para_i] = S1
            
            F_tp = real(tr(LD[para_i]*LD[para_j]*rho))
            arr2 = real(F_tp*p)
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
    bias = u[1]
    dbias = u[2]
    du[1] = dbias
    du[2] = -J_tp*dbias + F*bias - J_tp
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

function OBB(rho, drho, d2rho, x1, x2, p, dp; accuracy=1e-8)
    xrange = Vector(range(x1, stop=x2, length=length(p)))
    delta = xrange[2] - xrange[1]
    F = QFI(rho, drho[1], accuracy)
    L = SLD(rho, drho[1], accuracy=accuracy)
    dF = real(2tr(2*d2rho[1]*d2rho[1]*L - L*L*drho[1]))  ###Is it a real number?
    J = [dp[1][i]/p[i] - dF/F for i in 1:length(p)]

    prob = BVProblem(OBB_func, boundary_condition, [0.0, 0.0], (x1, x2), (F, J, xrange))
    sol = solve(prob, GeneralMIRK4(), dt=delta)
    # println(sol.retcode)
    bias = [sol.u[i][1] for i in 1:length(p)]
    dbias = [sol.u[i][2] for i in 1:length(p)]
    # println(sol.u)
    value = [p[i]*((1+dbias[i])^2/F + bias[i]^2) for i in 1:(length(p))]
    xrange = range(x1, stop=x2, length=length(p))
    return trapz(xrange, value)
end
