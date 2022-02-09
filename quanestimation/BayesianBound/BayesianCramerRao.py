
import numpy as np
from scipy import integrate, interpolate
from quanestimation.AsymptoticBound.CramerRao import QFIM, SLD

def BQCRB(rho, drho, p, x, accuracy=1e-8):
    x1, x2 = x[0], x[-1]
    xspan = np.linspace(x1, x2, len(p))

    F_tp = np.zeros(len(p))
    for m in range(len(p)):
        f = QFIM(rho[m], drho[m], accuracy=accuracy)
        F_tp[m] = f

    arr3 = [p[i]/F_tp[i] for i in range(len(p))]
    F = integrate.simps(arr3, xspan)
    return F

def TWC(rho, drho, p, dp, x):
    x1, x2 = x[0], x[-1]
    xspan = np.linspace(x1, x2, len(p))
    para_num = len(drho[0])
    
    if para_num == 1:
        I, F = 0.0, 0.0
        arr1 = [np.real(dp[0][i]*dp[0][i]/p[i]) for i in range(len(p))]  
        I = integrate.simps(arr1, xspan)
        
        F_tp = np.zeros(len(p))
        for i in range(len(p)):
            LD = SLD(rho[i], drho[i])  
            SLD_ac = np.dot(LD,LD)+np.dot(LD,LD)
            F_tp[i] = np.real(0.5*np.trace(np.dot(rho[i],SLD_ac)))

        arr2 = [F_tp[j]*p[j] for j in range(len(p))]
        F = integrate.simps(arr2, xspan)
        return 1.0/(I+F)
    
    else:
        CFIM_res = np.zeros([para_num,para_num])
        QFIM_res = np.zeros([para_num,para_num])
        LD = [SLD(rho[i], drho[i]) for i in range(len(p))]
        for para_i in range(0,para_num):
            for para_j in range(para_i,para_num):
                arr1 = [np.real(dp[para_i][i]*dp[para_j][i]/p[i]) for i in range(len(p))]
                CFIM_res[para_i][para_j] = integrate.simps(arr1, xspan)
                CFIM_res[para_j][para_i] = CFIM_res[para_i][para_j]

                F_tp = np.zeros(len(p))
                for j in range(len(p)):
                    SLD_ac = np.dot(LD[j][para_i],LD[j][para_j])+np.dot(LD[j][para_j],LD[j][para_i])
                    F_tp[j] = np.real(0.5*np.trace(np.dot(rho[j],SLD_ac)))

                arr2 = [F_tp[i]*p[i] for i in range(len(p))]
                QFIM_res[para_i][para_j] = integrate.simps(arr2, xspan)
                QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]
            
        F_total = CFIM_res + QFIM_res
        return np.linalg.pinv(F_total)

def OBB_func(x, y, t, J, F):
    interp_J = interpolate.interp1d(t, (J))
    interp_F = interpolate.interp1d(t, (F))
    J_tp, F_tp = interp_J(x), interp_F(x)
    return np.vstack((y[1],-J_tp*y[1]+F_tp*y[0]-J_tp))

def boundary_condition(ya, yb):
    return np.array([ya[1]+1.0, yb[1]+1.0])

def OBB(rho, drho, d2rho, p, dp, x, accuracy=1e-8):
    x1, x2 = x[0], x[-1]
    xspan = np.linspace(x1, x2, len(p))
    F, J = np.zeros(len(p)), np.zeros(len(p))
    bias, dbias = np.zeros(len(p)), np.zeros(len(p))
    for m in range(len(p)):
        f, LD = QFIM(rho[m], drho[m], dtype="SLD", rep="original", exportLD=True, accuracy=accuracy)
        F[m] = f
        term1 = np.dot(np.dot(d2rho[m][0], d2rho[m][0]), LD)
        term2 = np.dot(np.dot(LD, LD), drho[m][0])
        dF = np.real(np.trace(2*term1-term2))
        J[m] = dp[0][m]/p[m] - dF/f

    y_guess = np.zeros((2, xspan.size))
    fun = lambda x, y: OBB_func(x, y, xspan, J, F)
    result = integrate.solve_bvp(fun, boundary_condition, xspan, y_guess)
    res = result.sol(xspan)
    bias, dbias = res[0], res[1]
    
    value = [p[i]*((1+dbias[i])**2/F[i] + bias[i]**2) for i in range(len(p))]
    return integrate.simps(value, xspan)
