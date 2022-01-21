
import numpy as np
import scipy as sp
from quanestimation.AsymptoticBound.CramerRao import QFIM

def BQCRB(rho, drho, p, dp, x1, x2):
    if type(drho) != list:
        raise TypeError("Please make sure drho is a list!")
    if type(dp) != list:
        raise TypeError("Please make sure dp is a list!")
    xspan = np.linspace(x1, x2, len(p))
    para_num = len(drho)
    CFIM_res = np.zeros([para_num,para_num])
    QFIM_res = np.zeros([para_num,para_num])
    LD = SLD(rho, drho)
    for para_i in range(0,para_num):
        for para_j in range(para_i,para_num):
            arr1 = [np.real(dp[para_i][i]*dp[para_j][i]/p[i]) for i in range(len(p))]
            
            CFIM_res[para_i][para_j] = sp.integrate.simps(arr1, xspan)
            CFIM_res[para_j][para_i] = CFIM_res[para_i][para_j]
            
            F_tp = np.real(np.trace(np.dot(np.dot(LD[para_i], LD[para_j]), rho)))
            arr2 = [F_tp*p[i] for i in range(len(p))]
            QFIM_res[para_i][para_j] = sp.integrate.simps(arr2, xspan)
            QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]
            
    F_total = CFIM_res + QFIM_res
    return np.linalg.pinv(F_total)

def OBB_func(x, y, t, J, F):
    interp = sp.interpolate.interp1d(t, (J))
    J_tp = interp(x)
    return np.vstack((y[1],-J_tp*y[1]+F*y[0]-J_tp))

def boundary_condition(ya, yb):
    return np.array([ya[1]+1.0, yb[1]+1.0])

def OBB(rho, drho, d2rho, x1, x2, p, dp, accuracy=1e-8):
    
    F, L = QFIM(rho, drho, dtype="SLD", rep="original", exportLD=True, accuracy=1e-8)
    term1 = np.dot(np.dot(d2rho[0], d2rho[0]), L)
    term2 = np.dot(np.dot(L, L), drho[0])
    dF = np.real(np.trace(2*term1-term2)) ###Is it a real number?
    J = np.array([dp[0][i]/p[i] - dF/F for i in range(len(p))])

    xspan = np.linspace(x1, x2, len(p))
    y_guess = np.zeros((2, xspan.size))
    fun = lambda x, y: OBB_func(x, y, xspan, J, F)
    result = sp.integrate.solve_bvp(fun, boundary_condition, xspan, y_guess)
    res = result.sol(xspan)
    
    bias, dbias = res[0], res[1]
    value = [p[i]*((1+dbias[i])**2/F + bias[i]**2) for i in range(len(p))]
    return sp.integrate.simps(value, xspan)
