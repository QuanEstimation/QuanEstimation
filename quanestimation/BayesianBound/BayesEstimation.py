import numpy as np
from scipy.integrate import simps
from quanestimation.Common.common import extract_ele
from quanestimation.Common.common import SIC


def Bayes(x, p, rho, y, M=[], savefile=False):
    """
    Bayesian estimation. The prior distribution is updated via the posterior  
    distribution obtained by the Bayes’ rule and the estimated value of parameters
    obtained via the maximum a posteriori probability (MAP).

    Parameters
    ----------
    > **x:** `list`
        -- The regimes of the parameters for the integral.

    > **p:** `multidimensional array`
        -- The prior distribution.

    > **rho:** `multidimensional list`
        -- Parameterized density matrix.

    > **y:** `array`
        -- The experimental results obtained in practice.

    > **M:** `list of matrices`
        -- A set of positive operator-valued measure (POVM). The default measurement 
        is a set of rank-one symmetric informationally complete POVM (SIC-POVM).

    > **savefile:** `bool`
        -- Whether or not to save all the posterior distributions.  
        If set `True` then two files "pout.npy" and "xout.npy" will be generated including
        the posterior distributions and the estimated values in the iterations. If set 
        `False` the posterior distribution in the final iteration and the estimated values
        in all iterations will be saved in "pout.npy" and "xout.npy". 

    Returns
    ----------
    **pout and xout:** `array and float`
        -- The posterior distribution and the estimated values in the final iteration.

    **Note:** 
        SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
        which can be downloaded from the [website](http://www.physics.umb.edu/Research/QBism/
        solutions.html).
    """

    para_num = len(x)
    max_episode = len(y)
    if para_num == 1:
        #### single parameter scenario ####
        if M == []:
            M = SIC(len(rho[0]))
        else:
            if type(M) != list:
                raise TypeError("Please make sure M is a list!")
        if savefile == False:
            for mi in range(max_episode):
                res_exp = int(y[mi])
                pyx = np.zeros(len(x[0]))
                for xi in range(len(x[0])):
                    p_tp = np.real(np.trace(np.dot(rho[xi], M[res_exp])))
                    pyx[xi] = p_tp
                arr = [pyx[m] * p[m] for m in range(len(x[0]))]
                py = simps(arr, x[0])
                p_update = pyx * p / py
                p = p_update
            indx = np.where(p == max(p))[0][0]
            x_out = x[0][indx]
            return p, x_out
        else:
            p_out, x_out = [], []
            for mi in range(max_episode):
                res_exp = int(y[mi])
                pyx = np.zeros(len(x[0]))
                for xi in range(len(x[0])):
                    p_tp = np.real(np.trace(np.dot(rho[xi], M[res_exp])))
                    pyx[xi] = p_tp
                arr = [pyx[m] * p[m] for m in range(len(x[0]))]
                py = simps(arr, x[0])
                p_update = pyx * p / py
                p = p_update
                indx = np.where(p == max(p))[0][0]
                p_out.append(p)
                x_out.append(x[0][indx])
            np.save("pout", p_out)
            np.save("xout", x_out)
            return p, x_out[-1]
    else:
        #### multiparameter scenario ####
        p_shape = np.shape(p)
        p_ext = extract_ele(p, para_num)
        rho_ext = extract_ele(rho, para_num)

        p_list, rho_list = [], []
        for p_ele, rho_ele in zip(p_ext, rho_ext):
            p_list.append(p_ele)
            rho_list.append(rho_ele)

        dim = len(rho_list[0])
        if M == []:
            M = SIC(dim)
        else:
            if type(M) != list:
                raise TypeError("Please make sure M is a list!")

        if savefile == False:
            for mi in range(max_episode):
                res_exp = int(y[mi])
                pyx_list = np.zeros(len(p_list))
                for xi in range(len(p_list)):
                    p_tp = np.real(np.trace(np.dot(rho_list[xi], M[res_exp])))
                    pyx_list[xi] = p_tp
                pyx = pyx_list.reshape(p_shape)
                arr = p * pyx
                for si in reversed(range(para_num)):
                    arr = simps(arr, x[si])
                py = arr
                p_update = p * pyx / py
                p = p_update

            indx = np.where(np.array(p) == np.max(np.array(p)))
            x_out = [x[i][indx[i][0]] for i in range(para_num)]
            return p, x_out
        else:
            p_out, x_out = [], []
            for mi in range(max_episode):
                res_exp = int(y[mi])
                pyx_list = np.zeros(len(p_list))
                for xi in range(len(p_list)):
                    p_tp = np.real(np.trace(np.dot(rho_list[xi], M[res_exp])))
                    pyx_list[xi] = p_tp
                pyx = pyx_list.reshape(p_shape)
                arr = p * pyx
                for si in reversed(range(para_num)):
                    arr = simps(arr, x[si])
                py = arr
                p_update = p * pyx / py
                p = p_update

                indx = np.where(np.array(p) == np.max(np.array(p)))
                p_out.append(p)
                x_out.append([x[i][indx[i][0]] for i in range(para_num)])
            np.save("pout", p_out)
            np.save("xout", x_out)
            return p, x_out[-1]


def MLE(x, rho, y, M=[], savefile=False):
    """
    Bayesian estimation. The estimated value of parameters obtained via the 
    maximum likelihood estimation (MLE).

    Parameters
    ----------
    > **x:** `list`
        -- The regimes of the parameters for the integral.

    > **rho:** `multidimensional list`
        -- Parameterized density matrix.

    > **y:** `array`
        -- The experimental results obtained in practice.

    > **M:** `list of matrices`
        -- A set of positive operator-valued measure (POVM). The default measurement 
        is a set of rank-one symmetric informationally complete POVM (SIC-POVM).

    > **savefile:** `bool`
        -- Whether or not to save all the likelihood functions.  
        If set `True` then two files "Lout.npy" and "xout.npy" will be generated including
        the likelihood functions and the estimated values in the iterations. If set 
        `False` the likelihood function in the final iteration and the estimated values
        in all iterations will be saved in "Lout.npy" and "xout.npy". 

    Returns
    ----------
    **Lout and xout:** `array and float`
        -- The likelihood function and the estimated values in the final iteration.

    **Note:** 
        SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
        which can be downloaded from the [website](http://www.physics.umb.edu/Research/QBism/
        solutions.html).
    """

    para_num = len(x)
    max_episode = len(y)
    if para_num == 1:
        #### single parameter scenario ####
        if M == []:
            M = SIC(len(rho[0]))
        else:
            if type(M) != list:
                raise TypeError("Please make sure M is a list!")

        if savefile == False:
            L_out = np.ones(len(x[0]))
            for mi in range(max_episode):
                res_exp = int(y[mi])
                for xi in range(len(x[0])):
                    p_tp = np.real(np.trace(np.dot(rho[xi], M[res_exp])))
                    L_out[xi] = L_out[xi] * p_tp
            indx = np.where(L_out == max(L_out))[0][0]
            x_out = x[0][indx]
            return L_out, x_out
        else:
            L_out, x_out = [], []
            L_tp = np.ones(len(x[0]))
            for mi in range(max_episode):
                res_exp = int(y[mi])
                for xi in range(len(x[0])):
                    p_tp = np.real(np.trace(np.dot(rho[xi], M[res_exp])))
                    L_tp[xi] = L_tp[xi] * p_tp
                indx = np.where(L_tp == max(L_tp))[0][0]
                L_out.append(L_tp)
                x_out.append(x[0][indx])

            np.save("Lout", L_out)
            np.save("xout", x_out)
            return L_tp, x_out[-1]
    else:
        #### multiparameter scenario ####
        p_shape = []
        for i in range(para_num):
            p_shape.append(len(x[i]))
        rho_ext = extract_ele(rho, para_num)

        rho_list = []
        for rho_ele in rho_ext:
            rho_list.append(rho_ele)

        dim = len(rho_list[0])
        if M == []:
            M = SIC(dim)
        else:
            if type(M) != list:
                raise TypeError("Please make sure M is a list!")

        if savefile == False:
            L_list = np.ones(len(rho_list))
            for mi in range(max_episode):
                res_exp = int(y[mi])
                for xi in range(len(rho_list)):
                    p_tp = np.real(np.trace(np.dot(rho_list[xi], M[res_exp])))
                    L_list[xi] = L_list[xi] * p_tp
            L_out = L_list.reshape(p_shape)
            indx = np.where(L_out == np.max(L_out))
            x_out = [x[i][indx[i][0]] for i in range(para_num)]
            return L_out, x_out
        else:
            L_out, x_out = [], []
            L_list = np.ones(len(rho_list))
            for mi in range(max_episode):
                res_exp = int(y[mi])
                for xi in range(len(rho_list)):
                    p_tp = np.real(np.trace(np.dot(rho_list[xi], M[res_exp])))
                    L_list[xi] = L_list[xi] * p_tp
                L_tp = L_list.reshape(p_shape)
                indx = np.where(L_tp == np.max(L_tp))
                L_out.append(L_tp)
                x_out.append([x[i][indx[i][0]] for i in range(para_num)])

            np.save("Lout", L_out)
            np.save("xout", x_out)
            return L_tp, x_out[-1]
