import numpy as np
from scipy.integrate import simpson
from quanestimation.base.Common.Common import extract_ele
from quanestimation.base.Common.Common import SIC
from itertools import product
import math


def Bayes(x, p, rho, y, M=None, estimator="mean", savefile=False):
    """
    Bayesian estimation. The prior distribution is updated via the posterior 
    distribution obtained by Bayes' rule. The estimated value of parameters are 
    updated via either the expectation value of the distribution or maximum a 
    posteriori probability (MAP).

    Args:
        x (list): The regimes of the parameters for the integral.
        p (np.ndarray): The prior distribution as a multidimensional array.
        rho (list): Parameterized density matrix as a multidimensional list.
        y (np.ndarray): The experimental results obtained in practice.
        M (list, optional): A set of positive operator-valued measure (POVM). 
            Defaults to a set of rank-one symmetric informationally complete 
            POVM (SIC-POVM).
        estimator (str, optional): Estimators for the bayesian estimation. 
            Options are:
                "mean" (default) - The expectation value of the distribution.
                "MAP" - Maximum a posteriori probability.
        savefile (bool, optional): Whether to save all posterior distributions. 
            If True, generates "pout.npy" and "xout.npy" containing all 
            posterior distributions and estimated values across iterations. 
            If False, only saves the final posterior distribution and all 
            estimated values. Defaults to False.

    Returns:
        (tuple): 
            pout (np.ndarray): 
                The posterior distribution in the final iteration.

            xout (float/list): 
                The estimated values in the final iteration.

    Raises:
        TypeError: 
            If `M` is not a list.
        ValueError: 
            If estimator is not "mean" or "MAP".

    Note: 
        SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
        which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/solutions.html).
    """
    if M is None:
        M = []
        
    para_num = len(x)
    max_episode = len(y)
    
    # Single parameter scenario
    if para_num == 1:
        if not M:
            M = SIC(len(rho[0]))
        elif not isinstance(M, list):
            raise TypeError("M must be a list")
            
        x_out = []
        p_out = []
        
        for mi in range(max_episode):
            res_exp = int(y[mi])
            pyx = np.zeros(len(x[0]))
            
            # Calculate conditional probabilities
            for xi in range(len(x[0])):
                p_tp = np.real(np.trace(rho[xi] @ M[res_exp]))
                pyx[xi] = p_tp
                
            # Update posterior distribution
            arr = [pyx[m] * p[m] for m in range(len(x[0]))]
            py = simpson(arr, x[0])
            p_update = pyx * p / py
            p = p_update / simpson(p_update, x[0])
            
            # Handle estimator type
            if estimator == "mean":
                mean = simpson([p[m] * x[0][m] for m in range(len(x[0]))], x[0])
                x_out.append(mean)
            elif estimator == "MAP":
                indx = np.argmax(p)
                x_out.append(x[0][indx])
            else:
                raise ValueError(
                    "Invalid estimator: {}. Supported values are 'mean' and 'MAP'"
                    .format(estimator)
                )
                
            # Save intermediate results if requested
            if savefile:
                p_out.append(p)
                
        # Save final results
        if savefile:
            np.save("pout", p_out)
            np.save("xout", x_out)
            return p, x_out[-1]
        else:
            np.save("pout", p)
            np.save("xout", x_out)
            return p, x_out[-1]
            
    # Multiparameter scenario
    else:
        p_shape = np.shape(p)
        p_ext = extract_ele(p, para_num)
        rho_ext = extract_ele(rho, para_num)

        p_list, rho_list = [], []
        for p_ele, rho_ele in zip(p_ext, rho_ext):
            p_list.append(p_ele)
            rho_list.append(rho_ele)

        dim = len(rho_list[0])
        if not M:
            M = SIC(dim)
        elif not isinstance(M, list):
            raise TypeError("M must be a list")
            
        x_out = []
        p_out = []  # Always initialize as list
        
        for mi in range(max_episode):
            res_exp = int(y[mi])
            pyx_list = np.zeros(len(p_list))
            
            # Calculate conditional probabilities
            for xi in range(len(p_list)):
                p_tp = np.real(np.trace(rho_list[xi] @ M[res_exp]))
                pyx_list[xi] = p_tp
                
            # Reshape and update posterior distribution
            pyx = pyx_list.reshape(p_shape)
            arr = p * pyx
            for si in reversed(range(para_num)):
                arr = simpson(arr, x[si])
            py = arr
            p_update = p * pyx / py
            arr_norm = p_update.copy()
            for si in reversed(range(para_num)):
                arr_norm = simpson(arr_norm, x[si])
            p = p_update / arr_norm
            
            # Handle estimator type
            if estimator == "mean":
                mean = integ(x, p)
                x_out.append(mean)
            elif estimator == "MAP":
                # Use unravel_index for better performance and type safety
                max_idx = np.argmax(p)
                indices = np.unravel_index(max_idx, p_shape)
                x_out.append([x[i][indices[i]] for i in range(para_num)])
            else:
                raise ValueError(
                    f"Invalid estimator: {estimator}. Supported values are 'mean' and 'MAP'"
                )
                
            # Save intermediate results if requested
            if savefile:
                p_out.append(p)
                
        # Save final results
        if savefile:
            np.save("pout", p_out)
            np.save("xout", x_out)
            return p, x_out[-1]
        else:
            np.save("pout", p)
            np.save("xout", x_out)
            return p, x_out[-1]


def MLE(x, rho, y, M=None, savefile=False):
    """
    Maximum likelihood estimation (MLE) for parameter estimation.

    Args:
        x (list): 
            The regimes of the parameters for the integral.
        rho (list): 
            Parameterized density matrix as a multidimensional list.
        y (np.ndarray): 
            The experimental results obtained in practice.
        M (list, optional): 
            A set of positive operator-valued measure (POVM). Defaults to a set of rank-one 
            symmetric informationally complete POVM (SIC-POVM).
        savefile (bool, optional): 
            Whether to save all likelihood functions. If True, generates "Lout.npy" and 
            "xout.npy" containing all likelihood functions and estimated values across 
            iterations. If False, only saves the final likelihood function and all 
            estimated values. Defaults to False.

    Returns:
        (tuple): 
            Lout (np.ndarray): 
                The likelihood function in the final iteration.

            xout (float/list): 
                The estimated values in the final iteration.

    Raises:
        TypeError: If `M` is not a list.

    Note: 
        SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
        which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/solutions.html).
    """
    if M is None:
        M = []

    para_num = len(x)
    max_episode = len(y)
    if para_num == 1:
        #### single parameter scenario ####
        if M == []:
            M = SIC(len(rho[0]))
        else:
            if not isinstance(M, list):
                raise TypeError("Please make sure M is a list!")

        if savefile == False:
            x_out = []
            L_out = np.ones(len(x[0]))
            for mi in range(max_episode):
                res_exp = int(y[mi])
                for xi in range(len(x[0])):
                    p_tp = np.real(np.trace(rho[xi] @ M[res_exp]))
                    L_out[xi] += math.log(p_tp + 1e-16)
                indx = np.where(L_out == max(L_out))[0][0]
                x_out.append(x[0][indx])
            np.save("Lout", L_out)
            np.save("xout", x_out)

            return L_out, x_out[-1]
        else:
            L_out, x_out = [], []
            L_tp = np.ones(len(x[0]))
            for mi in range(max_episode):
                res_exp = int(y[mi])
                for xi in range(len(x[0])):
                    p_tp = np.real(np.trace(rho[xi] @ M[res_exp]))
                    L_tp[xi] += math.log(p_tp + 1e-16)
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
            if not isinstance(M, list):
                raise TypeError("Please make sure M is a list!")

        if savefile == False:
            x_out = []
            L_list = np.ones(len(rho_list))
            for mi in range(max_episode):
                res_exp = int(y[mi])
                for xi in range(len(rho_list)):
                    p_tp = np.real(np.trace(rho_list[xi] @ M[res_exp]))
                    L_list[xi] += math.log(p_tp + 1e-16)
                L_out = L_list.reshape(p_shape)
                indx = np.where(L_out == np.max(L_out))
                x_out.append([x[i][indx[i][0]] for i in range(para_num)])
            np.save("Lout", L_out)
            np.save("xout", x_out)

            return L_out, x_out[-1]
        else:
            L_out, x_out = [], []
            L_list = np.ones(len(rho_list))
            for mi in range(max_episode):
                res_exp = int(y[mi])
                for xi in range(len(rho_list)):
                    p_tp = np.real(np.trace(rho_list[xi] @ M[res_exp]))
                    L_list[xi] += math.log(p_tp + 1e-16)
                L_tp = L_list.reshape(p_shape)
                indx = np.where(L_tp == np.max(L_tp))
                L_out.append(L_tp)
                x_out.append([x[i][indx[i][0]] for i in range(para_num)])

            np.save("Lout", L_out)
            np.save("xout", x_out)
            return L_tp, x_out[-1]

def integ(x, p):
    """
    Compute the mean for each parameter by integrating over the other parameters.
    
    Args:
        x: List of arrays, the parameter ranges
        p: Multidimensional array, the probability distribution
        
    Returns:
        List of mean values for each parameter
    """
    para_num = len(x)
    mean = []
    
    for i in range(para_num):
        # Create a list of axes to integrate over (all except current axis i)
        axes = [j for j in range(para_num) if j != i]
        
        # Integrate over all other axes to get marginal distribution for axis i
        p_marginal = p
        for axis in sorted(axes, reverse=True):
            p_marginal = simpson(p_marginal, x[axis], axis=axis)
            
        # Compute mean for current parameter
        mean_val = simpson(x[i] * p_marginal, x[i])
        mean.append(mean_val)
        
    return mean

def BayesCost(x, p, xest, rho, M, W=None, eps=1e-8):
    """
    Calculation of the average Bayesian cost with a quadratic cost function.

    Args:
        x (list): 
            The regimes of the parameters for the integral.
        p (array): 
            The prior distribution as a multidimensional array.
        xest (float or list): 
            The estimators. For single-parameter estimation, a scalar constant
            estimator is used for every POVM outcome. For multi-parameter
            estimation, xest is a nested sequence with shape
            ``(len(M), len(x))``: xest[mi] is the full parameter vector for
            POVM outcome mi. The default SIC POVM has dim**2 outcomes.
        rho (list): 
            Parameterized density matrix as a multidimensional list.
        M (list): 
            A set of positive operator-valued measure (POVM).
        W (array, optional): 
            Weight matrix. Defaults to an identity matrix.
        eps (float, optional): 
            Machine epsilon.

    Returns:
        (float): 
            The average Bayesian cost.

    Raises:
        TypeError: 
            If `M` is not a list.
        ValueError:
            If multiparameter `xest` does not have shape `(len(M), len(x))`.
    """
    if M is None:
        M = []

    if W is None:
        W = []

    para_num = len(x)
    if para_num == 1:
        # single-parameter scenario
        if M == []:
            M = SIC(len(rho[0]))
        else:
            if not isinstance(M, list):
                raise TypeError("Please make sure M is a list!")
            
        # if isinstance(x, np.ndarray):
        #     x = [x]    

        p_num = len(x[0])

        value = [p[i]*sum([np.trace(rho[i] @ M[mi])*(x[0][i]-xest)**2 for mi in range(len(M))]) for i in range(p_num)]
        C = simpson(value, x[0])
        return np.real(C)
    else:
        # multi-parameter scenario
        p_shape = np.shape(p)
        p_ext = extract_ele(p, para_num)
        rho_ext = extract_ele(rho, para_num)

        p_list, rho_list = [], []
        for p_ele, rho_ele in zip(p_ext, rho_ext):
            p_list.append(p_ele)
            rho_list.append(rho_ele)

        x_pro = product(*x)
        x_list = []
        for x_ele in x_pro:
            x_list.append([x_ele[i] for i in range(para_num)])
            
        dim = len(rho_list[0])
        p_num = len(p_list)
        
        if W == []:
            W = np.identity(para_num)
            
        if M == []:
            M = SIC(dim)
        else:
            if not isinstance(M, list):
                raise TypeError("Please make sure M is a list!")

        expected_xest_shape = (len(M), para_num)
        try:
            xest_shape = np.shape(xest)
        except ValueError as exc:
            raise ValueError(
                "BayesCost: multiparameter xest must be a rectangular sequence "
                f"with shape (len(M), para_num); expected {expected_xest_shape}."
            ) from exc
        if xest_shape != expected_xest_shape:
            raise ValueError(
                "BayesCost: multiparameter xest must have shape "
                f"(len(M), para_num); got {xest_shape}, "
                f"expected {expected_xest_shape}."
            )

        value = [0.0 for i in range(p_num)]
        for i in range(p_num):
            x_tp = np.array(x_list[i])
            xCx = 0.0
            for mi in range(len(M)):
                xCx += np.trace(rho_list[i] @ M[mi])*np.dot((x_tp-xest[mi]).reshape(1, -1), W @ (x_tp-xest[mi]).reshape(-1, 1))[0][0]
            value[i] = p_list[i]*xCx
        C = np.array(value).reshape(p_shape)
        for si in reversed(range(para_num)):
            C = simpson(C, x[si])
        return np.real(C)
    
    
def BCB(x, p, rho, W=None, eps=1e-8):
    """
    Calculation of the Bayesian cost bound with a quadratic cost function.

    Args:
        x (list): 
            The regimes of the parameters for the integral.
        p (array): 
            The prior distribution as a multidimensional array.
        rho (list): 
            Parameterized density matrix as a multidimensional list.
        W (array, optional): 
            Weight matrix. Defaults to an identity matrix.
        eps (float, optional): 
            Machine epsilon. Defaults to 1e-8.

    Returns:
        (float): 
            The value of the minimum Bayesian cost.

    Note:
        This function calculates the Bayesian cost bound for parameter estimation.
    """
    if W is None:
        W = []
    para_num = len(x)
    if para_num == 1:
        # single-parameter scenario
        dim = len(rho[0])
        p_num = len(x[0])
        value = [p[i]*x[0][i]**2 for i in range(p_num)]
        delta2_x = simpson(value, x[0])
        rho_avg = np.zeros((dim, dim), dtype=np.complex128)
        rho_pri = np.zeros((dim, dim), dtype=np.complex128)
        for di in range(dim):
            for dj in range(dim):
                rho_avg_arr = [p[m]*rho[m][di][dj] for m in range(p_num)]
                rho_pri_arr = [p[n]*x[0][n]*rho[n][di][dj] for n in range(p_num)]
                rho_avg[di][dj] = simpson(rho_avg_arr, x[0])
                rho_pri[di][dj] = simpson(rho_pri_arr, x[0])
        Lambda = Lambda_avg(rho_avg, [rho_pri], eps=eps)
        minBC = delta2_x - np.real(np.trace(rho_avg @ Lambda[0] @ Lambda[0]))
        return minBC
    else:
        # multi-parameter scenario
        p_shape = np.shape(p)
        p_ext = extract_ele(p, para_num)
        rho_ext = extract_ele(rho, para_num)

        p_list, rho_list = [], []
        for p_ele, rho_ele in zip(p_ext, rho_ext):
            p_list.append(p_ele)
            rho_list.append(rho_ele)

        dim = len(rho_list[0])
        p_num = len(p_list)

        x_pro = product(*x)
        x_list = []
        for x_ele in x_pro:
            x_list.append([x_ele[i] for i in range(para_num)])
        
        if W == []:
            W = np.identity(para_num)
        
        value = [0.0 for i in range(p_num)]
        for i in range(p_num):
            x_tp = np.array(x_list[i])
            xCx = np.dot(x_tp.reshape(1, -1), np.dot(W, x_tp.reshape(-1, 1)))[0][0]
            value[i] = p_list[i]*xCx
        delta2_x = np.array(value).reshape(p_shape)
        for si in reversed(range(para_num)):
            delta2_x = simpson(delta2_x, x[si])
        rho_avg = np.zeros((dim, dim), dtype=np.complex128)
        rho_pri = [np.zeros((dim, dim), dtype=np.complex128) for i in range(para_num)]
        for di in range(dim):
            for dj in range(dim):
                rho_avg_arr = [p_list[m]*rho_list[m][di][dj] for m in range(p_num)]
                rho_avg_tp = np.array(rho_avg_arr).reshape(p_shape)
                for si in reversed(range(para_num)):
                    rho_avg_tp = simpson(rho_avg_tp, x[si])
                rho_avg[di][dj] = rho_avg_tp

                for para_i in range(para_num):
                    rho_pri_arr = [p_list[n]*x_list[n][para_i]*rho_list[n][di][dj] for n in range(p_num)]
                    rho_pri_tp = np.array(rho_pri_arr).reshape(p_shape)
                    for si in reversed(range(para_num)):
                        rho_pri_tp = simpson(rho_pri_tp, x[si])

                    rho_pri[para_i][di][dj] = rho_pri_tp
        Lambda = Lambda_avg(rho_avg, rho_pri, eps=eps)
        Mat = np.zeros((para_num, para_num), dtype=np.complex128)
        for para_m in range(para_num):
            for para_n in range(para_num):
                Mat += W[para_m][para_n] * np.dot(Lambda[para_m], Lambda[para_n])
                
        minBC = delta2_x-np.real(np.trace(rho_avg @ Mat))
        return minBC
        
def Lambda_avg(rho_avg, rho_pri, eps=1e-8):
    r"""Compute the average cost operators $\Lambda_a$ for the minimum
    Bayesian cost bound (BCB).

    Solves the Lyapunov-type equation

    $$
    \frac{1}{2}\bigl(\bar{\rho}\Lambda_a + \Lambda_a\bar{\rho}\bigr) =
    \bar{\rho}_{\mathrm{pri}}^{(a)},
    $$

    for each parameter $a$, where $\bar{\rho} = \int p(x)\rho(x)\,\mathrm{d}x$
    is the prior-averaged density matrix and
    $\bar{\rho}_{\mathrm{pri}}^{(a)} = \int p(x)\,x_a\,\rho(x)\,\mathrm{d}x$
    is the prior-weighted average. This is analogous to the SLD equation.

    Args:
        rho_avg (numpy.ndarray): Prior-averaged density matrix $\bar{\rho}$.
        rho_pri (list): List of prior-weighted density matrices
            $\bar{\rho}_{\mathrm{pri}}^{(a)}$, one per parameter.
        eps (float, optional): Eigenvalue truncation threshold.

    Returns:
        list: List of $\Lambda_a$ operators, one per parameter.

    See Also:
        :func:`BCB`: Bayesian cost bound using these operators.
    """
    para_num = len(rho_pri)
    dim = len(rho_avg)
    Lambda = [[] for i in range(0, para_num)]
    val, vec = np.linalg.eig(rho_avg)
    val = np.real(val)
    for para_i in range(0, para_num):
        Lambda_eig = np.array([[0.0 + 0.0 * 1.0j for i in range(0, dim)] for i in range(0, dim)])
        for fi in range(0, dim):
            for fj in range(0, dim):
                if np.abs(val[fi] + val[fj]) > eps:
                    Lambda_eig[fi][fj] = 2* (vec[:, fi].conj().transpose() @ rho_pri[para_i]@ vec[:, fj]) / (val[fi] + val[fj])
        Lambda_eig[Lambda_eig == np.inf] = 0.0
        Lambda[para_i] =  vec @ Lambda_eig @ vec.conj().transpose()
    return Lambda
