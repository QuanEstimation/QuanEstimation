import numpy as np
import warnings
import math
import os
import quanestimation.ControlOpt as ctrl

class ControlSystem:
    def __init__(self, tspan, rho0, H0, Hc, dH, decay, ctrl_bound, W, ctrl0, load, eps):
        
        """
        ----------
        Inputs
        ----------
        tspan: 
           --description: time series.
           --type: array
        
        rho0: 
           --description: initial state (density matrix).
           --type: matrix
        
        H0: 
           --description: free Hamiltonian.
           --type: matrix or a list of matrix
           
        Hc: 
           --description: control Hamiltonian.
           --type: list (of matrix)
        
        dH: 
           --description: derivatives of Hamiltonian on all parameters to
                          be estimated. For example, dH[0] is the derivative
                          vector on the first parameter.
           --type: list (of matrix)
           
        decay:
           --description: decay operators and the corresponding decay rates.
                          decay[0][0] represent the first decay operator and
                          decay[0][1] represent the corresponding decay rate.
           --type: list 

        ctrl_bound:   
           --description: lower and upper bounds of the control coefficients.
                          ctrl_bound[0] represent the lower bound of the control coefficients and
                          ctrl_bound[1] represent the upper bound of the control coefficients.
           --type: list 

        W:
            --description: weight matrix.
            --type: matrix
        
        ctrl0:
            --description: initial control coefficients.
            --type: list (of vector)

        eps:
            --description: calculation eps.
            --type: float
        
        """   
        self.tspan = tspan
        self.rho0 = np.array(rho0, dtype=np.complex128)

        if type(H0) == np.ndarray:
            self.freeHamiltonian = np.array(H0, dtype=np.complex128)
        else:
            self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0] 
        
        if Hc == []:
            Hc = [np.zeros((len(self.rho0), len(self.rho0)))]
        self.control_Hamiltonian = [np.array(x, dtype=np.complex128) for x in Hc]

        if type(dH) != list:
            raise TypeError("The derivative of Hamiltonian should be a list!") 

        if dH == []:
            dH = [np.zeros((len(self.rho0), len(self.rho0)))]
        self.Hamiltonian_derivative = [np.array(x, dtype=np.complex128) for x in dH]
        
        if ctrl0 == []:
            if ctrl_bound == []:
                ctrl0 = [2*np.random.random(len(self.tspan)-1)-np.ones(len(self.tspan)-1) for i in range(len(self.control_Hamiltonian))]
            else:
                a = ctrl_bound[0]
                b = ctrl_bound[1]
                ctrl0 = [(b-a)*np.random.random(len(self.tspan)-1)+a*np.ones(len(self.tspan)-1) for i in range(len(self.control_Hamiltonian))]
            self.control_coefficients = ctrl0
        elif len(ctrl0) >= 1:
            self.control_coefficients = [ctrl0[0][i] for i in range(len(self.control_Hamiltonian))]
        
        if decay == []:
            decay_opt = [np.zeros((len(self.rho0), len(self.rho0)))]
            self.gamma = [0.0]
        else:
            decay_opt = [decay[i][0] for i in range(len(decay))]
            self.gamma = [decay[i][1] for i in range(len(decay))]
        self.decay_opt = [np.array(x, dtype=np.complex128) for x in decay_opt]

        ctrl_bound = [float(ctrl_bound[0]), float(ctrl_bound[1])]
        if ctrl_bound == []:
            ctrl_bound = [-np.inf, np.inf]
        self.ctrl_bound = ctrl_bound
        
        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        self.eps = eps
        if load == True:
            if os.path.exists("controls.csv"):
                data = np.genfromtxt("controls.csv")[-len(self.control_Hamiltonian):]
                self.control_coefficients = [data[i] for i in range(len(data))]
            
        ctrl_num = len(self.control_coefficients)
        Hc_num = len(self.control_Hamiltonian)
        if Hc_num < ctrl_num:
            raise TypeError("There are %d control Hamiltonians but %d coefficients sequences: \
                                too many coefficients sequences"%(Hc_num,ctrl_num))
        elif Hc_num > ctrl_num:
            warnings.warn("Not enough coefficients sequences: there are %d control Hamiltonians \
                            but %d coefficients sequences. The rest of the control sequences are\
                            set to be 0."%(Hc_num,ctrl_num), DeprecationWarning)
            for i in range(Hc_num-ctrl_num):
                self.control_coefficients = np.concatenate((self.control_coefficients, np.zeros(len(self.control_coefficients[0]))))
        else: pass
        
        number = math.ceil((len(self.tspan)-1)/len(self.control_coefficients[0]))
        if len(self.tspan)-1 % len(self.control_coefficients[0]) != 0:
            tnum = number*len(self.control_coefficients[0])
            self.tspan = np.linspace(self.tspan[0], self.tspan[-1], tnum+1)
        else: pass

def ControlOpt(*args, method = "auto-GRAPE", **kwargs):

    if method == "auto-GRAPE":
        return ctrl.GRAPE_Copt(*args, **kwargs, auto=True)
    elif method == "GRAPE":
        return ctrl.GRAPE_Copt(*args, **kwargs, auto=False)
    elif method == "PSO":
        return ctrl.PSO_Copt(*args, **kwargs)
    elif method == "DE":
        return ctrl.DE_Copt(*args, **kwargs)
    elif method == "DDPG":
        return ctrl.DDPG_Copt(*args, **kwargs)
    else:
        raise ValueError("{!r} is not a valid value for method, supported values are 'auto-GRAPE', 'GRAPE', 'PSO', 'DE', 'DDPG'.".format(method))

def csv2npy_controls(controls, num):
    C_save = []
    N = int(len(controls)/num)
    for ci in range(N):
        C_tp = controls[ci*num:(ci+1)*num]
        C_save.append(C_tp)
    np.save("controls", C_save)