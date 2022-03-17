import numpy as np
import warnings
import math
import os
import quanestimation.ComprehensiveOpt as compopt
from quanestimation.Common.common import gramschmidt


class ComprehensiveSystem:
    def __init__(self, psi0, ctrl0, measurement0, save_file, seed, eps):

        """
        ----------
        Inputs
        ----------

        psi0:
           --description: initial guesses of states (kets).
           --type: array
           
        ctrl0:
            --description: initial control coefficients.
            --type: list (of vector)
            
        measurement0:
           --description: a set of POVMs.
           --type: list (of vector)
           
        save_file:
            --description: True: save the states (or controls, measurements) and the value of the 
                                 target function for each episode.
                           False: save the states (or controls, measurements) and all the value 
                                   of the target function for the last episode.
            --type: bool 

        eps:
            --description: calculation eps.
            --type: float

        """
        self.save_file = save_file
        self.ctrl0 = ctrl0
        self.psi0 = psi0
        self.psi = psi0
        self.eps = eps
        self.seed = seed
        self.measurement0 = measurement0

    def dynamics(self, tspan, H0, dH, Hc=[], decay=[], ctrl_bound=[]):
        """
        ----------
        Inputs
        ----------
        tspan:
           --description: time series.
           --type: array

        psi0:
           --description: initial state.
           --type: vector

        measurement0:
           --description: a set of POVMs.
           --type: list (of vector)

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
        """
        
        self.tspan = tspan

        if type(H0) == np.ndarray:
            self.freeHamiltonian = np.array(H0, dtype=np.complex128)
            self.dim = len(self.freeHamiltonian)
        else:
            self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0]
            self.dim = len(self.freeHamiltonian[0])

        if self.psi0 == []:
            np.random.seed(self.seed)
            for i in range(self.dim):
                r_ini = 2 * np.random.random(self.dim) - np.ones(self.dim)
                r = r_ini / np.linalg.norm(r_ini)
                phi = 2 * np.pi * np.random.random(self.dim)
                psi0 = [r[i] * np.exp(1.0j * phi[i]) for i in range(self.dim)]
            self.psi0 = np.array(psi0)
        else:
            self.psi0 = np.array(self.psi0[0], dtype=np.complex128)

        if Hc == []:
            Hc = [np.zeros((len(self.dim), len(self.dim)))]
        self.control_Hamiltonian = [np.array(x, dtype=np.complex128) for x in Hc]

        if type(dH) != list:
            raise TypeError("The derivative of Hamiltonian should be a list!")

        if dH == []:
            dH = [np.zeros((len(self.dim), len(self.dim)))]
        self.Hamiltonian_derivative = [np.array(x, dtype=np.complex128) for x in dH]
                
        if len(dH) == 1:
            self.para_type = "single_para"
        else:
            self.para_type = "multi_para"

        if decay == []:
            decay_opt = [np.zeros((len(self.dim), len(self.dim)))]
            self.gamma = [0.0]
        else:
            decay_opt = [decay[i][0] for i in range(len(decay))]
            self.gamma = [decay[i][1] for i in range(len(decay))]
        self.decay_opt = [np.array(x, dtype=np.complex128) for x in decay_opt]

        ctrl_bound = [float(ctrl_bound[0]), float(ctrl_bound[1])]
        if ctrl_bound == []:
            ctrl_bound = [-np.inf, np.inf]
        self.ctrl_bound = ctrl_bound

        if self.ctrl0 == []:
            if ctrl_bound == []:
                ctrl0 = [
                    2 * np.random.random(len(self.tspan) - 1)
                    - np.ones(len(self.tspan) - 1)
                    for i in range(len(self.control_Hamiltonian))
                ]
            else:
                a = ctrl_bound[0]
                b = ctrl_bound[1]
                ctrl0 = [
                    (b - a) * np.random.random(len(self.tspan) - 1)
                    + a * np.ones(len(self.tspan) - 1)
                    for i in range(len(self.control_Hamiltonian))
                ]
            self.control_coefficients = ctrl0
            self.ctrl0 = [ctrl0]

        elif len(self.ctrl0) >= 1:
            self.control_coefficients = [
                self.ctrl0[0][i] for i in range(len(self.control_Hamiltonian))
            ]

        ctrl_num = len(self.control_coefficients)
        Hc_num = len(self.control_Hamiltonian)
        if Hc_num < ctrl_num:
            raise TypeError(
                "There are %d control Hamiltonians but %d coefficients sequences: \
                                too many coefficients sequences"
                % (Hc_num, ctrl_num)
            )
        elif Hc_num > ctrl_num:
            warnings.warn(
                "Not enough coefficients sequences: there are %d control Hamiltonians \
                            but %d coefficients sequences. The rest of the control sequences are\
                            set to be 0."
                % (Hc_num, ctrl_num),
                DeprecationWarning,
            )
            for i in range(Hc_num - ctrl_num):
                self.control_coefficients = np.concatenate(
                    (
                        self.control_coefficients,
                        np.zeros(len(self.control_coefficients[0])),
                    )
                )
        else:
            pass

        if self.measurement0 == []:
            np.random.seed(self.seed)
            M = [[] for i in range(self.dim)]
            for i in range(self.dim):
                r_ini = 2 * np.random.random(self.dim) - np.ones(self.dim)
                r = r_ini / np.linalg.norm(r_ini)
                phi = 2 * np.pi * np.random.random(self.dim)
                M[i] = [r[i] * np.exp(1.0j * phi[i]) for i in range(self.dim)]
            self.M = gramschmidt(np.array(M))
        elif len(self.measurement0) >= 1:
            self.M = [self.measurement0[0][i] for i in range(self.dim)]
            self.M = [np.array(x, dtype=np.complex128) for x in self.M]

        number = math.ceil((len(self.tspan) - 1) / len(self.control_coefficients[0]))
        if len(self.tspan) - 1 % len(self.control_coefficients[0]) != 0:
            tnum = number * len(self.control_coefficients[0])
            self.tspan = np.linspace(self.tspan[0], self.tspan[-1], tnum + 1)
        else:
            pass
         
        self.dynamic = Main.QuanEstimation.Lindblad(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.psi0,
            self.tspan,
            self.decay_opt,
            self.gamma,
        )   
        
        self.dynamics_type = "dynamics"

    def kraus(self, K, dK):
        k_num = len(K)
        para_num = len(dK[0])
        dK_tp = [[np.array(dK[i][j], dtype=np.complex128) for i in range(k_num)] for j in range(para_num)]
        self.K = [np.array(x, dtype=np.complex128) for x in K]
        self.dK = dK_tp
        
        if para_num == 1:
            self.para_type = "single_para"
        else:
            self.para_type = "multi_para"

        self.dim = len(K[0])
        if self.psi0 == []:
            np.random.seed(self.seed)
            for i in range(self.dim):
                r_ini = 2 * np.random.random(self.dim) - np.ones(self.dim)
                r = r_ini / np.linalg.norm(r_ini)
                phi = 2 * np.pi * np.random.random(self.dim)
                psi0 = [r[i] * np.exp(1.0j * phi[i]) for i in range(self.dim)]
            self.psi0 = np.array(psi0)
        else:
            self.psi0 = np.array(self.psi0[0], dtype=np.complex128)

        if self.psi == []:
            self.psi = [self.psi0]

        if self.measurement0 == []:
            np.random.seed(self.seed)
            M = [[] for i in range(self.dim)]
            for i in range(self.dim):
                r_ini = 2 * np.random.random(self.dim) - np.ones(self.dim)
                r = r_ini / np.linalg.norm(r_ini)
                phi = 2 * np.pi * np.random.random(self.dim)
                M[i] = [r[i] * np.exp(1.0j * phi[i]) for i in range(self.dim)]
            self.M = gramschmidt(np.array(M))
        elif len(self.measurement0) >= 1:
            self.M = [self.measurement0[0][i] for i in range(self.dim)]
            self.M = [np.array(x, dtype=np.complex128) for x in self.M]
            
        self.dynamic = Main.QuanEstimation.Kraus(self.K,self.dK,self.rho0)

        self.dynamics_type = "kraus"
    
    
    def SC(self, W=[], M=[], target="QFIM", dtype="SLD"):
        """
        Description: use DE algorithm to optimize states and control coefficients.

        ---------
        Inputs
        ---------
        M:
            --description: a set of POVM.
            --type: list of matrix
            
        W:
            --description: weight matrix.
            --type: matrix

        """
        if self.dynamics_type != "dynamics":
            raise ValueError("{!r} is not a valid type for dynamics, supported type is \
                             Lindblad dynamics.".format(self.dynamics_type))

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W
        
        if M != []:
            M = [np.array(x, dtype=np.complex128) for x in M]
            self.obj = Main.QuanEstimation.CFIM_Obj(M, self.W, self.eps, self.para_type) 
        else:
            if target=="HCRB":
                if self.para_type == "single_para":
                    warnings.warn("In single parameter scenario, HCRB is equivalent to QFI. \
                           Please choose QFIM as the target function for control optimization",\
                           DeprecationWarning)
                else:
                    pass #### to be done
            elif target=="QFIM" and (dtype=="SLD" or dtype=="RLD" or dtype=="LLD"):
                self.obj = Main.QuanEstimation.QFIM_Obj(self.W, self.eps, self.para_type, dtype)
            else:
                raise ValueError("Please enter the correct values for target and dtype.\
                                  Supported target are 'QFIM', 'CFIM' and 'HCRB',  \
                                  supported dtype are 'SLD', 'RLD' and 'LLD'.")
        
        self.opt = Main.QuanEstimation.StateControlOpt(self.psi0, self.control_coefficients)
        self.output = Main.QuanEstimation.Output(self.opt, self.save_file) 

        system = Main.QuanEstimation.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        Main.QuanEstimation.run(system) 
        
    def CM(self, rho0, W=[]):
        """
        Description: use DE algorithm to optimize control coefficients and the measurements.

        ---------
        Inputs
        ---------
        rho0:
            --description: initial state.
            --type: density matrix
            
        W:
            --description: weight matrix.
            --type: matrix

        """
        if self.dynamics_type != "dynamics":
            raise ValueError("{!r} is not a valid type for dynamics, supported type is \
                             Lindblad dynamics.".format(self.dynamics_type))
            
        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W
        
        self.rho0 = np.array(rho0, dtype=np.complex128)
        
        self.obj = Main.QuanEstimation.CFIM_Obj(self.M, self.W, self.eps, self.para_type)
        self.opt = Main.QuanEstimation.ControlMeasurementOpt(self.M, self.control_coefficients)
        self.output = Main.QuanEstimation.Output(self.opt, self.save_file) 

        system = Main.QuanEstimation.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        Main.QuanEstimation.run(system) 

        self.load_save_meas()

    def SM(self, W=[]):
        """
        Description: use DE algorithm to optimize states and the measurements.

        ---------
        Inputs
        ---------
            
        W:
            --description: weight matrix.
            --type: matrix

        """
        if self.dynamics_type == "dynamics":
            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W

            if len(self.control_coefficients[0]) == 1:
                H0 = np.array(self.freeHamiltonian, dtype=np.complex128)
                Hc = [np.array(x, dtype=np.complex128) for x in self.control_Hamiltonian]
                Htot = H0 + sum(
                    [Hc[i] * self.control_coefficients[i][0]
                    for i in range(len(self.control_coefficients))])
                freeHamiltonian = np.array(Htot, dtype=np.complex128)
            else:
                H0 = np.array(self.freeHamiltonian, dtype=np.complex128)
                Hc = [np.array(x, dtype=np.complex128) for x in self.control_Hamiltonian]
                Htot = []
                for i in range(len(self.control_coefficients[0])):
                    S_ctrl = sum(
                        [Hc[j] * self.control_coefficients[j][i]
                        for j in range(len(self.control_coefficients))])
                    Htot.append(H0 + S_ctrl)
                freeHamiltonian = [np.array(x, dtype=np.complex128) for x in Htot]

            self.dynamic = Main.QuanEstimation.Lindblad(
                freeHamiltonian,
                self.Hamiltonian_derivative,
                self.psi0,
                self.tspan,
                self.decay_opt,
                self.gamma,
                )   
        
        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W

        self.obj = Main.QuanEstimation.CFIM_Obj(self.M, self.W, self.eps, self.para_type)
        self.opt = Main.QuanEstimation.StateMeasurementOpt(self.psi0, self.M)
        self.output = Main.QuanEstimation.Output(self.opt, self.save_file) 

        system = Main.QuanEstimation.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        Main.QuanEstimation.run(system)    
    
    def SCM(self, W=[]):
        """
        Description: use DE algorithm to optimize states, control coefficients and the measurements.

        ---------
        Inputs
        ---------
            
        W:
            --description: weight matrix.
            --type: matrix

        """
        if self.dynamics_type != "dynamics":
            raise ValueError("{!r} is not a valid type for dynamics, supported type is \
                             Lindblad dynamics.".format(self.dynamics_type))
        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W
        
        self.obj = Main.QuanEstimation.CFIM_Obj(self.M, self.W, self.eps, self.para_type)
        self.opt = Main.QuanEstimation.StateControlMeasurementOpt(self.control_coefficients, self.psi0, self.M)
        self.output = Main.QuanEstimation.Output(self.opt, self.save_file) 

        system = Main.QuanEstimation.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        Main.QuanEstimation.run(system)    
        
        self.load_save_state()
        self.load_save_meas()


    def load_save_state(self):
        if os.path.exists("states.csv"):
            file_load = open("states.csv", "r")
            file_load = "".join([i for i in file_load]).replace("im", "j")
            file_load = "".join([i for i in file_load]).replace(" ", "")
            file_save = open("states.csv", "w")
            file_save.writelines(file_load)
            file_save.close()
        else:
            pass

    def load_save_meas(self):
        if os.path.exists("measurements.csv"):
            file_load = open("measurements.csv", "r")
            file_load = "".join([i for i in file_load]).replace("im", "j")
            file_load = "".join([i for i in file_load]).replace(" ", "")
            file_save = open("measurements.csv", "w")
            file_save.writelines(file_load)
            file_save.close()
        else:
            pass

def ComprehensiveOpt(save_file=False, method="AD", **kwargs):

    if method == "AD":
        return compopt.AD_Compopt(save_file=save_file, **kwargs)
    elif method == "PSO":
        return compopt.PSO_Compopt(save_file=save_file, **kwargs)
    elif method == "DE":
        return compopt.DE_Compopt(save_file=save_file, **kwargs)
    else:
        raise ValueError("{!r} is not a valid value for method, supported values are 'AD', \
                         'PSO', 'DE'.".format(method))
