import numpy as np
import h5py
from scipy.interpolate import interp1d
import warnings
import math
import os
import quanestimation.ControlOpt as ctrl
from quanestimation.Common.Common import SIC
from quanestimation import QJL
from juliacall import Main as jl, convert as jlconvert
class ControlSystem:
    """
    Attributes
    ----------
    > **savefile:** `bool`
        -- Whether or not to save all the control coeffients.  
        If set `True` then the control coefficients and the values of the 
        objective function obtained in all episodes will be saved during 
        the training. If set `False` the control coefficients in the final 
        episode and the values of the objective function in all episodes 
        will be saved.

    > **ctrl0:** `list of arrays`
        -- Initial guesses of control coefficients.

    > **eps:** `float`
        -- Machine epsilon.

    > **load:** `bool`
        -- Whether or not to load control coefficients in the current location.  
        If set `True` then the program will load control coefficients from 
        "controls.csv" file in the current location and use it as the initial 
        control coefficients.
    """

    def __init__(self, savefile, ctrl0, eps, load):
        self.savefile = savefile
        self.ctrl0 = ctrl0
        self.eps = eps
        self.load = load
        
        self.QJLType_ctrl = QJL.Vector[QJL.Vector[QJL.Vector[QJL.Float64]]] 
        
    def load_save(self, cnum, max_episode):
        if os.path.exists("controls.dat"):
            fl = h5py.File("controls.dat",'r')
            dset = fl["controls"]
            if self.savefile:
                controls = np.array([[np.array(fl[fl[dset[i]][j]]) for j in range(cnum)] for i in range(max_episode)])
            else:
                controls = np.array([np.array(fl[dset[j]]) for j in range(cnum)])
            np.save("controls", controls)
        else: pass

    def dynamics(self, tspan, rho0, H0, dH, Hc, decay=[], ctrl_bound=[], dyn_method="expm"):
        r"""
        The dynamics of a density matrix is of the form 

        \begin{aligned}
            \partial_t\rho &=\mathcal{L}\rho \nonumber \\
            &=-i[H,\rho]+\sum_i \gamma_i\left(\Gamma_i\rho\Gamma^{\dagger}_i-\frac{1}{2}
            \left\{\rho,\Gamma^{\dagger}_i \Gamma_i \right\}\right),
        \end{aligned}
        
        where $\rho$ is the evolved density matrix, H is the Hamiltonian of the 
        system, $\Gamma_i$ and $\gamma_i$ are the $i\mathrm{th}$ decay 
        operator and corresponding decay rate.

        Parameters
        ----------
        > **tspan:** `array`
            -- Time length for the evolution.

        > **rho0:** `matrix`
            -- Initial state (density matrix).

        > **H0:** `matrix or list`
            -- Free Hamiltonian. It is a matrix when the free Hamiltonian is time-
            independent and a list of length equal to `tspan` when it is time-dependent.

        > **dH:** `list`
            -- Derivatives of the free Hamiltonian on the unknown parameters to be 
            estimated. For example, dH[0] is the derivative vector on the first 
            parameter.

        > **Hc:** `list`
            -- Control Hamiltonians.

        > **decay:** `list`
            -- Decay operators and the corresponding decay rates. Its input rule is 
            decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$,$\gamma_2$],...], where $\Gamma_1$ 
            $(\Gamma_2)$ represents the decay operator and $\gamma_1$ $(\gamma_2)$ is the 
            corresponding decay rate.

        > **ctrl_bound:** `array`
            -- Lower and upper bounds of the control coefficients.
            `ctrl_bound[0]` represents the lower bound of the control coefficients and
            `ctrl_bound[1]` represents the upper bound of the control coefficients.

        > **dyn_method:** `string`
            -- Setting the method for solving the Lindblad dynamics. Options are:  
            "expm" (default) -- Matrix exponential.  
            "ode" -- Solving the differential equations directly.  
        """

        self.tspan = tspan
        self.rho0 = np.array(rho0, dtype=np.complex128)

        if dyn_method == "expm":
            self.dyn_method = "Expm"
        elif dyn_method == "ode":
            self.dyn_method = "Ode"

        if type(H0) == np.ndarray:
            self.freeHamiltonian = np.array(H0, dtype=np.complex128)
        else:
            self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0[:-1]]

        if Hc == []:
            Hc = [np.zeros((len(self.rho0), len(self.rho0)))]
        self.control_Hamiltonian = [np.array(x, dtype=np.complex128) for x in Hc]

        if type(dH) != list:
            raise TypeError("The derivative of Hamiltonian should be a list!")

        if dH == []:
            dH = [np.zeros((len(self.rho0), len(self.rho0)))]
        self.Hamiltonian_derivative = [np.array(x, dtype=np.complex128) for x in dH]
        if len(dH) == 1:
            self.para_type = "single_para"
        else:
            self.para_type = "multi_para"

        if decay == []:
            decay_opt = [np.zeros((len(self.rho0), len(self.rho0)))]
            self.gamma = [0.0]
        else:
            decay_opt = [decay[i][0] for i in range(len(decay))]
            self.gamma = [decay[i][1] for i in range(len(decay))]
        self.decay_opt = [np.array(x, dtype=np.complex128) for x in decay_opt]

        if ctrl_bound == []:
            self.ctrl_bound =  [-jl.Inf, jl.Inf]
        else:
            self.ctrl_bound = [float(ctrl_bound[0]), float(ctrl_bound[1])]
        self.ctrl_bound = jl.convert(jl.Vector[jl.Float64], self.ctrl_bound)

        if self.ctrl0 == []:
            if ctrl_bound == []:
                ctrl0 = [
                    2 * np.random.random(len(self.tspan) - 1)
                    - np.ones(len(self.tspan) - 1)
                    for i in range(len(self.control_Hamiltonian))
                ]
                self.control_coefficients = ctrl0
                self.ctrl0 = [np.array(ctrl0)]
            else:
                a = ctrl_bound[0]
                b = ctrl_bound[1]
                ctrl0 = [
                    (b - a) * np.random.random(len(self.tspan) - 1)
                    + a * np.ones(len(self.tspan) - 1)
                    for i in range(len(self.control_Hamiltonian))
                ]
            self.control_coefficients = ctrl0
            self.ctrl0 = [np.array(ctrl0)]
        elif len(self.ctrl0) >= 1:
            self.control_coefficients = [
                self.ctrl0[0][i] for i in range(len(self.control_Hamiltonian))
            ]
        ## TODO
        self.ctrl0 = QJL.convert(self.QJLType_ctrl, [[c for c in ctrls ]for ctrls in self.ctrl0])

        if self.load == True:
            if os.path.exists("controls.csv"):
                data = np.genfromtxt("controls.csv")[-len(self.control_Hamiltonian) :]
                self.control_coefficients = [data[i] for i in range(len(data))]

        ctrl_num = len(self.control_coefficients)
        Hc_num = len(self.control_Hamiltonian)
        if Hc_num < ctrl_num:
            raise TypeError(
                "There are %d control Hamiltonians but %d coefficients sequences: too many coefficients sequences"
                % (Hc_num, ctrl_num)
            )
        elif Hc_num > ctrl_num:
            warnings.warn(
                "Not enough coefficients sequences: there are %d control Hamiltonians but %d coefficients sequences. The rest of the control sequences are set to be 0."
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
        else: pass

        if type(H0) != np.ndarray:
            #### linear interpolation  ####
            f = interp1d(self.tspan, H0, axis=0)
        else: pass
        number = math.ceil((len(self.tspan) - 1) / len(self.control_coefficients[0]))
        if len(self.tspan) - 1 % len(self.control_coefficients[0]) != 0:
            tnum = number * len(self.control_coefficients[0])
            self.tspan = np.linspace(self.tspan[0], self.tspan[-1], tnum + 1)
            if type(H0) != np.ndarray:
                H0_inter = f(self.tspan)
                self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0_inter[:-1]]
            else: pass
                
        else: pass

        
        self.opt = QJL.ControlOpt(
            ctrl = self.ctrl0,
            ctrl_bound=self.ctrl_bound, 
            seed=self.seed
        )
        self.dynamic = jl.QuanEstimation.Lindblad(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.rho0,
            self.tspan,
            self.decay_opt,
            self.gamma,
            dyn_method = self.dyn_method,
        )
        self.output = QJL.Output(self.opt, save=self.savefile)

        self.dynamics_type = "lindblad"

    def QFIM(self, W=[], LDtype="SLD"):
        r"""
        Choose QFI or $\mathrm{Tr}(WF^{-1})$ as the objective function. 
        In single parameter estimation the objective function is QFI and in 
        multiparameter estimation it will be $\mathrm{Tr}(WF^{-1})$.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.

        > **LDtype:** `string`
            -- Types of QFI (QFIM) can be set as the objective function. Options are:  
            "SLD" (default) -- QFI (QFIM) based on symmetric logarithmic derivative (SLD).  
            "RLD" -- QFI (QFIM) based on right logarithmic derivative (RLD).  
            "LLD" -- QFI (QFIM) based on left logarithmic derivative (LLD).  
        """

        if LDtype != "SLD" and LDtype != "RLD" and LDtype != "LLD":
            raise ValueError(
                "{!r} is not a valid value for LDtype, supported values are 'SLD', 'RLD' and 'LLD'.".format(
                    LDtype
                )
            )

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        self.obj = QJL.QFIM_obj(
            self.W, self.eps, self.para_type, LDtype
        )
        system = QJL.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        QJL.run(system)
        max_num = self.max_episode if type(self.max_episode) == int else self.max_episode[0]
        self.load_save(len(self.control_Hamiltonian), max_num)

    def CFIM(self, M=[], W=[]):
        r"""
        Choose CFI or $\mathrm{Tr}(WI^{-1})$ as the objective function. 
        In single parameter estimation the objective function is CFI and 
        in multiparameter estimation it will be $\mathrm{Tr}(WI^{-1})$.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.

        > **M:** `list`
            -- A set of positive operator-valued measure (POVM). The default measurement 
            is a set of rank-one symmetric informationally complete POVM (SIC-POVM).

        **Note:** 
            SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
            which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/
            solutions.html).
        """

        if M == []:
            M = SIC(len(self.rho0))
        M = [np.array(x, dtype=np.complex128) for x in M]

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        self.obj = QJL.CFIM_obj(M, self.W, self.eps, self.para_type)
        system = QJL.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        QJL.run(system)
        max_num = self.max_episode if type(self.max_episode) == int else self.max_episode[0]
        self.load_save(len(self.control_Hamiltonian), max_num)

    def HCRB(self, W=[]):
        """
        Choose HCRB as the objective function. 

        **Notes:** (1) In single parameter estimation, HCRB is equivalent to QFI, please
        choose QFI as the objective function. (2) GRAPE and auto-GRAPE are not available
        when the objective function is HCRB. Supported methods are PSO, DE and DDPG.

        Parameters
        ----------
        > **W:** `matrix` 
            -- Weight matrix.
        """

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        if len(self.Hamiltonian_derivative) == 1:
            print("Program terminated. In the single-parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the objective function."
                    )
        else:
            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W  

            self.obj = QJL.HCRB_obj(self.W, self.eps, self.para_type)
            system = QJL.QuanEstSystem(
                self.opt, self.alg, self.obj, self.dynamic, self.output
            )
            QJL.run(system)
            max_num = self.max_episode if type(self.max_episode) == int else self.max_episode[0]
        self.load_save(len(self.control_Hamiltonian), max_num)

    def mintime(self, f, W=[], M=[], method="binary", target="QFIM", LDtype="SLD"):
        """
        Search of the minimum time to reach a given value of the objective function.

        Parameters
        ----------
        > **f:** `float`
            -- The given value of the objective function.

        > **W:** `matrix`
            -- Weight matrix.

        > **M:** `list of matrices`
            -- A set of positive operator-valued measure (POVM). The default measurement 
            is a set of rank-one symmetric informationally complete POVM (SIC-POVM).

        > **method:** `string`
            -- Methods for searching the minimum time to reach the given value of the 
            objective function. Options are:  
            "binary" (default) -- Binary search (logarithmic search).  
            "forward" -- Forward search from the beginning of time.  

        > **target:** `string`
            -- Objective functions for searching the minimum time to reach the given 
            value of the objective function. Options are:  
            "QFIM" (default) -- Choose QFI (QFIM) as the objective function.  
            "CFIM" -- Choose CFI (CFIM) as the objective function.  
            "HCRB" -- Choose HCRB as the objective function.  

        > **LDtype:** `string`
            -- Types of QFI (QFIM) can be set as the objective function. Options are:  
            "SLD" (default) -- QFI (QFIM) based on symmetric logarithmic derivative (SLD).  
            "RLD" -- QFI (QFIM) based on right logarithmic derivative (RLD).  
            "LLD" -- QFI (QFIM) based on left logarithmic derivative (LLD).  
        """

        if not (method == "binary" or method == "forward"):
            raise ValueError(
                "{!r} is not a valid value for method, supported values are 'binary' and 'forward'.".format(
                    method
                )
            )

        if self.dynamics_type != "lindblad":
            raise ValueError(
                "Supported type of dynamics is Lindblad."
                )
        if self.savefile == True:
            warnings.warn(
                    "savefile is set to be False",
                    DeprecationWarning,
                )
        self.output = QJL.Output(self.opt)

        if len(self.Hamiltonian_derivative) > 1:
            f = 1 / f

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        if M != []:
            M = [np.array(x, dtype=np.complex128) for x in M]
            self.obj = QJL.CFIM_obj(M, self.W, self.eps, self.para_type)
        else:
            if target == "HCRB":
                if self.para_type == "single_para":
                    print(
                        "Program terminated. In the single-parameter scenario, the HCRB is equivalent to the QFI. Please choose 'QFIM' as the objective function.")
                self.obj = QJL.HCRB_obj(
                    self.W, self.eps, self.para_type
                )
            elif target == "QFIM" or (
                LDtype == "SLD" and LDtype == "LLD" and LDtype == "RLD"
            ):
                self.obj = QJL.QFIM_obj(
                    self.W, self.eps, self.para_type, LDtype
                )
            else:
                raise ValueError(
                    "Please enter the correct values for target and LDtype. Supported target are 'QFIM', 'CFIM' and 'HCRB', supported LDtype are 'SLD', 'RLD' and 'LLD'."
                )

        system = QJL.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        QJL.mintime(method, f, system)
        max_num = self.max_episode if type(self.max_episode) == int else self.max_episode[0]
        self.load_save(len(self.control_Hamiltonian), max_num)

def ControlOpt(savefile=False, method="auto-GRAPE", **kwargs):

    if method == "auto-GRAPE":
        return ctrl.GRAPE_Copt(savefile=savefile, **kwargs, auto=True)
    elif method == "GRAPE":
        return ctrl.GRAPE_Copt(savefile=savefile, **kwargs, auto=False)
    elif method == "PSO":
        return ctrl.PSO_Copt(savefile=savefile, **kwargs)
    elif method == "DE":
        return ctrl.DE_Copt(savefile=savefile, **kwargs)
    elif method == "DDPG":
        raise ValueError(
            "'DDPG' is currently deprecated and will be fixed soon."    
            )
        # return ctrl.DDPG_Copt(savefile=savefile, **kwargs)
    else:
        raise ValueError(
            "{!r} is not a valid value for method, supported values are 'auto-GRAPE', 'GRAPE', 'PSO', 'DE', 'DDPG'.".format(method
            )
        )

def csv2npy_controls(controls, num):
    C_save = []
    N = int(len(controls) / num)
    for ci in range(N):
        C_tp = controls[ci * num : (ci + 1) * num]
        C_save.append(C_tp)
    np.save("controls", C_save)