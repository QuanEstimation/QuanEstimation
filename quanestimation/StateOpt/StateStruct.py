import numpy as np
import h5py
from scipy.interpolate import interp1d
import os
import math
import warnings
from quanestimation import QJL
import quanestimation.StateOpt as stateoptimize
from quanestimation.Common.Common import SIC


class StateSystem:
    """
    Attributes
    ----------
    > **savefile:**  `bool`
        -- Whether or not to save all the states.
        If set `True` then the states and the values of the objective function 
        obtained in all episodes will be saved during the training. If set `False` 
        the state in the final episode and the values of the objective function in 
        all episodes will be saved.

    > **psi0:** `list of arrays`
        -- Initial guesses of states.

    > **seed:** `int`
        -- Random seed.

    > **eps:** `float`
        -- Machine epsilon.

    > **load:** `bool`
        -- Whether or not to load states in the current location.
        If set `True` then the program will load state from "states.csv"
        file in the current location and use it as the initial state.
    """

    def __init__(self, savefile, psi0, seed, eps, load):

        self.savefile = savefile
        self.psi0 = psi0
        self.psi = psi0
        self.eps = eps
        self.seed = seed

        if load == True:
            if os.path.exists("states.npy"):
                self.psi0 = [np.load("states.npy", dtype=np.complex128)]

    def load_save(self, max_episode):
        if os.path.exists("states.dat"):
            fl = h5py.File("states.dat",'r')
            dset = fl["states"]
            if self.savefile:
                psi = np.array([np.array(fl[dset[i]]).view('complex') for i in range(max_episode)])
            else:
                psi = np.array(dset).view('complex')
            np.save("states", psi)
        else: pass

    def dynamics(self, tspan, H0, dH, Hc=[], ctrl=[], decay=[], dyn_method="expm"):
        r"""
        The dynamics of a density matrix is of the form 
        
        \begin{align}
        \partial_t\rho &=\mathcal{L}\rho \nonumber \\
        &=-i[H,\rho]+\sum_i \gamma_i\left(\Gamma_i\rho\Gamma^{\dagger}_i-\frac{1}{2}
        \left\{\rho,\Gamma^{\dagger}_i \Gamma_i \right\}\right),
        \end{align}

        where $\rho$ is the evolved density matrix, H is the Hamiltonian of the 
        system, $\Gamma_i$ and $\gamma_i$ are the $i\mathrm{th}$ decay 
        operator and corresponding decay rate.

        Parameters
        ----------
        > **tspan:** `array`
            -- Time length for the evolution.

        > **H0:** `matrix or list`
            -- Free Hamiltonian. It is a matrix when the free Hamiltonian is time-
            independent and a list of length equal to `tspan` when it is time-dependent.

        > **dH:** `list`
            -- Derivatives of the free Hamiltonian on the unknown parameters to be 
            estimated. For example, dH[0] is the derivative vector on the first 
            parameter.

        > **Hc:** `list`
            -- Control Hamiltonians.

        > **ctrl:** `list of arrays`
            -- Control coefficients.

        > **decay:** `list`
            -- Decay operators and the corresponding decay rates. Its input rule is 
            decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$,$\gamma_2$],...], where $\Gamma_1$ 
            $(\Gamma_2)$ represents the decay operator and $\gamma_1$ $(\gamma_2)$ is the 
            corresponding decay rate.

        > **dyn_method:** `string`
            -- Setting the method for solving the Lindblad dynamics. Options are:  
            "expm" (default) -- Matrix exponential.  
            "ode" -- Solving the differential equations directly.
        """

        self.tspan = tspan

        if dyn_method == "expm":
            self.dyn_method = "Expm"
        elif dyn_method == "ode":
            self.dyn_method = "Ode"

        if Hc == [] or ctrl == []:
            if type(H0) == np.ndarray:
                self.freeHamiltonian = np.array(H0, dtype=np.complex128)
                self.dim = len(self.freeHamiltonian)
            else:
                self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0]
                self.dim = len(self.freeHamiltonian[0])
        else:
            ctrl_num = len(ctrl)
            Hc_num = len(Hc)
            if Hc_num < ctrl_num:
                raise TypeError(
                    "There are %d control Hamiltonians but %d coefficients sequences: \
                                 too many coefficients sequences."
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
                    ctrl = np.concatenate((ctrl, np.zeros(len(ctrl[0]))))
            else: pass

            if len(ctrl[0]) == 1:
                if type(H0) == np.ndarray:
                    H0 = np.array(H0, dtype=np.complex128)
                    Hc = [np.array(x, dtype=np.complex128) for x in Hc]
                    Htot = H0 + sum([Hc[i] * ctrl[i][0] for i in range(ctrl_num)])
                    self.freeHamiltonian = np.array(Htot, dtype=np.complex128)
                    self.dim = len(self.freeHamiltonian)
                else:
                    H0 = [np.array(x, dtype=np.complex128) for x in H0]
                    Htot = []
                    for i in range(len(H0)):
                        Htot.append(
                            H0[i] + sum([Hc[i] * ctrl[i][0] for i in range(ctrl_num)])
                        )
                    self.freeHamiltonian = [
                        np.array(x, dtype=np.complex128) for x in Htot
                    ]
                    self.dim = len(self.freeHamiltonian[0])
            else:
                if type(H0) != np.ndarray:
                    #### linear interpolation  ####
                    f = interp1d(self.tspan, H0, axis=0)
                else: pass
                number = math.ceil((len(self.tspan) - 1) / len(ctrl[0]))
                if len(self.tspan) - 1 % len(ctrl[0]) != 0:
                    tnum = number * len(ctrl[0])
                    self.tspan = np.linspace(self.tspan[0], self.tspan[-1], tnum + 1)
                    if type(H0) != np.ndarray:
                        H0_inter = f(self.tspan)
                        H0 = [np.array(x, dtype=np.complex128) for x in H0_inter]
                    else: pass
                else: pass

                if type(H0) == np.ndarray:
                    H0 = np.array(H0, dtype=np.complex128)
                    Hc = [np.array(x, dtype=np.complex128) for x in Hc]
                    ctrl = [np.array(ctrl[i]).repeat(number) for i in range(len(Hc))]
                    Htot = []
                    for i in range(len(ctrl[0])):
                        S_ctrl = sum([Hc[j] * ctrl[j][i] for j in range(len(ctrl))])
                        Htot.append(H0 + S_ctrl)
                    self.freeHamiltonian = [
                        np.array(x, dtype=np.complex128) for x in Htot
                    ]
                    self.dim = len(self.freeHamiltonian)
                else:
                    H0 = [np.array(x, dtype=np.complex128) for x in H0]
                    Hc = [np.array(x, dtype=np.complex128) for x in Hc]
                    ctrl = [np.array(ctrl[i]).repeat(number) for i in range(len(Hc))]
                    Htot = []
                    for i in range(len(ctrl[0])):
                        S_ctrl = sum([Hc[j] * ctrl[j][i] for j in range(len(ctrl))])
                        Htot.append(H0[i] + S_ctrl)
                    self.freeHamiltonian = [
                        np.array(x, dtype=np.complex128) for x in Htot
                    ]
                    self.dim = len(self.freeHamiltonian[0])
                    
        QJLType_psi = QJL.Vector[QJL.Vector[QJL.ComplexF64]]
        if self.psi0 == []:
            np.random.seed(self.seed)
            r_ini = 2 * np.random.random(self.dim) - np.ones(self.dim)
            r = r_ini / np.linalg.norm(r_ini)
            phi = 2 * np.pi * np.random.random(self.dim)
            psi0 = [r[i] * np.exp(1.0j * phi[i]) for i in range(self.dim)]
            self.psi0 = np.array(psi0)
            self.psi = QJL.convert(QJLType_psi, [self.psi0]) # Initial guesses of states (a list of arrays)
        else:
            self.psi0 = np.array(self.psi0[0], dtype=np.complex128)
            self.psi = QJL.convert(QJLType_psi, self.psi)
        
        if type(dH) != list:
            raise TypeError("The derivative of Hamiltonian should be a list!")

        if dH == []:
            dH = [np.zeros((len(self.psi0), len(self.psi0)))]
        self.Hamiltonian_derivative = [np.array(x, dtype=np.complex128) for x in dH]

        if decay == []:
            decay_opt = [np.zeros((len(self.psi0), len(self.psi0)))]
            self.gamma = [0.0]
        else:
            decay_opt = [decay[i][0] for i in range(len(decay))]
            self.gamma = [decay[i][1] for i in range(len(decay))]
        self.decay_opt = [np.array(x, dtype=np.complex128) for x in decay_opt]

        self.opt = QJL.StateOpt(psi=self.psi0, seed=self.seed)
        if any(self.gamma):
            self.dynamic = QJL.Lindblad(
                self.freeHamiltonian,
                self.Hamiltonian_derivative,
                list(self.psi0),
                self.tspan,
                self.decay_opt,
                self.gamma,
                dyn_method = self.dyn_method,
            )
        else:
            self.dynamic = QJL.Lindblad(
                self.freeHamiltonian,
                self.Hamiltonian_derivative,
                list(self.psi0),
                self.tspan,
                dyn_method = self.dyn_method,
            )
        self.output = QJL.Output(self.opt, save=self.savefile)

        self.dynamics_type = "dynamics"
        if len(self.Hamiltonian_derivative) == 1:
            self.para_type = "single_para"
        else:
            self.para_type = "multi_para"

    def Kraus(self, K, dK):
        r"""
        The parameterization of a state is
        \begin{align}
        \rho=\sum_i K_i\rho_0K_i^{\dagger},
        \end{align} 

        where $\rho$ is the evolved density matrix, $K_i$ is the Kraus operator.

        Parameters
        ----------
        > **K:** `list`
            -- Kraus operators.

        > **dK:** `list`
            -- Derivatives of the Kraus operators on the unknown parameters to be 
            estimated. For example, dK[0] is the derivative vector on the first 
            parameter.
        """

        k_num = len(K)
        para_num = len(dK[0])
        self.para_num = para_num
        self.K = [np.array(x, dtype=np.complex128) for x in K]
        self.dK = [
            [np.array(dK[i][j], dtype=np.complex128) for j in range(para_num)]
            for i in range(k_num)
        ]

        self.dim = len(self.K[0])

        if self.psi0 == []:
            np.random.seed(self.seed)
            r_ini = 2 * np.random.random(self.dim) - np.ones(self.dim)
            r = r_ini / np.linalg.norm(r_ini)
            phi = 2 * np.pi * np.random.random(self.dim)
            psi0 = [r[i] * np.exp(1.0j * phi[i]) for i in range(self.dim)]
            self.psi0 = np.array(psi0)  # Initial state (an array)
            self.psi = [self.psi0] # Initial guesses of states (a list of arrays)
        else:
            self.psi0 = np.array(self.psi0[0], dtype=np.complex128)
            self.psi = [np.array(psi, dtype=np.complex128) for psi in self.psi]

        self.opt = QJL.StateOpt(psi=self.psi0, seed=self.seed)
        self.dynamic = QJL.Kraus(list(self.psi0), self.K, self.dK)
        self.output = QJL.Output(self.opt, save=self.savefile)

        self.dynamics_type = "Kraus"
        if para_num == 1:
            self.para_type = "single_para"
        else:
            self.para_type = "multi_para"

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

        if self.dynamics_type == "dynamics":
            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W

        elif self.dynamics_type == "Kraus":
            if W == []:
                W = np.eye(self.para_num)
            self.W = W
        else:
            pass

        self.obj = QJL.QFIM_obj(
            self.W, self.eps, self.para_type, LDtype
        )
        system = QJL.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        QJL.run(system)

        max_num = self.max_episode if isinstance(self.max_episode, int) else self.max_episode[0]
        self.load_save(max_num)

    def CFIM(self, M=[], W=[]):
        r"""
        Choose CFI or $\mathrm{Tr}(WI^{-1})$ as the objective function. 
        In single parameter estimation the objective function is CFI and 
        in multiparameter estimation it will be $\mathrm{Tr}(WI^{-1})$.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.

        > **M:** `list of matrices`
            -- A set of positive operator-valued measure (POVM). The default measurement 
            is a set of rank-one symmetric informationally complete POVM (SIC-POVM).

        **Note:** 
            SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
            which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/
            solutions.html).
        """

        if M == []:
            M = SIC(len(self.psi0))
        M = [np.array(x, dtype=np.complex128) for x in M]

        if self.dynamics_type == "dynamics":
            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W

        elif self.dynamics_type == "Kraus":
            if W == []:
                W = np.eye(self.para_num)
            self.W = W

        self.obj = QJL.CFIM_obj(M, self.W, self.eps, self.para_type)
        system = QJL.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        QJL.run(system)

        max_num = self.max_episode if isinstance(self.max_episode, int) else self.max_episode[0]
        self.load_save(max_num)

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
        
        if self.dynamics_type == "dynamics":
            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W
            if len(self.Hamiltonian_derivative) == 1:
                print("Program terminated. In the single-parameter scenario, the HCRB is equivalent to the QFI. Please choose 'QFIM' as the objective function"
                    )
            else: pass

        elif self.dynamics_type == "Kraus":
            if W == []:
                W = np.eye(self.para_num)
            self.W = W
            if len(self.dK) == 1:
                raise ValueError(
                    "In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the target function for control optimization",
                )
            else: pass
        else:
            raise ValueError(
                "Supported type of dynamics are Lindblad and Kraus."
                )

        self.obj = QJL.HCRB_obj(self.W, self.eps, self.para_type)
        system = QJL.QuanEstSystem(
                self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        QJL.run(system)

        max_num = self.max_episode if isinstance(self.max_episode, int) else self.max_episode[0]
        self.load_save(max_num)


def StateOpt(savefile=False, method="AD", **kwargs):

    if method == "AD":
        return stateoptimize.AD_Sopt(savefile=savefile, **kwargs)
    elif method == "PSO":
        return stateoptimize.PSO_Sopt(savefile=savefile, **kwargs)
    elif method == "DE":
        return stateoptimize.DE_Sopt(savefile=savefile, **kwargs)
    elif method == "DDPG":
        raise ValueError(
            "'DDPG' is currently deprecated and will be fixed soon."    
            )
        # return stateoptimize.DDPG_Sopt(savefile=savefile, **kwargs)
    elif method == "NM":
        return stateoptimize.NM_Sopt(savefile=savefile, **kwargs)
    elif method == "RI":
        return stateoptimize.RI_Sopt(savefile=savefile, **kwargs)
    else:
        raise ValueError(
            "{!r} is not a valid value for method, supported values are 'AD', 'PSO', 'DE', 'NM', 'DDPG' and 'RI.".format(
                method
            )
        )


def csv2npy_states(states, num=1):
    S_save = []
    N = int(len(states) / num)
    for si in range(N):
        S_tp = states[si * num : (si + 1) * num]
        S_save.append(S_tp)
    np.save("states", S_save)
