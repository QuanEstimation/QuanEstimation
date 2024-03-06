import numpy as np
import h5py
from scipy.interpolate import interp1d
import warnings
import math
import os
from quanestimation import QJL
import quanestimation.ComprehensiveOpt as compopt
from quanestimation.Common.Common import gramschmidt, SIC


class ComprehensiveSystem:
    """
    Attributes
    ----------
    > **savefile:** `bool`
        -- Whether or not to save all the optimized variables (probe states, 
        control coefficients and measurements).  
        If set `True` then the optimized variables and the values of the 
        objective function obtained in all episodes will be saved during 
        the training. If set `False` the optimized variables in the final 
        episode and the values of the objective function in all episodes 
        will be saved.

    > **psi0:** `list of arrays`
        -- Initial guesses of states.

    > **ctrl0:** `list of arrays`
        -- Initial guesses of control coefficients.

    > **measurement0:** `list of arrays`
        -- Initial guesses of measurements.

    > **seed:** `int`
        -- Random seed.

    > **eps:** `float`
        -- Machine epsilon.
    """

    def __init__(self, savefile, psi0, ctrl0, measurement0, seed, eps):

        self.savefile = savefile
        self.ctrl0 = ctrl0
        self.psi0 = psi0
        self.eps = eps
        self.seed = seed
        self.measurement0 = measurement0
        
    def load_save_ctrls(self, cnum, max_episode):
        if os.path.exists("controls.dat"):
            fl = h5py.File("controls.dat",'r')
            dset = fl["controls"]
            if self.savefile:
                controls = np.array([[np.array(fl[fl[dset[i]][j]]) for j in range(cnum)] for i in range(max_episode)])
            else:
                controls = np.array([np.array(fl[dset[j]]) for j in range(cnum)])
            np.save("controls", controls)
        else: pass
        
    def load_save_ctrls_alt(self, cnum, max_episode):
        if os.path.exists("controls.dat"):
            fl = h5py.File("controls.dat",'r')
            dset = fl["controls"]
            if self.savefile:
                controls = np.array([[np.array(fl[fl[dset[i]][j]]) for j in range(cnum)] for i in range(max_episode)])
            else:
                controls = np.array([dset[:,i] for i in range(cnum)])
            np.save("controls", controls)
        else: pass
            
    def load_save_states(self, max_episode):
        if os.path.exists("states.dat"):
            fl = h5py.File("states.dat",'r')
            dset = fl["states"]
            if self.savefile:
                psi = np.array([np.array(fl[dset[i]]).view('complex') for i in range(max_episode)])
            else:
                psi = np.array(dset).view('complex')
            np.save("states", psi)
        else: pass
        
    def load_save_meas(self, mnum, max_episode):
        if os.path.exists("measurements.dat"):
            fl = h5py.File("measurements.dat",'r')
            dset = fl["measurements"]
            if self.savefile:
                mdata = np.array([[np.array(fl[fl[dset[i]][j]]).view('complex') for j in range(mnum)] for i in range(max_episode)])
            else:
                mdata = np.array([np.array(fl[dset[j]]).view('complex') for j in range(mnum)])
            np.save("measurements", mdata)
        else: pass

    def dynamics(self, tspan, H0, dH, Hc=[], ctrl=[], decay=[], ctrl_bound=[], dyn_method="expm"):
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
        self.ctrl = ctrl
        self.Hc = Hc

        if dyn_method == "expm":
            self.dyn_method = "Expm"
        elif dyn_method == "ode":
            self.dyn_method = "Ode"

        if type(H0) == np.ndarray:
            self.freeHamiltonian = np.array(H0, dtype=np.complex128)
            self.dim = len(self.freeHamiltonian)
        else:
            self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0[:-1]]
            self.dim = len(self.freeHamiltonian[0])

        QJLType_psi = QJL.Vector[QJL.Vector[QJL.ComplexF64]]
        if self.psi0 == []:
            np.random.seed(self.seed)
            r_ini = 2 * np.random.random(self.dim) - np.ones(self.dim)
            r = r_ini / np.linalg.norm(r_ini)
            phi = 2 * np.pi * np.random.random(self.dim)
            psi = np.array([r[i] * np.exp(1.0j * phi[i]) for i in range(self.dim)])
            self.psi0 = np.array(psi)
            self.psi = QJL.convert(QJLType_psi, [self.psi0]) # Initial guesses of states (a list of arrays)
        else:
            self.psi0 = np.array(self.psi0[0], dtype=np.complex128)
            self.psi = QJL.convert(QJLType_psi, self.psi)

        if Hc == []:
            Hc = [np.zeros((self.dim, self.dim))]
        self.control_Hamiltonian = [np.array(x, dtype=np.complex128) for x in Hc]

        if type(dH) != list:
            raise TypeError("The derivative of Hamiltonian should be a list!")

        if dH == []:
            dH = [np.zeros((self.dim, self.dim))]
        self.Hamiltonian_derivative = [np.array(x, dtype=np.complex128) for x in dH]

        if len(dH) == 1:
            self.para_type = "single_para"
        else:
            self.para_type = "multi_para"

        if decay == []:
            decay_opt = [np.zeros((self.dim, self.dim))]
            self.gamma = [0.0]
        else:
            decay_opt = [decay[i][0] for i in range(len(decay))]
            self.gamma = [decay[i][1] for i in range(len(decay))]
        self.decay_opt = [np.array(x, dtype=np.complex128) for x in decay_opt]

        if ctrl_bound == []:
            self.ctrl_bound = [-np.inf, np.inf]
        else:
            self.ctrl_bound = [float(ctrl_bound[0]), float(ctrl_bound[1])]

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
            self.ctrl0 = [np.array(ctrl0)]

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
        else: pass
        
        # ## TODO
        QJLType_ctrl = QJL.Vector[QJL.Vector[QJL.Vector[QJL.Float64]]] 
        self.ctrl0 = QJL.convert(QJLType_ctrl, [[c for c in ctrls ]for ctrls in self.ctrl0])
        
        QJLType_C = QJL.Vector[QJL.Vector[QJL.ComplexF64]]
        if self.measurement0 == []:
            np.random.seed(self.seed)
            M = [[] for i in range(self.dim)]
            for i in range(self.dim):
                r_ini = 2 * np.random.random(self.dim) - np.ones(self.dim)
                r = r_ini / np.linalg.norm(r_ini)
                phi = 2 * np.pi * np.random.random(self.dim)
                M[i] = [r[j] * np.exp(1.0j * phi[j]) for j in range(self.dim)]
            self.C = QJL.convert(QJLType_C, gramschmidt(np.array(M)))
            self.measurement0 = QJL.Vector([self.C])
        else:
            self.C = [self.measurement0[0][i] for i in range(len(self.rho0))]
            self.C = QJL.convert(QJLType_C, self.C)
            self.measurement0 = QJL.Vector([self.C])

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

        self.dynamics_type = "dynamics"

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
        dK_tp = [
            [np.array(dK[i][j], dtype=np.complex128) for j in range(para_num)]
            for i in range(k_num)
        ]
        self.K = [np.array(x, dtype=np.complex128) for x in K]
        self.dK = dK_tp

        if para_num == 1:
            self.para_type = "single_para"
        else:
            self.para_type = "multi_para"

        self.dim = len(K[0])
        QJLType_psi = QJL.Vector[QJL.Vector[QJL.ComplexF64]]
        if self.psi0 == []:
            np.random.seed(self.seed)
            r_ini = 2 * np.random.random(self.dim) - np.ones(self.dim)
            r = r_ini / np.linalg.norm(r_ini)
            phi = 2 * np.pi * np.random.random(self.dim)
            psi = np.array([r[i] * np.exp(1.0j * phi[i]) for i in range(self.dim)])
            self.psi0 = np.array(psi)
            self.psi = QJL.convert(QJLType_psi, [self.psi0]) # Initial guesses of states (a list of arrays)
        else:
            self.psi0 = np.array(self.psi0[0], dtype=np.complex128)
            self.psi = QJL.convert(QJLType_psi, self.psi)

        if self.measurement0 == []:
            np.random.seed(self.seed)
            M = [[] for i in range(self.dim)]
            for i in range(self.dim):
                r_ini = 2 * np.random.random(self.dim) - np.ones(self.dim)
                r = r_ini / np.linalg.norm(r_ini)
                phi = 2 * np.pi * np.random.random(self.dim)
                M[i] = [r[i] * np.exp(1.0j * phi[i]) for i in range(self.dim)]
            self.C = gramschmidt(np.array(M))
            self.measurement0 = [np.array([self.C[i] for i in range(len(self.psi))])]
        elif len(self.measurement0) >= 1:
            self.C = [self.measurement0[0][i] for i in range(len(self.psi))]
            self.C = [np.array(x, dtype=np.complex128) for x in self.C]

        self.dynamic = QJL.Kraus(list(self.psi0), self.K, self.dK)

        self.dynamics_type = "Kraus"

    def SC(self, W=[], M=[], target="QFIM", LDtype="SLD"):
        """
        Comprehensive optimization of the probe state and control (SC).

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.

        > **M:** `list of matrices`
            -- A set of positive operator-valued measure (POVM). The default measurement 
            is a set of rank-one symmetric informationally complete POVM (SIC-POVM).

        > **target:** `string`
            -- Objective functions for comprehensive optimization. Options are:  
            "QFIM" (default) -- choose QFI (QFIM) as the objective function.  
            "CFIM" -- choose CFI (CFIM) as the objective function.  
            "HCRB" -- choose HCRB as the objective function.  

        > **LDtype:** `string`
            -- Types of QFI (QFIM) can be set as the objective function. Options are:  
            "SLD" (default) -- QFI (QFIM) based on symmetric logarithmic derivative (SLD).  
            "RLD" -- QFI (QFIM) based on right logarithmic derivative (RLD).  
            "LLD" -- QFI (QFIM) based on left logarithmic derivative (LLD). 

        **Note:** 
            SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
            which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/
            solutions.html).
        """

        if self.dynamics_type != "dynamics":
            raise ValueError(
                "Supported type of dynamics is Lindblad."
                )

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
                        "Program terminated. In the single-parameter scenario, the HCRB is equivalent to the QFI. Please choose 'QFIM' as the objective function"
                    )
                else:
                    self.obj = QJL.HCRB_obj(self.W, self.eps, self.para_type)
            elif target == "QFIM" and (
                LDtype == "SLD" or LDtype == "RLD" or LDtype == "LLD"
            ):
                self.obj = QJL.QFIM_obj(
                    self.W, self.eps, self.para_type, LDtype
                )
            elif target == "CFIM":
                M = SIC(len(self.psi))
                self.obj = QJL.CFIM_obj(M, self.W, self.eps, self.para_type)
            else:
                raise ValueError(
                    "Please enter the correct values for target and LDtype. Supported target are 'QFIM', 'CFIM' and 'HCRB', supported LDtype are 'SLD', 'RLD' and 'LLD'."
                )

        self.opt = QJL.StateControlOpt(
            psi=self.psi, ctrl=self.control_coefficients, ctrl_bound=self.ctrl_bound, seed=self.seed
        )
        self.output = QJL.Output(self.opt, save=self.savefile)

        self.dynamic = QJL.Lindblad(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.control_Hamiltonian,
            self.control_coefficients,
            list(self.psi0),
            self.tspan,
            self.decay_opt,
            self.gamma,
            dyn_method = self.dyn_method,
            )
        system = QJL.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        QJL.run(system)

        max_num = self.max_episode if type(self.max_episode) == int else self.max_episode[0]
        self.load_save_states(max_num)
        self.load_save_ctrls(len(self.control_Hamiltonian), max_num)

    def CM(self, rho0, W=[]):
        """
        Comprehensive optimization of the control and measurement (CM).

        Parameters
        ----------
        > **rho0:** `matrix`
            -- Initial state (density matrix).

        > **W:** `matrix`
            -- Weight matrix.
        """

        if self.dynamics_type != "dynamics":
            raise ValueError(
                "Supported type of dynamics is Lindblad."
                )

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        self.rho0 = np.array(rho0, dtype=np.complex128)

        self.obj = QJL.CFIM_obj([], self.W, self.eps, self.para_type)
        self.opt = QJL.ControlMeasurementOpt(
            ctrl=self.control_coefficients, M=self.C, ctrl_bound=self.ctrl_bound, seed=self.seed
        )
        self.output = QJL.Output(self.opt, save=self.savefile)

        self.dynamic = QJL.Lindblad(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.rho0,
            self.tspan,
            self.decay_opt,
            self.gamma,
            dyn_method =self.dyn_method,
            )

        system = QJL.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        QJL.run(system)

        max_num = self.max_episode if type(self.max_episode) == int else self.max_episode[0]
        self.load_save_ctrls_alt(len(self.control_Hamiltonian), max_num)
        self.load_save_meas(self.dim, max_num)

    def SM(self, W=[]):
        """
        Comprehensive optimization of the probe state and measurement (SM).

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.
        """

        if self.dynamics_type == "dynamics":
            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W

            if self.Hc == [] or self.ctrl == []:
                freeHamiltonian = self.freeHamiltonian
            else:
                ctrl_num = len(self.ctrl)
                Hc_num = len(self.control_Hamiltonian)
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
                        self.ctrl = np.concatenate(
                            (self.ctrl, np.zeros(len(self.ctrl[0])))
                        )
                else:
                    pass

                if len(self.ctrl[0]) == 1:
                    if type(self.freeHamiltonian) == np.ndarray:
                        H0 = np.array(self.freeHamiltonian, dtype=np.complex128)
                        Hc = [
                            np.array(x, dtype=np.complex128)
                            for x in self.control_Hamiltonian
                        ]
                        Htot = H0 + sum(
                            [
                                self.control_Hamiltonian[i] * self.ctrl[i][0]
                                for i in range(ctrl_num)
                            ]
                        )
                        freeHamiltonian = np.array(Htot, dtype=np.complex128)
                    else:
                        H0 = [
                            np.array(x, dtype=np.complex128)
                            for x in self.freeHamiltonian
                        ]
                        Htot = []
                        for i in range(len(H0)):
                            Htot.append(
                                H0[i]
                                + sum(
                                    [
                                        self.control_Hamiltonian[i] * self.ctrl[i][0]
                                        for i in range(ctrl_num)
                                    ]
                                )
                            )
                        freeHamiltonian = [
                            np.array(x, dtype=np.complex128) for x in Htot
                        ]
                else:
                    if type(self.freeHamiltonian) != np.ndarray:
                        #### linear interpolation  ####
                        f = interp1d(self.tspan, self.freeHamiltonian, axis=0)
                    else: pass
                    number = math.ceil((len(self.tspan) - 1) / len(self.ctrl[0]))
                    if len(self.tspan) - 1 % len(self.ctrl[0]) != 0:
                        tnum = number * len(self.ctrl[0])
                        self.tspan = np.linspace(
                            self.tspan[0], self.tspan[-1], tnum + 1
                        )
                        if type(self.freeHamiltonian) != np.ndarray:
                            H0_inter = f(self.tspan)
                            self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0_inter]
                        else: pass
                    else: pass

                    if type(self.freeHamiltonian) == np.ndarray:
                        H0 = np.array(self.freeHamiltonian, dtype=np.complex128)
                        Hc = [
                            np.array(x, dtype=np.complex128)
                            for x in self.control_Hamiltonian
                        ]
                        self.ctrl = [np.array(self.ctrl[i]).repeat(number) for i in range(len(Hc))]
                        Htot = []
                        for i in range(len(self.ctrl[0])):
                            S_ctrl = sum(
                                [Hc[j] * self.ctrl[j][i] for j in range(len(self.ctrl))]
                            )
                            Htot.append(H0 + S_ctrl)
                        freeHamiltonian = [
                            np.array(x, dtype=np.complex128) for x in Htot
                        ]
                    else:
                        H0 = [
                            np.array(x, dtype=np.complex128)
                            for x in self.freeHamiltonian
                        ]
                        Hc = [
                            np.array(x, dtype=np.complex128)
                            for x in self.control_Hamiltonian
                        ]
                        self.ctrl = [np.array(self.ctrl[i]).repeat(number) for i in range(len(Hc))]
                        Htot = []
                        for i in range(len(self.ctrl[0])):
                            S_ctrl = sum(
                                [Hc[j] * self.ctrl[j][i] for j in range(len(self.ctrl))]
                            )
                            Htot.append(H0[i] + S_ctrl)
                        freeHamiltonian = [
                            np.array(x, dtype=np.complex128) for x in Htot
                        ]

            self.dynamic = QJL.Lindblad(
                freeHamiltonian,
                self.Hamiltonian_derivative,
                list(self.psi0),
                self.tspan,
                self.decay_opt,
                self.gamma,
                dyn_method = self.dyn_method,
            )
        elif self.dynamics_type == "Kraus":
            if W == []:
                W = np.eye(self.para_num)
            self.W = W
        else:
            raise ValueError(
                "Supported type of dynamics are Lindblad and Kraus."
                )

        self.obj = QJL.CFIM_obj([], self.W, self.eps, self.para_type)
        self.opt = QJL.StateMeasurementOpt(psi=list(self.psi), M=self.C, seed=self.seed)
        self.output = QJL.Output(self.opt, save=self.savefile)

        system = QJL.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        QJL.run(system)

        max_num = self.max_episode if type(self.max_episode) == int else self.max_episode[0]
        self.load_save_states(max_num)
        self.load_save_meas(self.dim, max_num)
        
    def SCM(self, W=[]):
        """
        Comprehensive optimization of the probe state, control and measurement (SCM).

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.
        """

        if self.dynamics_type != "dynamics":
            raise ValueError(
                "Supported type of dynamics is Lindblad."
                )
        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        self.obj = QJL.CFIM_obj([], self.W, self.eps, self.para_type)
        self.opt = QJL.StateControlMeasurementOpt(
            psi=self.psi, ctrl=self.control_coefficients, M=self.C, ctrl_bound=self.ctrl_bound, seed=self.seed
        )
        self.output = QJL.Output(self.opt, save=self.savefile)

        self.dynamic = QJL.Lindblad(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.control_Hamiltonian,
            self.control_coefficients,
            list(self.psi0),
            self.tspan,
            self.decay_opt,
            self.gamma,
            )

        system = QJL.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        QJL.run(system)

        max_num = self.max_episode if type(self.max_episode) == int else self.max_episode[0]
        self.load_save_states(max_num)
        self.load_save_ctrls(len(self.control_Hamiltonian), max_num)
        self.load_save_meas(self.dim, max_num)

def ComprehensiveOpt(savefile=False, method="DE", **kwargs):

    if method == "AD":
        return compopt.AD_Compopt(savefile=savefile, **kwargs)
    elif method == "PSO":
        return compopt.PSO_Compopt(savefile=savefile, **kwargs)
    elif method == "DE":
        return compopt.DE_Compopt(savefile=savefile, **kwargs)
    else:
        raise ValueError(
            "{!r} is not a valid value for method, supported values are 'AD', 'PSO', 'DE'.".format(
                method
            )
        )
