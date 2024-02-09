import numpy as np
import h5py
from scipy.interpolate import interp1d
import os
import math
import warnings
from quanestimation import QJL
import quanestimation.MeasurementOpt as Measure
from quanestimation.Common.Common import gramschmidt, sic_povm


class MeasurementSystem:
    """
    Attributes
    ----------
    > **mtype:** `string`
        -- The type of scenarios for the measurement optimization. Options are:  
        "projection" (default) -- Optimization of rank-one projective measurements.  
        "input" -- Find the optimal linear combination or the optimal rotated measurement 
        of a given set of POVM.

    > **minput:** `list`
        -- In the case of optimization of rank-one projective measurements, the 
        `minput` should keep empty. For finding the optimal linear combination and 
        the optimal rotated measurement of a given set of POVM, the input rule are 
        `minput=["LC", [Pi1,Pi2,...], m]` and `minput=["LC", [Pi1,Pi2,...]]` respectively.
        Here `[Pi1,Pi2,...]` represents a list of input POVM and `m` is the number of operators 
        of the output measurement. 

    > **savefile:** `bool`
        -- Whether or not to save all the measurements.  
        If set `True` then the measurements and the values of the 
        objective function obtained in all episodes will be saved during 
        the training. If set `False` the measurement in the final 
        episode and the values of the objective function in all episodes 
        will be saved.

   > **measurement0:** `list of arrays`
        -- Initial guesses of measurements.

    > **seed:** `int`
        -- Random seed.

    > **eps:** `float`
        -- Machine epsilon.

    > **load:** `bool`
        -- Whether or not to load measurements in the current location.  
        If set `True` then the program will load measurement from "measurements.csv"
        file in the current location and use it as the initial measurement.

    > **dyn_method:** `string`
        -- The method for solving the Lindblad dynamcs. Options are:
        "expm" (default) -- matrix exponential.
        "ode" -- ordinary differential equation solvers.  
    """

    def __init__(self, mtype, minput, savefile, measurement0, seed, eps, load):

        self.mtype = mtype
        self.minput = minput
        self.savefile = savefile
        self.eps = eps
        self.seed = seed
        self.load = load
        self.measurement0 = measurement0

    def load_save(self, mnum, max_episode):
        if os.path.exists("measurements.dat"):
            fl = h5py.File("measurements.dat",'r')
            dset = fl["measurements"]
            if self.savefile:
                mdata = np.array([[np.array(fl[fl[dset[i]][j]]).view('complex') for j in range(mnum)] for i in range(max_episode)])
            else:
                mdata = np.array([np.array(fl[dset[j]]).view('complex') for j in range(mnum)])
            np.save("measurements", mdata)
        else: pass

    def dynamics(self, tspan, rho0, H0, dH, Hc=[], ctrl=[], decay=[], dyn_method="expm"):
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
        self.rho0 = np.array(rho0, dtype=np.complex128)

        self.dynamics_type = "dynamics"

        if len(dH) == 1:
            self.para_type = "single_para"
        else:
            self.para_type = "multi_para"

        if dyn_method == "expm":
            self.dyn_method = "Expm"
        elif dyn_method == "ode":
            self.dyn_method = "Ode"

        if self.mtype == "projection":
            self.M_num = len(self.rho0)
            QJLType_C = QJL.Vector[QJL.Vector[QJL.ComplexF64]]
            
            if self.measurement0 == []:
                np.random.seed(self.seed)
                M = [[] for i in range(len(self.rho0))]
                for i in range(len(self.rho0)):
                    r_ini = 2 * np.random.random(len(self.rho0)) - np.ones(
                        len(self.rho0)
                    )
                    r = r_ini / np.linalg.norm(r_ini)
                    phi = 2 * np.pi * np.random.random(len(self.rho0))
                    M[i] = [r[j] * np.exp(1.0j * phi[j]) for j in range(len(self.rho0))]
                self.C = QJL.convert(QJLType_C, gramschmidt(np.array(M)))
                self.measurement0 = QJL.Vector([self.C])
            else:
                self.C = [self.measurement0[0][i] for i in range(len(self.rho0))]
                self.C = QJL.convert(QJLType_C, self.C)
                self.measurement0 = QJL.Vector([self.C])
            self.opt = QJL.Mopt_Projection(M=self.C, seed=self.seed)

        elif self.mtype == "input":
            if self.minput[0] == "LC":
                self.M_num = self.minput[2]
                ## optimize the combination of a set of SIC-POVM
                if self.minput[1] == []:
                    file_path = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "sic_fiducial_vectors/d%d.txt" % (len(self.rho0)),
                    )
                    data = np.loadtxt(file_path)
                    fiducial = data[:, 0] + data[:, 1] * 1.0j
                    fiducial = np.array(fiducial).reshape(len(fiducial), 1)
                    self.povm_basis = sic_povm(fiducial)
                else:
                    ## optimize the combination of a set of given POVMs
                    if type(self.minput[1]) != list:
                        raise TypeError("The given POVMs should be a list!")
                    else:
                        accu = len(str(int(1 / self.eps))) - 1
                        for i in range(len(self.minput[1])):
                            val, vec = np.linalg.eig(self.minput[1])
                            if np.all(val.round(accu) >= 0):
                                pass
                            else:
                                raise TypeError(
                                    "The given POVMs should be semidefinite!"
                                )
                        M = np.zeros(
                            (len(self.rho0), len(self.rho0)), dtype=np.complex128
                        )
                        for i in range(len(self.minput[1])):
                            M += self.minput[1][i]
                        if np.all(M.round(accu) - np.identity(len(self.rho0)) == 0):
                            pass
                        else:
                            raise TypeError(
                                "The sum of the given POVMs should be identity matrix!"
                            )
                        self.povm_basis = [
                            np.array(x, dtype=np.complex128) for x in self.minput[1]
                        ]

                if self.measurement0 == []:
                    np.random.seed(self.seed)
                    self.B = [
                        np.random.random(len(self.povm_basis))
                        for i in range(self.M_num)
                    ]
                    self.measurement0 = [self.B]
                elif len(self.measurement0) >= 1:
                    self.B = [self.measurement0[0][i] for i in range(self.M_num)]
                    self.measurement0 = [[m for m in m0] for m0 in self.measurement0]
                    
                
                QJLType_B = QJL.Vector[QJL.Vector[QJL.Float64]]
                QJLType_pb = QJL.Vector[QJL.Matrix[QJL.ComplexF64]]
                QJLType_m0 = QJL.Vector[QJL.Vector[QJL.Vector[QJL.ComplexF64]]]
                self.B = QJL.convert(QJLType_B, self.B)
                self.povm_basis = QJL.convert(QJLType_pb, self.povm_basis)
                self.measurement0 = QJL.convert(QJLType_m0, self.measurement0)
                
                self.opt = QJL.Mopt_LinearComb(
                    B=self.B, POVM_basis=self.povm_basis, M_num=self.M_num, seed=self.seed
                )

            elif self.minput[0] == "rotation":
                self.M_num = len(self.minput[1])
                ## optimize the coefficients of the rotation matrix
                if type(self.minput[1]) != list:
                    raise TypeError("The given POVMs should be a list!")
                else:
                    if self.minput[1] == []:
                        raise TypeError("The initial POVM should not be empty!")
                    accu = len(str(int(1 / self.eps))) - 1
                    for i in range(len(self.minput[1])):
                        val, vec = np.linalg.eig(self.minput[1])
                        if np.all(val.round(accu) >= 0):
                            pass
                        else:
                            raise TypeError("The given POVMs should be semidefinite!")
                    M = np.zeros((len(self.rho0), len(self.rho0)), dtype=np.complex128)
                    for i in range(len(self.minput[1])):
                        M += self.minput[1][i]
                    if np.all(M.round(accu) - np.identity(len(self.rho0)) == 0):
                        pass
                    else:
                        raise TypeError(
                            "The sum of the given POVMs should be identity matrix!"
                        )
                    self.povm_basis = [
                        np.array(x, dtype=np.complex128) for x in self.minput[1]
                    ]
                    self.mtype = "rotation"

                if self.measurement0 == []:
                    np.random.seed(self.seed)
                    self.s = np.random.random(len(self.rho0) ** 2)
                    self.measurement0 = [self.s]
                elif len(self.measurement0) >= 1:
                    self.s = [
                        self.measurement0[0][i]
                        for i in range(len(self.rho0) * len(self.rho0))
                    ]

                self.s = QJL.Vector(self.s)
                QJLType_pb = QJL.Vector[QJL.Matrix[QJL.ComplexF64]]
                self.povm_basis = QJL.convert(QJLType_pb, self.povm_basis)
                self.opt = QJL.Mopt_Rotation(
                    s=self.s, POVM_basis=self.povm_basis, Lambda=[], seed=self.seed
                )

            else:
                raise ValueError(
                    "{!r} is not a valid value for the first input of minput, supported values are 'LC' and 'rotation'.".format(
                        self.minput[0]
                    )
                )
        else:
            raise ValueError(
                "{!r} is not a valid value for mtype, supported values are 'projection' and 'input'.".format(
                    self.mtype
                )
            )

        if Hc == [] or ctrl == []:
            if type(H0) == np.ndarray:
                self.freeHamiltonian = np.array(H0, dtype=np.complex128)
            else:
                self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0]
        else:
            ctrl_num = len(ctrl)
            Hc_num = len(Hc)
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
                    ctrl = np.concatenate((ctrl, np.zeros(len(ctrl[0]))))
            else: pass

            if len(ctrl[0]) == 1:
                if type(H0) == np.ndarray:
                    H0 = np.array(H0, dtype=np.complex128)
                    Hc = [np.array(x, dtype=np.complex128) for x in Hc]
                    Htot = H0 + sum([Hc[i] * ctrl[i][0] for i in range(ctrl_num)])
                    self.freeHamiltonian = np.array(Htot, dtype=np.complex128)
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

        if type(dH) != list:
            raise TypeError("The derivative of Hamiltonian should be a list!")

        if dH == []:
            dH = [np.zeros((len(self.rho0), len(self.rho0)))]
        self.Hamiltonian_derivative = [np.array(x, dtype=np.complex128) for x in dH]

        if decay == []:
            decay_opt = [np.zeros((len(self.rho0), len(self.rho0)))]
            self.gamma = [0.0]
        else:
            decay_opt = [decay[i][0] for i in range(len(decay))]
            self.gamma = [decay[i][1] for i in range(len(decay))]
        self.decay_opt = [np.array(x, dtype=np.complex128) for x in decay_opt]

        if any(self.gamma):
            self.dynamic = QJL.Lindblad(
                self.freeHamiltonian,
                self.Hamiltonian_derivative,
                self.rho0,
                self.tspan,
                self.decay_opt,
                self.gamma,
                dyn_method = self.dyn_method,
            )
        else:
            self.dynamic = QJL.Lindblad(
                self.freeHamiltonian,
                self.Hamiltonian_derivative,
                self.rho0,
                self.tspan,
                dyn_method = self.dyn_method,
            )
        self.output = QJL.Output(self.opt, save=self.savefile)
        
        self.dynamics_type = "dynamics"


    def Kraus(self, rho0, K, dK):
        r"""
        The parameterization of a state is
        \begin{align}
        \rho=\sum_i K_i\rho_0K_i^{\dagger},
        \end{align} 

        where $\rho$ is the evolved density matrix, $K_i$ is the Kraus operator.

        Parameters
        ----------
        > **rho0:** `matrix`
            -- Initial state (density matrix).

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
        self.dK = [
            [np.array(dK[i][j], dtype=np.complex128) for j in range(para_num)]
            for i in range(k_num)
        ]
        self.rho0 = np.array(rho0, dtype=np.complex128)
        self.K = [np.array(x, dtype=np.complex128) for x in K]

        if para_num == 1:
            self.para_type = "single_para"
        else:
            self.para_type = "multi_para"

        if self.mtype == "projection":
            self.M_num = len(self.rho0)
            if self.measurement0 == []:
                np.random.seed(self.seed)
                M = [[] for i in range(len(self.rho0))]
                for i in range(len(self.rho0)):
                    r_ini = 2 * np.random.random(len(self.rho0)) - np.ones(
                        len(self.rho0)
                    )
                    r = r_ini / np.linalg.norm(r_ini)
                    phi = 2 * np.pi * np.random.random(len(self.rho0))
                    M[i] = [r[j] * np.exp(1.0j * phi[j]) for j in range(len(self.rho0))]
                self.C = gramschmidt(np.array(M))
                self.measurement0 = [self.C]
            else:
                self.C = [self.measurement0[0][i] for i in range(len(self.rho0))]
                self.C = [np.array(x, dtype=np.complex128) for x in self.C]
            self.opt = QJL.Mopt_Projection(M=self.C, seed=self.seed)

        elif self.mtype == "input":
            if self.minput[0] == "LC":
                self.M_num = self.minput[2]
                ## optimize the combination of a set of SIC-POVM
                if self.minput[1] == []:
                    file_path = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "sic_fiducial_vectors/d%d.txt" % (len(self.rho0)),
                    )
                    data = np.loadtxt(file_path)
                    fiducial = data[:, 0] + data[:, 1] * 1.0j
                    fiducial = np.array(fiducial).reshape(len(fiducial), 1)
                    self.povm_basis = sic_povm(fiducial)
                else:
                    ## optimize the combination of a set of given POVMs
                    if type(self.minput[1]) != list:
                        raise TypeError("The given POVMs should be a list!")
                    else:
                        accu = len(str(int(1 / self.eps))) - 1
                        for i in range(len(self.minput[1])):
                            val, vec = np.linalg.eig(self.minput[1])
                            if np.all(val.round(accu) >= 0):
                                pass
                            else:
                                raise TypeError(
                                    "The given POVMs should be semidefinite!"
                                )
                        M = np.zeros(
                            (len(self.rho0), len(self.rho0)), dtype=np.complex128
                        )
                        for i in range(len(self.minput[1])):
                            M += self.minput[1][i]
                        if np.all(M.round(accu) - np.identity(len(self.rho0)) == 0):
                            pass
                        else:
                            raise TypeError(
                                "The sum of the given POVMs should be identity matrix!"
                            )
                        self.povm_basis = [
                            np.array(x, dtype=np.complex128) for x in self.minput[1]
                        ]

                if self.measurement0 == []:
                    np.random.seed(self.seed)
                    self.B = [
                        np.random.random(len(self.povm_basis))
                        for i in range(self.M_num)
                    ]
                    self.measurement0 = [np.array(self.B)]
                elif len(self.measurement0) >= 1:
                    self.B = [
                        self.measurement0[0][i] for i in range(len(self.povm_basis))
                    ]
                self.opt = QJL.Mopt_LinearComb(
                    B=self.B, POVM_basis=self.povm_basis, M_num=self.M_num, seed=self.seed
                )

            elif self.minput[0] == "rotation":
                self.M_num = len(self.minput[1])
                ## optimize the coefficients of the rotation matrix
                if type(self.minput[1]) != list:
                    raise TypeError("The given POVMs should be a list!")
                else:
                    if self.minput[1] == []:
                        raise TypeError("The initial POVM should not be empty!")
                    accu = len(str(int(1 / self.eps))) - 1
                    for i in range(len(self.minput[1])):
                        val, vec = np.linalg.eig(self.minput[1])
                        if np.all(val.round(accu) >= 0):
                            pass
                        else:
                            raise TypeError("The given POVMs should be semidefinite!")
                    M = np.zeros((len(self.rho0), len(self.rho0)), dtype=np.complex128)
                    for i in range(len(self.minput[1])):
                        M += self.minput[1][i]
                    if np.all(M.round(accu) - np.identity(len(self.rho0)) == 0):
                        pass
                    else:
                        raise TypeError(
                            "The sum of the given POVMs should be identity matrix!"
                        )
                    self.povm_basis = [
                        np.array(x, dtype=np.complex128) for x in self.minput[1]
                    ]
                    self.mtype = "rotation"

                if self.measurement0 == []:
                    np.random.seed(self.seed)
                    self.s = np.random.random(len(self.rho0) ** 2)
                    self.measurement0 = [self.s]
                elif len(self.measurement0) >= 1:
                    self.s = [
                        self.measurement0[0][i]
                        for i in range(len(self.rho0) * len(self.rho0))
                    ]

                self.opt = QJL.Mopt_Rotation(
                    s=self.s, POVM_basis=self.povm_basis, Lambda=[], seed=self.seed
                )

            else:
                raise ValueError(
                    "{!r} is not a valid value for the first input of minput, supported values are 'LC' and 'rotation'.".format(
                        self.minput[0]
                    )
                )
        else:
            raise ValueError(
                "{!r} is not a valid value for mtype, supported values are 'projection' and 'input'.".format(
                    self.mtype
                )
            )

        self.dynamic = QJL.Kraus(self.rho0, self.K, self.dK)
        self.output = QJL.Output(self.opt, save=self.savefile)

        self.dynamics_type = "Kraus"

    def CFIM(self, W=[]):
        r"""
        Choose CFI or $\mathrm{Tr}(WI^{-1})$ as the objective function. 
        In single parameter estimation the objective function is CFI and 
        in multiparameter estimation it will be $\mathrm{Tr}(WI^{-1})$.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.
        """

        if self.dynamics_type == "dynamics":
            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W
        elif self.dynamics_type == "Kraus":
            if W == []:
                W = np.eye(self.para_num)
            self.W = W
        else:
            raise ValueError(
                "Supported type of dynamics are Lindblad and Kraus."
                )

        self.obj = QJL.CFIM_obj(
            [], self.W, self.eps, self.para_type
        )  #### m=[]
        system = QJL.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        QJL.run(system)
        max_num = self.max_episode if type(self.max_episode) == int else self.max_episode[0]
        self.load_save(self.M_num, max_num)


def MeasurementOpt(
    mtype="projection", minput=[], savefile=False, method="DE", **kwargs
):

    if method == "AD":
        return Measure.AD_Mopt(mtype, minput, savefile=savefile, **kwargs)
    elif method == "PSO":
        return Measure.PSO_Mopt(mtype, minput, savefile=savefile, **kwargs)
    elif method == "DE":
        return Measure.DE_Mopt(mtype, minput, savefile=savefile, **kwargs)
    else:
        raise ValueError(
            "{!r} is not a valid value for method, supported values are 'AD', 'PSO' and 'DE'.".format(
                method
            )
        )


def csv2npy_measurements(M, num):
    n = int(np.sqrt(len(M[0])))
    N = int(len(M) / num)
    M_save = []
    for mi in range(N):
        M_tp = M[mi * num : (mi + 1) * num]
        M_mi = [M_tp[i].reshape(n, n).T for i in range(num)]
        M_save.append(M_mi)
    np.save("measurements", M_save)


def load_measurements(M, num, indx=-1):
    n = int(np.sqrt(len(M[0])))
    N = int(len(M) / num)
    M_save = []
    for mi in range(N):
        M_tp = M[mi * num : (mi + 1) * num]
        M_mi = [M_tp[i].reshape(n, n).T for i in range(num)]
        M_save.append(M_mi)
    return M_save[indx]
