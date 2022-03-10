import numpy as np
from julia import Main
import quanestimation.ComprehensiveOpt.ComprehensiveStruct as Comp
from quanestimation.Common.common import SIC


class PSO_Compopt(Comp.ComprehensiveSystem):
    def __init__(
        self,
        save_file=False,
        particle_num=10,
        psi0=[],
        ctrl0=[],
        measurement0=[],
        max_episode=[1000, 100],
        c0=1.0,
        c1=2.0,
        c2=2.0,
        seed=1234,
        eps=1e-8,
    ):

        Comp.ComprehensiveSystem.__init__(
            self, psi0, ctrl0, measurement0, save_file, seed, eps
        )

        """
        --------
        inputs
        --------
        save_file:
            --description: True: save the states (or controls, measurements) and the value of the 
                                 target function for each episode.
                           False: save the states (or controls, measurements) and all the value 
                                   of the target function for the last episode.
            --type: bool 
        particle_num:
           --description: the number of particles.
           --type: int
        
        psi0:
           --description: initial guesses of states (kets).
           --type: array
           
        ctrl0:
            --description: initial control coefficients.
            --type: list (of vector)
            
        measurement0:
           --description: a set of POVMs.
           --type: list (of vector)
           
        max_episode:
            --description: max number of the training episodes.
            --type: int or array
        
        c0:
            --description: damping factor that assists convergence.
            --type: float

        c1:
            --description: exploitation weight that attract the particle to its best previous position.
            --type: float
        
        c2:
            --description: exploitation weight that attract the particle to the best position in the neighborhood.
            --type: float
        
        seed:
            --description: random seed.
            --type: int
        
        """

        self.particle_num = particle_num
        self.ini_particle = self.psi
        self.max_episode = max_episode
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

    def SC(self, W=[], M=[], target="QFIM", dtype="SLD"):
        """
        Description: use PSO algorithm to optimize states and control coefficients.

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
            raise ValueError(
                "{!r} is not a valid type for dynamics, supported type is \
                             Lindblad dynamics.".format(
                    self.dynamics_type
                )
            )

        if M == []:
            M = SIC(len(self.rho0))
        M = [np.array(x, dtype=np.complex128) for x in M]

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        pso = Main.QuanEstimation.SC_Compopt(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.psi0,
            self.tspan,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.ctrl_bound,
            self.W,
            self.eps,
        )

        if M != []:
            Main.QuanEstimation.SC_PSO_Compopt(
                M,
                pso,
                self.max_episode,
                self.particle_num,
                self.ini_particle,
                self.ctrl0,
                self.c0,
                self.c1,
                self.c2,
                self.seed,
                self.save_file,
            )
            self.load_save_state()
        else:
            if target == "HCRB":
                if len(self.Hamiltonian_derivative) == 1:
                    warnings.warn(
                        "In single parameter scenario, HCRB is equivalent to QFI. \
                           Please choose QFIM as the target function for control optimization",
                        DeprecationWarning,
                    )
                else:
                    pass  #### to be done
            elif target == "QFIM" and dtype == "SLD":
                Main.QuanEstimation.SC_PSO_Compopt(
                    pso,
                    self.max_episode,
                    self.particle_num,
                    self.ini_particle,
                    self.ctrl0,
                    self.c0,
                    self.c1,
                    self.c2,
                    self.seed,
                    self.save_file,
                )
                self.load_save_state()
            elif target == "QFIM" and dtype == "RLD":
                pass  #### to be done
            elif target == "QFIM" and dtype == "LLD":
                pass  #### to be done
            else:
                raise ValueError(
                    "Please enter the correct values for target and dtype.\
                                  Supported target are 'QFIM', 'CFIM' and 'HCRB',  \
                                  supported dtype are 'SLD', 'RLD' and 'LLD'."
                )

    def CM(self, rho0, W=[]):
        """
        Description: use PSO algorithm to optimize control coefficients and the measurements.

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
            raise ValueError(
                "{!r} is not a valid type for dynamics, supported type is \
                             Lindblad dynamics.".format(
                    self.dynamics_type
                )
            )

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        rho0 = np.array(rho0, dtype=np.complex128)
        pso = Main.QuanEstimation.CM_Compopt(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.tspan,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.ctrl_bound,
            self.M,
            self.W,
            self.eps,
        )
        Main.QuanEstimation.CM_PSO_Compopt(
            rho0,
            pso,
            self.max_episode,
            self.particle_num,
            self.ini_particle,
            self.ctrl0,
            self.measurement0,
            self.c0,
            self.c1,
            self.c2,
            self.seed,
            self.save_file,
        )
        self.load_save_meas()

    def SM(self, W=[]):
        """
        Description: use PSO algorithm to optimize states and the measurements.

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
                Hc = [
                    np.array(x, dtype=np.complex128) for x in self.control_Hamiltonian
                ]
                Htot = H0 + sum(
                    [
                        Hc[i] * self.control_coefficients[i][0]
                        for i in range(len(self.control_coefficients))
                    ]
                )
                freeHamiltonian = np.array(Htot, dtype=np.complex128)
            else:
                H0 = np.array(self.freeHamiltonian, dtype=np.complex128)
                Hc = [
                    np.array(x, dtype=np.complex128) for x in self.control_Hamiltonian
                ]
                Htot = []
                for i in range(len(self.control_coefficients[0])):
                    S_ctrl = sum(
                        [
                            Hc[j] * self.control_coefficients[j][i]
                            for j in range(len(self.control_coefficients))
                        ]
                    )
                    Htot.append(H0 + S_ctrl)
                freeHamiltonian = [np.array(x, dtype=np.complex128) for x in Htot]

            pso = Main.QuanEstimation.SM_Compopt(
                freeHamiltonian,
                self.Hamiltonian_derivative,
                self.psi0,
                self.tspan,
                self.decay_opt,
                self.gamma,
                self.M,
                self.W,
                self.eps,
            )
            Main.QuanEstimation.SM_PSO_Compopt(
                pso,
                self.max_episode,
                self.particle_num,
                self.ini_particle,
                self.measurement0,
                self.c0,
                self.c1,
                self.c2,
                self.seed,
                self.save_file,
            )
        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W

            pso = Main.QuanEstimation.SM_Compopt_Kraus(
                self.K, self.dK, self.psi0, self.M, self.W, self.eps
            )
            Main.QuanEstimation.SM_PSO_Compopt(
                pso,
                self.max_episode,
                self.particle_num,
                self.ini_particle,
                self.measurement0,
                self.c0,
                self.c1,
                self.c2,
                self.seed,
                self.save_file,
            )
        self.load_save_meas()

    def SCM(self, W=[]):
        """
        Description: use PSO algorithm to optimize states, control coefficients and the measurements.

        ---------
        Inputs
        ---------

        W:
            --description: weight matrix.
            --type: matrix

        """
        if self.dynamics_type != "dynamics":
            raise ValueError(
                "{!r} is not a valid type for dynamics, supported type is \
                             Lindblad dynamics.".format(
                    self.dynamics_type
                )
            )

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        pso = Main.QuanEstimation.SCM_Compopt(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.psi0,
            self.tspan,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.ctrl_bound,
            self.M,
            self.W,
            self.eps,
        )
        Main.QuanEstimation.SCM_PSO_Compopt(
            pso,
            self.max_episode,
            self.particle_num,
            self.ini_particle,
            self.ctrl0,
            self.measurement0,
            self.c0,
            self.c1,
            self.c2,
            self.seed,
            self.save_file,
        )
        self.load_save_state()
        self.load_save_meas()
