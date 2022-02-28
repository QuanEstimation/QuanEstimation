import numpy as np
from julia import Main
import warnings
import quanestimation.ComprehensiveOpt.ComprehensiveStruct as Comp


class DE_Compopt(Comp.ComprehensiveSystem):
    def __init__(
        self,
        psi0=[],
        ctrl0=[],
        measurement0=[],
        popsize=10,
        max_episode=1000,
        c=1.0,
        cr=0.5,
        seed=1234,
    ):

        Comp.ComprehensiveSystem.__init__(
            self,
            psi0,
            ctrl0,
            measurement0,
            seed,
            eps=1e-8,
        )

        """
        --------
        inputs
        --------
        popsize:
           --description: the number of populations.
           --type: int
        
        ctrl0:
           --description: initial guesses of controls.
           --type: array

        max_episode:
            --description: max number of the training episodes.
            --type: int
        
        c:
            --description: mutation constant.
            --type: float

        cr:
            --description: crossover constant.
            --type: float
        
        seed:
            --description: random seed.
            --type: int
        
        """
        self.ini_population = self.psi
        
        if ctrl0 == []:
            self.ctrl0 = [np.array(self.control_coefficients)]

        self.popsize = popsize
        self.max_episode = max_episode
        self.c = c
        self.cr = cr
        self.seed = seed

    def SC(self, target="QFIM", M=[], W=[], save_file=False):
        if self.dynamics_type != "dynamics":
            raise ValueError(
                "{!r} is not a valid type for dynamics, supported type is 'Lindblad dynamics'.".format(
                    self.dynamics_type
                )
            )

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        diffevo = Main.QuanEstimation.SC_Compopt(
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
        if target == "QFIM":
            Main.QuanEstimation.SC_DE_Compopt(
                diffevo,
                self.popsize,
                self.ini_population,
                self.ctrl0,
                self.c,
                self.cr,
                self.seed,
                self.max_episode,
                save_file,
            )
            self.load_save_state()
        elif target == "CFIM":
            if M == []:
                raise ValueError("M should not be empty.")
            else:
                M = [np.array(x, dtype=np.complex128) for x in M]
                Main.QuanEstimation.SC_DE_Compopt(
                    M,
                    diffevo,
                    self.popsize,
                    self.ini_population,
                    self.ctrl0,
                    self.c,
                    self.cr,
                    self.seed,
                    self.max_episode,
                    save_file,
                )
                self.load_save_state()
        else:
            raise ValueError(
                "{!r} is not a valid value for target, supported values are 'QFIM', 'CFIM'.".format(
                    target
                )
            )

    def CM(self, rho0, save_file=False):
        if self.dynamics_type != "dynamics":
            raise ValueError(
                "{!r} is not a valid type for dynamics, supported type is 'Lindblad dynamics'.".format(
                    self.dynamics_type
                )
            )

        rho0 = np.array(rho0, dtype=np.complex128)
        diffevo = Main.QuanEstimation.CM_Compopt(
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
        Main.QuanEstimation.CM_DE_Compopt(
            rho0,
            diffevo,
            self.popsize,
            self.ini_population,
            self.measurement0,
            self.c,
            self.cr,
            self.seed,
            self.max_episode,
            save_file,
        )
        self.load_save_meas()

    def SM(self, W=[], save_file=False):
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

            diffevo = Main.QuanEstimation.SM_Compopt(
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
            Main.QuanEstimation.SM_DE_Compopt(
                diffevo,
                self.popsize,
                self.ini_population,
                self.measurement0,
                self.c,
                self.cr,
                self.seed,
                self.max_episode,
                save_file,
            )
        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W

            diffevo = Main.QuanEstimation.SM_Compopt_Kraus(
                self.K,
                self.dK,
                self.psi0,
                self.M,
                self.W,
                self.eps,
            )
            Main.QuanEstimation.SM_DE_Compopt(
                diffevo,
                self.popsize,
                self.ini_population,
                self.measurement0,
                self.c,
                self.cr,
                self.seed,
                self.max_episode,
                save_file,
            )

        self.load_save_meas()

    def SCM(self, save_file=False):
        if self.dynamics_type != "dynamics":
            raise ValueError(
                "{!r} is not a valid type for dynamics, supported type is 'Lindblad dynamics'.".format(
                    self.dynamics_type
                )
            )

        diffevo = Main.QuanEstimation.SCM_Compopt(
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
        Main.QuanEstimation.SCM_DE_Compopt(
            diffevo,
            self.popsize,
            self.ini_population,
            self.ctrl0,
            self.measurement0,
            self.c,
            self.cr,
            self.seed,
            self.max_episode,
            save_file,
        )
        self.load_save_state()
        self.load_save_meas()
