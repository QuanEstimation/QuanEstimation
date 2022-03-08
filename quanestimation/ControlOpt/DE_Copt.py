import numpy as np
from julia import Main
import quanestimation.ControlOpt.ControlStruct as Control


class DE_Copt(Control.ControlSystem):
    def __init__(
        self,
        tspan,
        rho0,
        H0,
        dH,
        Hc,
        decay=[],
        ctrl_bound=[],
        popsize=10,
        ctrl0=[],
        max_episode=1000,
        c=1.0,
        cr=0.5,
        seed=1234,
        load=False,
    ):

        Control.ControlSystem.__init__(
            self, tspan, rho0, H0, Hc, dH, decay, ctrl_bound, ctrl0, load, eps=1e-8
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

        if ctrl0 == []:
            ini_population = [np.array(self.control_coefficients)]
        else:
            ini_population = ctrl0

        self.popsize = popsize
        self.ini_population = ini_population
        self.max_episode = max_episode
        self.c = c
        self.cr = cr
        self.seed = seed

    def QFIM(self, W=[], save_file=False):
        """
        Description: use differential evolution algorithm to update the control coefficients that maximize the
                     QFI (1/Tr(WF^{-1} with F the QFIM).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the control coefficients for each episode but overwrite in the next episode and
                                 all the QFI (Tr(WF^{-1})).
                           False: save the control coefficients for the last episode and all the QFI (Tr(WF^{-1})).
            --type: bool
        """

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        diffevo = Main.QuanEstimation.DE_Copt(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.rho0,
            self.tspan,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.ctrl_bound,
            self.W,
            self.eps,
        )
        Main.QuanEstimation.QFIM_DE_Copt(
            diffevo,
            self.popsize,
            self.ini_population,
            self.c,
            self.cr,
            self.seed,
            self.max_episode,
            save_file,
        )

    def CFIM(self, M, W=[], save_file=False):
        """
        Description: use differential evolution algorithm to update the control coefficients that maximize the
                     CFI (1/Tr(WF^{-1} with F the CFIM).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the control coefficients for each episode but overwrite in the next episode and all the CFI (Tr(WF^{-1})).
                           False: save the control coefficients for the last episode and all the CFI (Tr(WF^{-1})).
            --type: bool
        """
        M = [np.array(x, dtype=np.complex128) for x in M]

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        diffevo = Main.QuanEstimation.DE_Copt(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.rho0,
            self.tspan,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.ctrl_bound,
            self.W,
            self.eps,
        )
        Main.QuanEstimation.CFIM_DE_Copt(
            M,
            diffevo,
            self.popsize,
            self.ini_population,
            self.c,
            self.cr,
            self.seed,
            self.max_episode,
            save_file,
        )

    def HCRB(self, W=[], save_file=False):
        """
        Description: use differential evolution algorithm to update the control coefficients that maximize the
                     HCRB.

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the control coefficients for each episode but overwrite in the next episode and all the HCRB.
                           False: save the control coefficients for the last episode and all the HCRB.
            --type: bool
        """
        if len(self.Hamiltonian_derivative) == 1:
            warnings.warn(
                "In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the objection function \
                           for control optimization",
                DeprecationWarning,
            )
        else:

            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W

            diffevo = Main.QuanEstimation.DE_Copt(
                self.freeHamiltonian,
                self.Hamiltonian_derivative,
                self.rho0,
                self.tspan,
                self.decay_opt,
                self.gamma,
                self.control_Hamiltonian,
                self.control_coefficients,
                self.ctrl_bound,
                self.W,
                self.eps,
            )
            Main.QuanEstimation.HCRB_DE_Copt(
                diffevo,
                self.popsize,
                self.ini_population,
                self.c,
                self.cr,
                self.seed,
                self.max_episode,
                save_file,
            )

    def mintime(self, f, target="QFIM", dtype="SLD", W=[], M=[], method="binary"):
        if len(self.Hamiltonian_derivative) > 1:
            f = 1 / f
        M = [np.array(x, dtype=np.complex128) for x in M]

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        diffevo = Main.QuanEstimation.DE_Copt(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.rho0,
            self.tspan,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.ctrl_bound,
            self.W,
            self.eps,
        )

        if not (method == "binary" or method == "forward"):
            warnings.warn(
                "Method {!r} is currently not surppoted.".format(method),
                DeprecationWarning,
            )

        if target == "QFIM":
            Main.QuanEstimation.mintime(
                Main.eval("Val{:" + method + "}()"),
                "QFIM_DE_Copt",
                diffevo,
                f,
                self.popsize,
                self.ini_population,
                self.c,
                self.cr,
                self.seed,
                self.max_episode,
                dtype,
            )
        elif target == "CFIM":
            Main.QuanEstimation.mintime(
                Main.eval("Val{:" + method + "}()"),
                "CFIM_DE_Copt",
                diffevo,
                f,
                M,
                self.popsize,
                self.ini_population,
                self.c,
                self.cr,
                self.seed,
                self.max_episode,
                dtype,
            )
        elif target == "HCRB":
            Main.QuanEstimation.mintime(
                Main.eval("Val{:" + method + "}()"),
                "HCRB_DE_Copt",
                diffevo,
                f,
                self.popsize,
                self.ini_population,
                self.c,
                self.cr,
                self.seed,
                self.max_episode,
                dtype,
            )
        else:
            warnings.warn(
                "DE is not available with the objective function {!r}.".format(target),
                DeprecationWarning,
            )
