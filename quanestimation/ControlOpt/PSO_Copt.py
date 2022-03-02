import numpy as np
from julia import Main
import quanestimation.ControlOpt.ControlStruct as Control


class PSO_Copt(Control.ControlSystem):
    def __init__(
        self,
        tspan,
        rho0,
        H0,
        dH,
        Hc,
        decay=[],
        ctrl_bound=[],
        W=[],
        particle_num=10,
        ctrl0=[],
        max_episode=[1000, 100],
        c0=1.0,
        c1=2.0,
        c2=2.0,
        seed=1234,
        load=False,
    ):

        Control.ControlSystem.__init__(
            self, tspan, rho0, H0, Hc, dH, decay, ctrl_bound, W, ctrl0, load, eps=1e-8
        )

        """
        -------- 
        inputs
        --------
        particle_num:
           --description: the number of particles.
           --type: int
        
        ctrl0:
           --description: initial guesses of controls.
           --type: array

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

        if ctrl0 == []:
            ini_particle = [np.array(self.control_coefficients)]
        else:
            ini_particle = ctrl0

        self.particle_num = particle_num
        self.ini_particle = ini_particle
        self.max_episode = max_episode
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

    def QFIM(self, save_file=False):
        """
        Description: use particle swarm optimization algorithm to update the control coefficients
                     that maximize the QFI (1/Tr(WF^{-1} with F the QFIM).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save all the control coefficients for each episode but overwrite in the next episode and all the QFI (Tr(WF^{-1})).
                           False: save the control coefficients for the last episode and all the QFI (Tr(WF^{-1})).
            --type: bool
        """
        pso = Main.QuanEstimation.PSO_Copt(
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
        Main.QuanEstimation.QFIM_PSO_Copt(
            pso,
            self.max_episode,
            self.particle_num,
            self.ini_particle,
            self.c0,
            self.c1,
            self.c2,
            self.seed,
            save_file,
        )

    def CFIM(self, M, save_file=False):
        """
        Description: use particle swarm optimization algorithm to update the control coefficients
                     that maximize the CFI (1/Tr(WF^{-1} with F the CFIM).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the control coefficients for each episode but overwrite in the next episode and all the CFI (Tr(WF^{-1})).
                           False: save the control coefficients for the last episode and all the CFI (Tr(WF^{-1})).
            --type: bool
        """
        M = [np.array(x, dtype=np.complex128) for x in M]
        pso = Main.QuanEstimation.PSO_Copt(
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
        Main.QuanEstimation.CFIM_PSO_Copt(
            M,
            pso,
            self.max_episode,
            self.particle_num,
            self.ini_particle,
            self.c0,
            self.c1,
            self.c2,
            self.seed,
            save_file,
        )

    def HCRB(self, save_file=False):
        """
        Description: use particle swarm optimization algorithm to update the control coefficients
                     that maximize the HCRB.

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
            pso = Main.QuanEstimation.PSO_Copt(
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
            Main.QuanEstimation.HCRB_PSO_Copt(
                pso,
                self.max_episode,
                self.particle_num,
                self.ini_particle,
                self.c0,
                self.c1,
                self.c2,
                self.seed,
                save_file,
            )

    def mintime(self, f, target="QFIM", W=[], M=[], method="binary"):
        if len(self.Hamiltonian_derivative) > 1:
            f = 1 / f
        M = [np.array(x, dtype=np.complex128) for x in M]
        pso = Main.QuanEstimation.PSO_Copt(
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
                "QFIM_PSO_Copt",
                pso,
                f,
                self.max_episode,
                self.particle_num,
                self.ini_particle,
                self.c0,
                self.c1,
                self.c2,
                self.seed,
            )
        elif target == "CFIM":
            Main.QuanEstimation.mintime(
                Main.eval("Val{:" + method + "}()"),
                "CFIM_PSO_Copt",
                pso,
                f,
                M,
                self.max_episode,
                self.particle_num,
                self.ini_particle,
                self.c0,
                self.c1,
                self.c2,
                self.seed,
            )
        elif target == "HCRB":
            Main.QuanEstimation.mintime(
                Main.eval("Val{:" + method + "}()"),
                "HCRB_PSO_Copt",
                pso,
                f,
                self.max_episode,
                self.particle_num,
                self.ini_particle,
                self.c0,
                self.c1,
                self.c2,
                self.seed,
            )
        else:
            warnings.warn(
                "PSO is not available with the objective function {!r}.".format(target),
                DeprecationWarning,
            )
