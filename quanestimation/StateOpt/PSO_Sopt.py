from julia import Main
import warnings
import quanestimation.StateOpt.StateStruct as State
import numpy as np
from quanestimation.Common.common import SIC

class PSO_Sopt(State.StateSystem):
    def __init__(
        self,
        save_file=False,
        particle_num=10,
        psi0=[],
        max_episode=[1000, 100],
        c0=1.0,
        c1=2.0,
        c2=2.0,
        seed=1234,
        load=False,
        eps=eps):

        State.StateSystem.__init__(self, save_file, psi0, seed, load, eps)

        """
        --------
        inputs
        --------
        particle_num:
           --description: the number of particles.
           --type: int

        psi0:
           --description: initial guesses of states (kets).
           --type: array

        max_episode:
            --description: max number of the training episodes.
            --type: int

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
        self.v0 = 0.1
        self.seed = seed

    def QFIM(self, W=[], dtype="SLD"):
        """
        Description: use particle swarm optimizaiton algorithm to search the optimal initial state that maximize the
                     QFI (1/Tr(WF^{-1} with F the QFIM).

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

            if any(self.gamma):
                pso = Main.QuanEstimation.TimeIndepend_noise(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.decay_opt,
                    self.gamma,
                    self.W,
                    self.eps)
                if dtype == "SLD":
                    Main.QuanEstimation.QFIM_PSO_Sopt(
                        pso,
                        self.max_episode,
                        self.particle_num,
                        self.ini_particle,
                        self.c0,
                        self.c1,
                        self.c2,
                        self.v0,
                        self.seed,
                        self.save_file)
                elif dtype == "RLD":
                    pass #### to be done
                elif dtype == "LLD":
                    pass #### to be done
                else:
                    raise ValueError("{!r} is not a valid value for dtype, supported \
                              values are 'SLD', 'RLD' and 'LLD'.".format(dtype))
            else:
                pso = Main.QuanEstimation.TimeIndepend_noiseless(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.W,
                    self.eps)
                if dtype == "SLD":
                    Main.QuanEstimation.QFIM_PSO_Sopt(
                        pso,
                        self.max_episode,
                        self.particle_num,
                        self.ini_particle,
                        self.c0,
                        self.c1,
                        self.c2,
                        self.v0,
                        self.seed,
                        self.save_file)
                elif dtype == "RLD":
                    pass #### to be done
                elif dtype == "LLD":
                    pass #### to be done
                else:
                    raise ValueError("{!r} is not a valid value for dtype, supported \
                              values are 'SLD', 'RLD' and 'LLD'.".format(dtype))

        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W

            pso = Main.QuanEstimation.TimeIndepend_Kraus(self.K, self.dK, self.psi0, self.W, self.eps)
            if dtype == "SLD":
                Main.QuanEstimation.QFIM_PSO_Sopt(
                    pso,
                    self.max_episode,
                    self.particle_num,
                    self.ini_particle,
                    self.c0,
                    self.c1,
                    self.c2,
                    self.v0,
                    self.seed,
                    self.save_file)
            elif dtype == "RLD":
                pass #### to be done
            elif dtype == "LLD":
                pass #### to be done
            else:
                raise ValueError("{!r} is not a valid value for dtype, supported \
                                  values are 'SLD', 'RLD' and 'LLD'.".format(dtype))

        self.load_save()

    def CFIM(self, M=[], W=[]):
        """
        Description: use particle swarm optimizaiton algorithm to search the optimal initial state that maximize the
                     CFI (1/Tr(WF^{-1} with F the CFIM).

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
        if M==[]:
            M = SIC(len(self.psi0))
        M = [np.array(x, dtype=np.complex128) for x in M]
        
        if self.dynamics_type == "dynamics":
            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W

            if any(self.gamma):
                pso = Main.QuanEstimation.TimeIndepend_noise(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.decay_opt,
                    self.gamma,
                    self.W,
                    self.eps)
                Main.QuanEstimation.CFIM_PSO_Sopt(
                    M,
                    pso,
                    self.max_episode,
                    self.particle_num,
                    self.ini_particle,
                    self.c0,
                    self.c1,
                    self.c2,
                    self.v0,
                    self.seed,
                    self.save_file)
            else:
                pso = Main.QuanEstimation.TimeIndepend_noiseless(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.W,
                    self.eps)
                Main.QuanEstimation.CFIM_PSO_Sopt(
                    M,
                    pso,
                    self.max_episode,
                    self.particle_num,
                    self.ini_particle,
                    self.c0,
                    self.c1,
                    self.c2,
                    self.v0,
                    self.seed,
                    self.save_file)

        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W
            
            pso = Main.QuanEstimation.TimeIndepend_Kraus(self.K, self.dK, self.psi0, self.W, self.eps)
            Main.QuanEstimation.CFIM_PSO_Sopt(
                M,
                pso,
                self.max_episode,
                self.particle_num,
                self.ini_particle,
                self.c0,
                self.c1,
                self.c2,
                self.v0,
                self.seed,
                self.save_file)

        self.load_save()

    def HCRB(self, W=[]):
        """
        Description: use particle swarm optimizaiton algorithm to search the optimal initial state that maximize the HCRB.

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

            if len(self.Hamiltonian_derivative) == 1:
                warnings.warn("In single parameter scenario, HCRB is equivalent to QFI. Please \
                               choose QFIM as the target function for control optimization",\
                               DeprecationWarning)
            else:
                if any(self.gamma):
                    pso = Main.QuanEstimation.TimeIndepend_noise(
                        self.freeHamiltonian,
                        self.Hamiltonian_derivative,
                        self.psi0,
                        self.tspan,
                        self.decay_opt,
                        self.gamma,
                        self.W,
                        self.eps)
                    Main.QuanEstimation.HCRB_PSO_Sopt(
                        pso,
                        self.max_episode,
                        self.particle_num,
                        self.ini_particle,
                        self.c0,
                        self.c1,
                        self.c2,
                        self.v0,
                        self.seed,
                        self.save_file)
                else:
                    pso = Main.QuanEstimation.TimeIndepend_noiseless(
                        self.freeHamiltonian,
                        self.Hamiltonian_derivative,
                        self.psi0,
                        self.tspan,
                        self.W,
                        self.eps)
                    Main.QuanEstimation.HCRB_PSO_Sopt(
                        pso,
                        self.max_episode,
                        self.particle_num,
                        self.ini_particle,
                        self.c0,
                        self.c1,
                        self.c2,
                        self.v0,
                        self.seed,
                        self.save_file)

        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W

            pso = Main.QuanEstimation.TimeIndepend_Kraus(self.K, self.dK, self.psi0, self.W, self.eps)
            Main.QuanEstimation.HCRB_PSO_Sopt(
                pso,
                self.max_episode,
                self.particle_num,
                self.ini_particle,
                self.c0,
                self.c1,
                self.c2,
                self.v0,
                self.seed,
                self.save_file)

        self.load_save()
