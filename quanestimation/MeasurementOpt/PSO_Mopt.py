import numpy as np
from julia import Main
import quanestimation.MeasurementOpt.MeasurementStruct as Measurement


class PSO_Mopt(Measurement.MeasurementSystem):
    def __init__(
        self,
        mtype,
        minput,
        save_file=False,
        particle_num=10,
        measurement0=[],
        max_episode=[1000, 100],
        c0=1.0,
        c1=2.0,
        c2=2.0,
        seed=1234,
        load=False,
        eps=1e-8):

        Measurement.MeasurementSystem.__init__(
            self, mtype, minput, save_file, measurement0, seed, load, eps)

        """
        --------
        inputs
        --------
        particle_num:
           --description: the number of particles.
           --type: int
        
        measurement0:
           --description: initial guesses of measurements.
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
        self.particle_num = particle_num
        self.max_episode = max_episode
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

    def CFIM(self, W=[]):
        """
        Description: use particle swarm optimization algorithm to update the measurements that maximize the
                     CFI (1/Tr(WF^{-1} with F the CFIM).

        ---------
        Inputs
        ---------
        W:
            --description: weight matrix.
            --type: matrix
        """
        if self.mtype == "projection":
            if self.dynamics_type == "dynamics":
                if W == []:
                    W = np.eye(len(self.Hamiltonian_derivative))
                self.W = W

                pso = Main.QuanEstimation.projection_Mopt(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.rho0,
                    self.tspan,
                    self.decay_opt,
                    self.gamma,
                    self.M,
                    self.W,
                    self.eps)
                Main.QuanEstimation.CFIM_PSO_Mopt(
                    pso,
                    self.max_episode,
                    self.particle_num,
                    self.measurement0,
                    self.c0,
                    self.c1,
                    self.c2,
                    self.seed,
                    self.save_file)
            elif self.dynamics_type == "kraus":
                if W == []:
                    W = np.eye(len(self.dK))
                self.W = W

                pso = Main.QuanEstimation.projection_Mopt_Kraus(
                    self.K,
                    self.dK,
                    self.rho0,
                    self.M,
                    self.W,
                    self.eps)
                Main.QuanEstimation.CFIM_PSO_Mopt(
                    pso,
                    self.max_episode,
                    self.particle_num,
                    self.measurement0,
                    self.c0,
                    self.c1,
                    self.c2,
                    self.seed,
                    self.save_file)
            self.load_save()
        elif self.mtype == "input":
            if self.dynamics_type == "dynamics":
                if W == []:
                    W = np.eye(len(self.Hamiltonian_derivative))
                self.W = W

                pso = Main.QuanEstimation.LinearComb_Mopt(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.rho0,
                    self.tspan,
                    self.decay_opt,
                    self.gamma,
                    self.povm_basis,
                    self.M_num,
                    self.W,
                    self.eps)
                Main.QuanEstimation.CFIM_PSO_Mopt(
                    pso,
                    self.max_episode,
                    self.particle_num,
                    self.c0,
                    self.c1,
                    self.c2,
                    self.seed,
                    self.save_file)
            elif self.dynamics_type == "kraus":
                if W == []:
                    W = np.eye(len(self.dK))
                self.W = W

                pso = Main.QuanEstimation.LinearComb_Mopt_Kraus(
                    self.K,
                    self.dK,
                    self.rho0,
                    self.povm_basis,
                    self.M_num,
                    self.W,
                    self.eps)
                Main.QuanEstimation.CFIM_PSO_Mopt(
                    pso,
                    self.max_episode,
                    self.particle_num,
                    self.c0,
                    self.c1,
                    self.c2,
                    self.seed,
                    self.save_file)
            self.load_save()

        elif self.mtype == "rotation":
            if self.dynamics_type == "dynamics":
                if W == []:
                    W = np.eye(len(self.Hamiltonian_derivative))
                self.W = W

                pso = Main.QuanEstimation.RotateBasis_Mopt(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.rho0,
                    self.tspan,
                    self.decay_opt,
                    self.gamma,
                    self.povm_basis,
                    self.W,
                    self.eps)
                Main.QuanEstimation.CFIM_PSO_Mopt(
                    pso,
                    self.max_episode,
                    self.particle_num,
                    self.c0,
                    self.c1,
                    self.c2,
                    self.seed,
                    self.save_file)
            elif self.dynamics_type == "kraus":
                if W == []:
                    W = np.eye(len(self.dK))
                self.W = W

                pso = Main.QuanEstimation.RotateBasis_Mopt_Kraus(
                    self.K,
                    self.dK,
                    self.rho0,
                    self.povm_basis,
                    self.W,
                    self.eps)
                Main.QuanEstimation.CFIM_PSO_Mopt(
                    pso,
                    self.max_episode,
                    self.particle_num,
                    self.c0,
                    self.c1,
                    self.c2,
                    self.seed,
                    self.save_file)
            self.load_save()
        else:
            raise ValueError("{!r} is not a valid value for method, supported values are \
                             'projection' and 'input'.".format(self.mtype))
