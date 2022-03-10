import numpy as np
from julia import Main
import warnings
import quanestimation.MeasurementOpt.MeasurementStruct as Measurement


class AD_Mopt(Measurement.MeasurementSystem):
    def __init__(
        self,
        mtype,
        minput,
        save_file=False,
        Adam=False,
        measurement0=[],
        max_episode=300,
        epsilon=0.01,
        beta1=0.90,
        beta2=0.99,
        seed=1234,
        load=False,
        eps=1e-8,
    ):

        Measurement.MeasurementSystem.__init__(
            self, mtype, minput, save_file, measurement0, seed, load, eps
        )

        """
        --------
        inputs
        --------
        Adam:
            --description: whether to use Adam to update the controls.
            --type: bool (True or False)
            
        measurement0:
           --description: initial guess of measurements.
           --type: array

        max_episode:
            --description: max number of the training episodes.
            --type: int

        epsilon:
            --description: learning rate.
            --type: float

        beta1:
            --description: the exponential decay rate for the first moment estimates .
            --type: float

        beta2:
            --description: the exponential decay rate for the second moment estimates .
            --type: float

        eps:
            --description: calculation eps.
            --type: float
        
        """
        self.Adam = Adam
        self.max_episode = max_episode
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0
        self.seed = seed
        self.update_basis = 50

    def CFIM(self, W=[]):
        """
        Description: use particle autodifferential algorithm to update the measurements that maximize the
                     CFI (1/Tr(WF^{-1} with F the CFIM).

        ---------
        Inputs
        ---------
        W:
            --description: weight matrix.
            --type: matrix
        """

        if self.mtype == "projection":
            warnings.warn(
                "AD is not available when mtype is projection. Supported methods are \
                           'PSO' and 'DE'.",
                DeprecationWarning,
            )
        elif self.mtype == "input":
            if self.dynamics_type == "dynamics":
                if W == []:
                    W = np.eye(len(self.Hamiltonian_derivative))
                self.W = W

                AD = Main.QuanEstimation.LinearComb_Mopt(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.rho0,
                    self.tspan,
                    self.decay_opt,
                    self.gamma,
                    self.povm_basis,
                    self.M_num,
                    self.W,
                    self.eps,
                )
                Main.QuanEstimation.CFIM_AD_Mopt(
                    AD,
                    self.mt,
                    self.vt,
                    self.epsilon,
                    self.beta1,
                    self.beta2,
                    self.max_episode,
                    self.Adam,
                    self.save_file,
                    self.seed,
                )
            elif self.dynamics_type == "kraus":
                if W == []:
                    W = np.eye(len(self.dK))
                self.W = W

                AD = Main.QuanEstimation.LinearComb_Mopt_Kraus(
                    self.K,
                    self.dK,
                    self.rho0,
                    self.povm_basis,
                    self.M_num,
                    self.W,
                    self.eps,
                )
                Main.QuanEstimation.CFIM_AD_Mopt(
                    AD,
                    self.mt,
                    self.vt,
                    self.epsilon,
                    self.beta1,
                    self.beta2,
                    self.max_episode,
                    self.Adam,
                    self.save_file,
                    self.seed,
                )
            self.load_save()
        elif self.mtype == "rotation":
            if self.dynamics_type == "dynamics":
                if W == []:
                    W = np.eye(len(self.Hamiltonian_derivative))
                self.W = W

                AD = Main.QuanEstimation.RotateBasis_Mopt(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.rho0,
                    self.tspan,
                    self.decay_opt,
                    self.gamma,
                    self.povm_basis,
                    self.W,
                    self.eps,
                )
                Main.QuanEstimation.CFIM_AD_Mopt(
                    AD,
                    self.mt,
                    self.vt,
                    self.epsilon,
                    self.beta1,
                    self.beta2,
                    self.max_episode,
                    self.Adam,
                    self.save_file,
                    self.seed,
                )
            elif self.dynamics_type == "kraus":
                if W == []:
                    W = np.eye(len(self.dK))
                self.W = W

                AD = Main.QuanEstimation.RotateBasis_Mopt_Kraus(
                    self.K, self.dK, self.rho0, self.povm_basis, self.W, self.eps
                )
                Main.QuanEstimation.CFIM_AD_Mopt(
                    AD,
                    self.mt,
                    self.vt,
                    self.epsilon,
                    self.beta1,
                    self.beta2,
                    self.max_episode,
                    self.Adam,
                    self.save_file,
                    self.seed,
                )
            self.load_save()
        else:
            raise ValueError(
                "{!r} is not a valid value for method, supported values are \
                             'projection' and 'input'.".format(
                    self.mtype
                )
            )
