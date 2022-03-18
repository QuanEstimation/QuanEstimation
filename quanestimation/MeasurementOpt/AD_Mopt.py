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

        if self.Adam:
            self.alg = Main.QuanEstimation.AD(
                self.max_episode, self.epsilon, self.beta1, self.beta2
            )
        else:
            self.alg = Main.QuanEstimation.AD(self.max_episode, self.epsilon)

    def CFIM(self, W=[]):
        if self.mtype == "projection":
            warnings.warn(
                "AD is not available when mtype is projection. Supported methods are \
                           'PSO' and 'DE'.",
                DeprecationWarning,
            )
        else:
            super().CFIM(W)
