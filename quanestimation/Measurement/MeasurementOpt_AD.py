import numpy as np
from julia import Main
import quanestimation.Measurement.Measurement_struct as Measurement_struct

class AD_Meas(Measurement_struct.MeasurementSystem):
    def __init__(self, tspan, rho0, H0, dH=[], decay=[], W=[], ini_measurement=[], \
                 Adam=True, max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99):

        Measurement_struct.MeasurementSystem.__init__(self, tspan, rho0, H0, dH, decay, W, ini_Measurement, seed=1234, accuracy=1e-8)
        
        """
        --------
        inputs
        --------
        Adam:
            --description: whether to use Adam to update the controls.
            --type: bool (True or False)

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

        accuracy:
            --description: calculation accuracy.
            --type: float
        
        """
        self.Adam = Adam
        self.max_episode = max_episode
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0 
        
    def CFIM(self, save_file=False):
        """
        Description: use particle autodifferential algorithm to update the measurements that maximize the 
                     CFI (1/Tr(WF^{-1} with F the CFIM).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the measurements for each episode but overwrite in the next episode and all the CFI (Tr(WF^{-1})).
                           False: save the measurements for the last episode and all the CFI (Tr(WF^{-1})).
            --type: bool
        """
        ad = Main.QuanEstimation.MeasurementOpt(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho0, self.tspan,\
                                                self.decay_opt, self.gamma, self.Measurement, self.W, self.accuracy)
        Main.QuanEstimation.AD_CFIM(ad, self.mt, self.vt, self.epsilon, self.beta1, self.beta2, self.max_episode, self.Adam, save_file)
        self.load_save()
        