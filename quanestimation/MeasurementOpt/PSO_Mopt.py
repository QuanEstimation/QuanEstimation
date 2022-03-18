import numpy as np
from julia import Main
import quanestimation.MeasurementOpt.MeasurementStruct as Measurement

class PSO_Mopt(Measurement.MeasurementSystem):
    def __init__(
        self,
        mtype,
        minput,
        savefile=False,
        particle_num=10,
        measurement0=[],
        max_episode=[1000, 100],
        c0=1.0,
        c1=2.0,
        c2=2.0,
        seed=1234,
        load=False,
        eps=1e-8,
    ):

        Measurement.MeasurementSystem.__init__(
            self, mtype, minput, savefile, measurement0, seed, load, eps
        )

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
        self.p_num = particle_num
        self.max_episode = max_episode
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

    def CFIM(self, W=[]):

        self.alg = Main.QuanEstimation.PSO(
            self.max_episode,
            self.p_num,
            self.measurement0,
            self.c0,
            self.c1,
            self.c2,
            self.seed,
        )

        super().CFIM(W)
