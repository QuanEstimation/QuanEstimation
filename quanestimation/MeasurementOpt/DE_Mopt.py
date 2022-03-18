import numpy as np
from julia import Main
import quanestimation.MeasurementOpt.MeasurementStruct as Measurement


class DE_Mopt(Measurement.MeasurementSystem):
    def __init__(
        self,
        mtype,
        minput,
        savefile=False,
        popsize=10,
        measurement0=[],
        max_episode=1000,
        c=1.0,
        cr=0.5,
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
        popsize:
           --description: the number of populations.
           --type: int
        
        measurement0:
           --description: initial guesses of measurements.
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
        self.p_num = popsize
        self.max_episode = max_episode
        self.c = c
        self.cr = cr

    def CFIM(self, W=[]):

        self.alg = Main.QuanEstimation.DE(
            self.max_episode,
            self.p_num,
            self.measurement0,
            self.c,
            self.cr,
            self.seed,
        )
        super().CFIM(W)
