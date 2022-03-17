import numpy as np
from julia import Main
import quanestimation.MeasurementOpt.MeasurementStruct as Measurement


class DE_Mopt(Measurement.MeasurementSystem):
    def __init__(
        self,
        mtype,
        minput,
        save_file=False,
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
            self, mtype, minput, save_file, measurement0, seed, load, eps
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
        self.popsize = popsize
        self.max_episode = max_episode
        self.c = c
        self.cr = cr

    def CFIM(self, W=[]):
        if self.mtype == "projection":
            ini_population = Main.vec(self.measurement0)
        elif self.mtype == "input":
            if self.minput[0] == "LC":
                ini_population = Main.vec(self.povm_basis)
            elif self.minput[0] == "rotation":
                ini_population = Main.vec(self.povm_basis)

        self.alg = Main.QuanEstimation.DE(
            self.max_episode,
            self.popsize,
            ini_population,
            self.c,
            self.cr,
            self.seed,
        )
        super().CFIM(W)
