import numpy as np
import warnings
from julia import Main
import quanestimation.ControlOpt.ControlStruct as Control
from quanestimation.Common.common import SIC


class DE_Copt(Control.ControlSystem):
    def __init__(
        self,
        save_file=False,
        popsize=10,
        ctrl0=[],
        max_episode=1000,
        c=1.0,
        cr=0.5,
        seed=1234,
        load=False,
        eps=1e-8,
    ):

        Control.ControlSystem.__init__(self, save_file, ctrl0, load, eps)

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

        self.p_num = popsize
        self.max_episode = max_episode
        self.c = c
        self.cr = cr
        self.seed = seed

    def QFIM(self, W=[], dtype="SLD"):
        ini_population = ([self.ctrl0],)
        self.alg = Main.QuanEstimation.DE(
            self.max_episode,
            self.p_num,
            ini_population,
            self.c,
            self.cr,
            self.seed,
        )

        super().QFIM(W, dtype)

    def CFIM(self, M=[], W=[]):
        ini_population = ([self.ctrl0],)
        self.alg = Main.QuanEstimation.DE(
            self.max_episode,
            self.p_num,
            ini_population,
            self.c,
            self.cr,
            self.seed,
        )

        super().CFIM(M, W)

    def HCRB(self, W=[]):
        ini_population = ([self.ctrl0],)
        self.alg = Main.QuanEstimation.DE(
            self.max_episode,
            self.p_num,
            ini_population,
            self.c,
            self.cr,
            self.seed,
        )
        
        super().HCRB(W)

    def mintime(self, f, W=[], M=[], method="binary", target="QFIM", dtype="SLD"):
        ini_population = ([self.ctrl0],)
        self.alg = Main.QuanEstimation.DE(
            self.max_episode,
            self.popsize,
            ini_population,
            self.c,
            self.cr,
            self.seed,
        )

        super().mintime(f,W,M,method,target,dtype)