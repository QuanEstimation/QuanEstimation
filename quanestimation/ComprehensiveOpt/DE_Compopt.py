import numpy as np
from julia import Main
import warnings
import quanestimation.ComprehensiveOpt.ComprehensiveStruct as Comp
from quanestimation.Common.common import SIC

class DE_Compopt(Comp.ComprehensiveSystem):
    def __init__(
        self,
        save_file=False,
        popsize=10,
        psi0=[],
        ctrl0=[],
        measurement0=[],
        max_episode=1000,
        c=1.0,
        cr=0.5,
        seed=1234,
        eps=1e-8):
        
        Comp.ComprehensiveSystem.__init__(self, psi0, ctrl0, measurement0, save_file, seed, eps)

        """
        --------
        inputs
        --------
        save_file:
            --description: True: save the states (or controls, measurements) and the value of the 
                                 target function for each episode.
                           False: save the states (or controls, measurements) and all the value 
                                   of the target function for the last episode.
            --type: bool 
            
        popsize:
           --description: the number of populations.
           --type: int
        
        psi0:
           --description: initial guesses of states (kets).
           --type: array
           
        ctrl0:
            --description: initial control coefficients.
            --type: list (of vector)
            
        measurement0:
           --description: a set of POVMs.
           --type: list (of vector)

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

    def SC(self, W=[], M=[], target="QFIM", dtype="SLD"):
        ini_population = (self.psi0, self.ctrl0)
        self.alg = Main.QuanEstimation.DE(
            self.max_episode,
            self.p_num,
            ini_population,
            self.c,
            self.cr,
            self.seed,
        )

        super().SC(W,M,target,dtype)


    def CM(self, rho0, W=[]):
        ini_population = (self.ctrl0, self.measurement0)
        self.alg = Main.QuanEstimation.DE(
            self.max_episode,
            self.p_num,
            ini_population,
            self.c,
            self.cr,
            self.seed,
        )

        super().CM(rho0, W)
        

    def SM(self, W=[]):
        ini_population = (self.psi0, self.measurement0)
        self.alg = Main.QuanEstimation.DE(
            self.max_episode,
            self.p_num,
            ini_population,
            self.c,
            self.cr,
            self.seed,
        )

        super().SM(W)

    def SCM(self, W=[]):
        ini_population = (self.psi0, self.ctrl0, self.measurement0)
        self.alg = Main.QuanEstimation.DE(
            self.max_episode,
            self.p_num,
            ini_population,
            self.c,
            self.cr,
            self.seed,
        )

        super().SCM(W)
