from julia import Main
import warnings
import numpy as np
import quanestimation.StateOpt.StateStruct as State
from quanestimation.Common.common import SIC

class DDPG_Sopt(State.StateSystem):
    def __init__(
        self,
        save_file=False,
        max_episode=500,
        layer_num=3,
        layer_dim=200,
        seed=1234,
        psi0=[],
        load=False,
        eps=1e-8):

        State.StateSystem.__init__(self, save_file, psi0, seed, load, eps)
        """
        ----------
        Inputs
        ----------
        layer_num:
            --description: the number of layers (including the input and output layer).
            --type: int

        layer_dim:
            --description: the number ofP neurons in the hidden layer.
            --type: int
        
        seed:
            --description: random seed.
            --type: int
        """
        self.max_episode = max_episode
        self.layer_num = layer_num
        self.layer_dim = layer_dim
        
        self.seed = seed
 
        self.alg = Main.QuanEstimation.DDPG(
            self.max_episode,
            self.layer_num,
            self.layer_dim,
            self.seed
        ) 

    def QFIM(self, W=[], dtype="SLD"):
        super().QFIM(W, dtype)

    def CFIM(self, M=[], W=[]):
        super().CFIM(M, W)

    def HCRB(self, W=[]):
        super().HCRB(W)