import numpy as np
import warnings
from julia import Main
import quanestimation.ControlOpt.ControlStruct as Control
from quanestimation.Common.common import SIC


class DDPG_Copt(Control.ControlSystem):
    def __init__(
        self,
        save_file=False,
        max_episode=500,
        layer_num=3,
        layer_dim=200,
        seed=1234,
        ctrl0=[],
        load=False,
        eps=1e-8,
    ):

        Control.ControlSystem.__init__(self, save_file, ctrl0, load, eps)

        """                                           
        ----------
        Inputs
        ----------
        max_episode:
            --description: max number of the training episodes.
            --type: int

        layer_num:
            --description: the number of layers (including the input and output layer).
            --type: int

        layer_dim:
            --description: the number of neurons in the hidden layer.
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
            self.max_episode, self.layer_num, self.layer_dim, self.seed
        )

    def QFIM(self, W=[], dtype="SLD"):
        super().QFIM(W, dtype)

    def CFIM(self, M=[], W=[]):
        super().CFIM(M, W)

    def HCRB(self, W=[]):
        super().HCRB(W)

    def mintime(self, f, W=[], M=[], method="binary", target="QFIM", dtype="SLD"):
        super().mintime(f, W, M, method, target, dtype)
