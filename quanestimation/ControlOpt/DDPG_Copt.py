from quanestimation import QJL
import quanestimation.ControlOpt.ControlStruct as Control


class DDPG_Copt(Control.ControlSystem):
    """
    Attributes
    ----------
    > **savefile:**  `bool`
        -- Whether or not to save all the control coeffients.  
        If set `True` then the control coefficients and the values of the 
        objective function obtained in all episodes will be saved during 
        the training. If set `False` the control coefficients in the final 
        episode and the values of the objective function in all episodes 
        will be saved.

    > **ctrl0:** `list of arrays`
        -- Initial guesses of control coefficients.

    > **max_episode:** `int`
        -- The number of episodes.
  
    > **layer_num:** `int`
        -- The number of layers (include the input and output layer).

    > **layer_dim:** `int`
        -- The number of neurons in the hidden layer.

    > **seed:** `int`
        -- Random seed.

    > **eps:** `float`
        -- Machine epsilon.

    > **load:** `bool`
        -- Whether or not to load control coefficients in the current location.  
        If set `True` then the program will load control coefficients from 
        "controls.csv" file in the current location and use it as the initial 
        control coefficients.
    """

    def __init__(
        self,
        savefile=False,
        ctrl0=[],
        max_episode=500,
        layer_num=3,
        layer_dim=200,
        seed=1234,
        eps=1e-8,
        load=False,
    ):

        Control.ControlSystem.__init__(self, savefile, ctrl0, eps, load)

        self.max_episode = max_episode
        self.layer_num = layer_num
        self.layer_dim = layer_dim

        self.seed = seed

        self.alg = QJL.DDPG(
            self.max_episode, self.layer_num, self.layer_dim, self.seed
        )

    def QFIM(self, W=[], LDtype="SLD"):
        r"""
        Choose QFI or $\mathrm{Tr}(WF^{-1})$ as the objective function. 
        In single parameter estimation the objective function is QFI and in 
        multiparameter estimation it will be $\mathrm{Tr}(WF^{-1})$.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.

        > **LDtype:** `string`
            -- Types of QFI (QFIM) can be set as the objective function. Options are:  
            "SLD" (default) -- QFI (QFIM) based on symmetric logarithmic derivative (SLD).  
            "RLD" -- QFI (QFIM) based on right logarithmic derivative (RLD).  
            "LLD" -- QFI (QFIM) based on left logarithmic derivative (LLD).
        """

        super().QFIM(W, LDtype)

    def CFIM(self, M=[], W=[]):
        r"""
        Choose CFI or $\mathrm{Tr}(WI^{-1})$ as the objective function. 
        In single parameter estimation the objective function is CFI and 
        in multiparameter estimation it will be $\mathrm{Tr}(WI^{-1})$.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.

        > **M:** `list of matrices`
            -- A set of positive operator-valued measure (POVM). The default measurement 
            is a set of rank-one symmetric informationally complete POVM (SIC-POVM).

        **Note:** 
            SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
            which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/
            solutions.html).
        """

        super().CFIM(M, W)

    def HCRB(self, W=[]):
        """
        Choose HCRB as the objective function. 

        **Note:** in single parameter estimation, HCRB is equivalent to QFI, please choose 
        QFI as the objective function.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.
        """
        super().HCRB(W)

    def mintime(self, f, W=[], M=[], method="binary", target="QFIM", LDtype="SLD"):
        """
        Search of the minimum time to reach a given value of the objective function.

        Parameters
        ----------
        > **f:** `float`
            -- The given value of the objective function.

        > **W:** `matrix`
            -- Weight matrix.

        > **M:** `list of matrices`
            -- A set of positive operator-valued measure (POVM). The default measurement 
            is a set of rank-one symmetric informationally complete POVM (SIC-POVM).

        > **method:** `string`
            -- Methods for searching the minimum time to reach the given value of the 
            objective function. Options are:  
            "binary" (default) -- Binary search (logarithmic search).  
            "forward" -- Forward search from the beginning of time.

        > **target:** `string`
            -- Objective functions for searching the minimum time to reach the given 
            value of the objective function. Options are:  
            "QFIM" (default) -- Choose QFI (QFIM) as the objective function.  
            "CFIM" -- Choose CFI (CFIM) as the objective function.  
            "HCRB" -- Choose HCRB as the objective function.

        > **LDtype:** `string`
            -- Types of QFI (QFIM) can be set as the objective function. Options are:  
            "SLD" (default) -- QFI (QFIM) based on symmetric logarithmic derivative (SLD).  
            "RLD" -- QFI (QFIM) based on right logarithmic derivative (RLD).  
            "LLD" -- QFI (QFIM) based on left logarithmic derivative (LLD).

        **Note:** 
            SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
            which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/
            solutions.html).
        """

        super().mintime(f, W, M, method, target, LDtype)
