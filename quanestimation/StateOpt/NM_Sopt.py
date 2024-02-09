from quanestimation import QJL
import quanestimation.StateOpt.StateStruct as State


class NM_Sopt(State.StateSystem):
    """
    Attributes
    ----------
    > **savefile:**  `bool`
        -- Whether or not to save all the states.  
        If set `True` then the states and the values of the 
        objective function obtained in all episodes will be saved during 
        the training. If set `False` the state in the final 
        episode and the values of the objective function in all episodes 
        will be saved.

    > **p_num:** `int`
        -- The number of the input states.

    > **psi0:** `list of arrays`
        -- Initial guesses of states.

    > **max_episode:** `int`
        -- The number of episodes.
  
    > **ar:** `float`
        -- Reflection constant.

    > **ae:** `float`
        -- Expansion constant.

    > **ac:** `float`
        -- Constraction constant.

    > **as0:** `float`
        -- Shrink constant.

    > **seed:** `int`
        -- Random seed.

    > **eps:** `float`
        -- Machine epsilon.

    > **load:** `bool`
        -- Whether or not to load states in the current location.  
        If set `True` then the program will load state from "states.csv"
        file in the current location and use it as the initial state.
    """

    def __init__(
        self,
        savefile=False,
        p_num=10,
        psi0=[],
        max_episode=1000,
        ar=1.0,
        ae=2.0,
        ac=0.5,
        as0=0.5,
        seed=1234,
        eps=1e-8,
        load=False,
    ):

        State.StateSystem.__init__(self, savefile, psi0, seed, eps, load)

        self.p_num = p_num
        self.max_episode = max_episode
        self.ar = ar
        self.ae = ae
        self.ac = ac
        self.as0 = as0
        self.seed = seed

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
        ini_state = self.psi
        self.alg = QJL.NM(
            self.max_episode,
            self.p_num,
            ini_state,
            self.ar,
            self.ae,
            self.ac,
            self.as0,
        )

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
        ini_state = self.psi
        self.alg = QJL.NM(
            self.max_episode,
            self.p_num,
            ini_state,
            self.ar,
            self.ae,
            self.ac,
            self.as0,
        )

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
        ini_state = self.psi
        self.alg = QJL.NM(
            self.max_episode,
            self.p_num,
            ini_state,
            self.ar,
            self.ae,
            self.ac,
            self.as0,
        )

        super().HCRB(W)
