from quanestimation import QJL
import quanestimation.StateOpt.StateStruct as State


class RI_Sopt(State.StateSystem):
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

    > **psi0:** `list of arrays`
        -- Initial guesses of states.

    > **max_episode:** `int`
        -- The number of episodes.

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
        psi0=[],
        max_episode=300,
        seed=1234,
        eps=1e-8,
        load=False,
    ):

        State.StateSystem.__init__(self, savefile, psi0, seed, eps, load)

        self.max_episode = max_episode
        self.seed = seed

    def QFIM(self, W=[], LDtype="SLD"):
        r"""
        Choose QFI as the objective function. 

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.

        > **LDtype:** `string`
            -- Types of QFI (QFIM) can be set as the objective function. Only SLD can
            is available here.
        """
        self.alg = QJL.RI(
            self.max_episode,
        )
        if self.dynamics_type != "Kraus":
            raise ValueError("Only the parameterization with Kraus operators is available.")
        
        if LDtype == "SLD":
            super().QFIM(W, LDtype)
        else:
            raise ValueError("Only SLD is available.")

    def CFIM(self, M=[], W=[]):
        """
        Choose CFIM as the objective function. 

        **Note:** CFIM is not available.

        Parameters
        ----------
        > **M:** `list`
            -- POVM.
            
        > **W:** `matrix`
            -- Weight matrix.
        """
        raise ValueError("CFIM is not available.")

    def HCRB(self, W=[]):
        """
        Choose HCRB as the objective function. 

        **Note:** Here HCRB is not available.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.
        """
        raise ValueError("HCRB is not available.")
