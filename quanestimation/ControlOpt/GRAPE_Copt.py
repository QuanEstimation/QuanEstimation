import warnings
import quanestimation.ControlOpt.ControlStruct as Control
from quanestimation import QJL

class GRAPE_Copt(Control.ControlSystem):
    """
    Attributes
    ----------
    > **savefile:** `bool`
        -- Whether or not to save all the control coeffients.  
        If set `True` then the control coefficients and the values of the 
        objective function obtained in all episodes will be saved during 
        the training. If set `False` the control coefficients in the final 
        episode and the values of the objective function in all episodes 
        will be saved.

    > **Adam:** `bool`
        -- Whether or not to use Adam for updating control coefficients.

    > **ctrl0:** `list of arrays`
        -- Initial guesses of control coefficients.

    > **max_episode:** `int`
        -- The number of episodes.
  
    > **epsilon:** `float`
        -- Learning rate.

    > **beta1:** `float`
        -- The exponential decay rate for the first moment estimates.

    > **beta2:** `float`
        -- The exponential decay rate for the second moment estimates.

    > **eps:** `float`
        -- Machine epsilon.

    > **load:** `bool`
        -- Whether or not to load control coefficients in the current location.  
        If set `True` then the program will load control coefficients from 
        "controls.csv" file in the current location and use it as the initial 
        control coefficients.

    > **auto:** `bool`
        -- Whether or not to invoke automatic differentiation algorithm to evaluate  
        the gradient. If set `True` then the gradient will be calculated with 
        automatic differentiation algorithm otherwise it will be calculated 
        using analytical method.
    """
    
    def __init__(
        self,
        savefile=False,
        Adam=True,
        ctrl0=[],
        max_episode=300,
        epsilon=0.01,
        beta1=0.90,
        beta2=0.99,
        eps=1e-8,
        seed=1234,
        load=False,
        auto=True,
    ):

        Control.ControlSystem.__init__(self, savefile, ctrl0, eps, load)

        self.Adam = Adam
        self.max_episode = max_episode
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0
        self.seed = seed
        self.auto = auto

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

        if self.auto:
            if self.Adam:
                self.alg = QJL.autoGRAPE(
                    self.max_episode, self.epsilon, self.beta1, self.beta2
                )
            else:
                self.alg = QJL.autoGRAPE(self.max_episode, self.epsilon)
        else:
            if (len(self.tspan) - 1) != len(self.control_coefficients[0]):
                warnings.warn("GRAPE is not available when the length of each control is not \
                               equal to the length of time, and is replaced by auto-GRAPE.",
                               DeprecationWarning)
                #### call autoGRAPE automatically ####
                if self.Adam:
                    self.alg = QJL.autoGRAPE(
                        self.max_episode, self.epsilon, self.beta1, self.beta2
                    )
                else:
                    self.alg = QJL.autoGRAPE(self.max_episode, self.epsilon)
            else:
                if LDtype == "SLD":
                    if self.Adam:
                        self.alg = QJL.GRAPE(
                            self.max_episode, self.epsilon, self.beta1, self.beta2
                            )
                    else:
                        self.alg = QJL.GRAPE(self.max_episode, self.epsilon)
                else:
                    raise ValueError("GRAPE is only available when LDtype is SLD.")

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

        if self.auto:
            if self.Adam:
                self.alg = QJL.autoGRAPE(
                    self.max_episode, self.epsilon, self.beta1, self.beta2
                )
            else:
                self.alg = QJL.autoGRAPE(self.max_episode, self.epsilon)
        else:
            if (len(self.tspan) - 1) != len(self.control_coefficients[0]):
                warnings.warn("GRAPE is not available when the length of each control is not \
                               equal to the length of time, and is replaced by auto-GRAPE.",
                               DeprecationWarning)
                #### call autoGRAPE automatically ####
                if self.Adam:
                    self.alg = QJL.autoGRAPE(
                        self.max_episode, self.epsilon, self.beta1, self.beta2
                    )
                else:
                    self.alg = QJL.autoGRAPE(self.max_episode, self.epsilon)
            else:    
                if self.Adam:
                    self.alg = QJL.GRAPE(
                        self.max_episode, self.epsilon, self.beta1, self.beta2
                    )
                else:
                    self.alg = QJL.GRAPE(self.max_episode, self.epsilon)

        super().CFIM(M, W)

    def HCRB(self, W=[]):
        """
        GRAPE and auto-GRAPE are not available when the objective function is HCRB. 
        Supported methods are PSO, DE and DDPG.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.
        """
        raise ValueError(
            "GRAPE and auto-GRAPE are not available when the objective function is HCRB. Supported methods are 'PSO', 'DE' and 'DDPG'.",
        )

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

        if target == "HCRB":
            raise ValueError(
                "GRAPE and auto-GRAPE are not available when the objective function is HCRB. Supported methods are 'PSO', 'DE' and 'DDPG'.",
            )
        if self.auto:
            if self.Adam:
                self.alg = QJL.autoGRAPE(
                    self.max_episode, self.epsilon, self.beta1, self.beta2
                )
            else:
                self.alg = QJL.autoGRAPE(self.max_episode, self.epsilon)
        else:
            
            if self.Adam:
                self.alg = QJL.GRAPE(
                        self.max_episode, self.epsilon, self.beta1, self.beta2
                    )
            else:
                self.alg = QJL.GRAPE(self.max_episode, self.epsilon)

        super().mintime(f, W, M, method, target, LDtype)
