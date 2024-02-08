from quanestimation import QJL
import quanestimation.ComprehensiveOpt.ComprehensiveStruct as Comp


class AD_Compopt(Comp.ComprehensiveSystem):
    """
    Attributes
    ----------
    > **savefile:** `bool`
        -- Whether or not to save all the optimized variables (probe states, 
        control coefficients and measurements).  
        If set `True` then the optimized variables and the values of the 
        objective function obtained in all episodes will be saved during 
        the training. If set `False` the optimized variables in the final 
        episode and the values of the objective function in all episodes 
        will be saved.

    > **Adam:** `bool`
        -- Whether or not to use Adam for updating.

    > **psi0:** `list of arrays`
        -- Initial guesses of states.

    > **ctrl0:** `list of arrays`
        -- Initial guesses of control coefficients.

    > **measurement0:** `list of arrays`
        -- Initial guesses of measurements.

    > **max_episode:** `int`
        -- The number of episodes.
  
    > **epsilon:** `float`
        -- Learning rate.

    > **beta1:** `float`
        -- The exponential decay rate for the first moment estimates.

    > **beta2:** `float`
        -- The exponential decay rate for the second moment estimates.

    > **seed:** `int`
        -- Random seed.

    > **eps:** `float`
        -- Machine epsilon.
    """
    def __init__(
        self,
        savefile=False,
        Adam=False,
        psi0=[],
        ctrl0=[],
        measurement0=[],
        max_episode=300,
        epsilon=0.01,
        beta1=0.90,
        beta2=0.99,
        seed=1234,
        eps=1e-8,
    ):

        Comp.ComprehensiveSystem.__init__(
            self, savefile, psi0, ctrl0, measurement0, seed, eps
        )

        self.Adam = Adam
        self.max_episode = max_episode
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0
        self.seed = seed

    def SC(self, W=[], M=[], target="QFIM", LDtype="SLD"):
        """
        Comprehensive optimization of the probe state and control (SC).

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.

        > **M:** `list of matrices`
            -- A set of positive operator-valued measure (POVM). The default measurement 
            is a set of rank-one symmetric informationally complete POVM (SIC-POVM).

        > **target:** `string`
            -- Objective functions for searching the minimum time to reach the given 
            value of the objective function. Options are:  
            "QFIM" (default) -- choose QFI (QFIM) as the objective function.  
            "CFIM" -- choose CFI (CFIM) as the objective function.  
            "HCRB" -- choose HCRB as the objective function.  

        > **LDtype:** `string`
            -- Types of QFI (QFIM) can be set as the objective function. Options are:  
            "SLD" (default) -- QFI (QFIM) based on symmetric logarithmic derivative (SLD).  
            "RLD" -- QFI (QFIM) based on right logarithmic derivative (RLD).  
            "LLD" -- QFI (QFIM) based on left logarithmic derivative (LLD). 
        
        **Note:** AD is only available when target is 'QFIM'.
        """
        if M != []:
            raise ValueError(
                "AD is not available when target is 'CFIM'. Supported methods are 'PSO' and 'DE'.",
            )
        elif target == "HCRB":
            raise ValueError(
                "AD is not available when the target function is HCRB. Supported methods are 'PSO' and 'DE'.",
            )

        if self.Adam:
            self.alg = QJL.QuanEstimation.AD(
                self.max_episode, self.epsilon, self.beta1, self.beta2
            )
        else:
            self.alg = QJL.QuanEstimation.AD(self.max_episode, self.epsilon)

        super().SC(W, M, target, LDtype)
