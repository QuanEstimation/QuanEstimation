from quanestimation import QJL
import quanestimation.ComprehensiveOpt.ComprehensiveStruct as Comp


class DE_Compopt(Comp.ComprehensiveSystem):
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

    > **p_num:** `int`
        -- The number of populations.

    > **psi0:** `list of arrays`
        -- Initial guesses of states.

    > **ctrl0:** `list of arrays`
        -- Initial guesses of control coefficients.

    > **measurement0:** `list of arrays`
        -- Initial guesses of measurements.

    > **max_episode:** `int`
        -- The number of episodes.
  
    > **c:** `float`
        -- Mutation constant.

    > **cr:** `float`
        -- Crossover constant.

    > **seed:** `int`
        -- Random seed.

    > **eps:** `float`
        -- Machine epsilon.
    """
    def __init__(
        self,
        savefile=False,
        p_num=10,
        psi0=[],
        ctrl0=[],
        measurement0=[],
        max_episode=1000,
        c=1.0,
        cr=0.5,
        seed=1234,
        eps=1e-8,
    ):

        Comp.ComprehensiveSystem.__init__(
            self, savefile, psi0, ctrl0, measurement0, seed, eps
        )

        self.p_num = p_num
        self.max_episode = max_episode
        self.c = c
        self.cr = cr
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

        **Note:** 
            SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
            which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/
            solutions.html).
        """
        ini_population = (self.psi, self.ctrl0)
        self.alg = QJL.DE(
            self.max_episode,
            self.p_num,
            ini_population,
            self.c,
            self.cr,
        )
        
        super().SC(W, M, target, LDtype)

    def CM(self, rho0, W=[]):
        """
        Comprehensive optimization of the control and measurement (CM).

        Parameters
        ----------
        > **rho0:** `matrix`
            -- Initial state (density matrix).

        > **W:** `matrix`
            -- Weight matrix.
        """
        ini_population = (self.ctrl0, self.measurement0)
        self.alg = QJL.DE(
            self.max_episode,
            self.p_num,
            ini_population,
            self.c,
            self.cr,
        )

        super().CM(rho0, W)

    def SM(self, W=[]):
        """
        Comprehensive optimization of the probe state and measurement (SM).

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.
        """
        ini_population = (self.psi, self.measurement0)
        self.alg = QJL.DE(
            self.max_episode,
            self.p_num,
            ini_population,
            self.c,
            self.cr,
        )

        super().SM(W)

    def SCM(self, W=[]):
        """
        Comprehensive optimization of the probe state, control and measurement (SCM).

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.
        """
        ini_population = (self.psi, self.ctrl0, self.measurement0)
        self.alg = QJL.DE(
            self.max_episode,
            self.p_num,
            ini_population,
            self.c,
            self.cr,
        )

        super().SCM(W)
