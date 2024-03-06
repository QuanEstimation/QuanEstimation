from quanestimation import QJL
import quanestimation.StateOpt.StateStruct as State


class PSO_Sopt(State.StateSystem):
    """
    Attributes
    ----------
    > **savefile:**  `bool`
        -- Whether or not to save all the states.  
        If set `True` then the states and the values of the objective function 
        obtained in all episodes will be saved during the training. If set `False` 
        the state in the final episode and the values of the objective function 
        in all episodes will be saved.

    > **p_num:** `int`
        -- The number of particles.

    > **psi0:** `list of arrays`
        -- Initial guesses of states.

    > **max_episode:** `int or list`
        -- If it is an integer, for example max_episode=1000, it means the 
        program will continuously run 1000 episodes. However, if it is an
        array, for example max_episode=[1000,100], the program will run 
        1000 episodes in total but replace states of all  the particles 
        with global best every 100 episodes.
  
    > **c0:** `float`
        -- The damping factor that assists convergence, also known as inertia weight.

    > **c1:** `float`
        -- The exploitation weight that attracts the particle to its best previous 
        position, also known as cognitive learning factor.

    > **c2:** `float`
        -- The exploitation weight that attracts the particle to the best position  
        in the neighborhood, also known as social learning factor.

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
        max_episode=[1000, 100],
        c0=1.0,
        c1=2.0,
        c2=2.0,
        seed=1234,
        eps=1e-8,
        load=False,
    ):

        State.StateSystem.__init__(self, savefile, psi0, seed, eps, load)

        """
        --------
        inputs
        --------
        p_num:
           --description: the number of particles.
           --type: int

        psi0:
           --description: initial guesses of states (kets).
           --type: array

        max_episode:
            --description: max number of the training episodes.
            --type: int

        c0:
            --description: damping factor that assists convergence.
            --type: float

        c1:
            --description: exploitation weight that attract the particle to its best 
            previous position.
            --type: float

        c2:
            --description: exploitation weight that attract the particle to the best 
            position in the neighborhood.
            --type: float

        seed:
            --description: random seed.
            --type: int

        """

        is_int = isinstance(max_episode, int)
        self.max_episode = max_episode if is_int else QJL.Vector[QJL.Int64](max_episode)
        self.p_num = p_num
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
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
        ini_particle = (self.psi,)
        self.alg = QJL.PSO(
            self.max_episode,
            self.p_num,
            ini_particle,
            self.c0,
            self.c1,
            self.c2,
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
        ini_particle = (self.psi,)
        self.alg = QJL.PSO(
            self.max_episode,
            self.p_num,
            ini_particle,
            self.c0,
            self.c1,
            self.c2,
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
        ini_particle = (self.psi,)
        self.alg = QJL.PSO(
            self.max_episode,
            self.p_num,
            ini_particle,
            self.c0,
            self.c1,
            self.c2,
        )
        
        super().HCRB(W)
