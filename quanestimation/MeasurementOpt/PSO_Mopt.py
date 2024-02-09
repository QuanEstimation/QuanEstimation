from quanestimation import QJL
import quanestimation.MeasurementOpt.MeasurementStruct as Measurement


class PSO_Mopt(Measurement.MeasurementSystem):
    """
    Attributes
    ----------
    > **savefile:** `bool`
        -- Whether or not to save all the measurements.  
        If set `True` then the measurements and the values of the 
        objective function obtained in all episodes will be saved during 
        the training. If set `False` the measurement in the final 
        episode and the values of the objective function in all episodes 
        will be saved.

    > **p_num:** `int`
        -- The number of particles.

    > **measurement0:** `list of arrays`
        -- Initial guesses of measurements.

    > **max_episode:** `int or list`
        -- If it is an integer, for example max_episode=1000, it means the 
        program will continuously run 1000 episodes. However, if it is an
        array, for example max_episode=[1000,100], the program will run 
        1000 episodes in total but replace measurements of all  the particles 
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
        -- Whether or not to load measurements in the current location.  
        If set `True` then the program will load measurement from "measurements.csv"
        file in the current location and use it as the initial measurement.
    """

    def __init__(
        self,
        mtype,
        minput,
        savefile=False,
        p_num=10,
        measurement0=[],
        max_episode=[1000, 100],
        c0=1.0,
        c1=2.0,
        c2=2.0,
        seed=1234,
        eps=1e-8,
        load=False,
    ):

        Measurement.MeasurementSystem.__init__(
            self, mtype, minput, savefile, measurement0, seed, eps, load
        )

        self.p_num = p_num
        is_int = isinstance(max_episode, int)
        self.max_episode = max_episode if is_int else QJL.Vector[QJL.Int64](max_episode)
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

    def CFIM(self, W=[]):
        r"""
        Choose CFI or $\mathrm{Tr}(WI^{-1})$ as the objective function. 
        In single parameter estimation the objective function is CFI and 
        in multiparameter estimation it will be $\mathrm{Tr}(WI^{-1})$.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.
        """
        ini_particle = (self.measurement0,)
        self.alg = QJL.PSO(
            self.max_episode,
            self.p_num,
            ini_particle,
            self.c0,
            self.c1,
            self.c2,
        )
        
        super().CFIM(W)
