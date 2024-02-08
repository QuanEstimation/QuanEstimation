from quanestimation import QJL
import quanestimation.MeasurementOpt.MeasurementStruct as Measurement


class AD_Mopt(Measurement.MeasurementSystem):
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

    > **Adam:** `bool`
        -- Whether or not to use Adam for updating measurements.

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
        Adam=False,
        measurement0=[],
        max_episode=300,
        epsilon=0.01,
        beta1=0.90,
        beta2=0.99,
        seed=1234,
        eps=1e-8,
        load=False,
    ):

        Measurement.MeasurementSystem.__init__(
            self, mtype, minput, savefile, measurement0, seed, eps, load 
        )

        self.Adam = Adam
        self.max_episode = max_episode
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0
        self.seed = seed

        if self.Adam:
            self.alg = QJL.AD(
                self.max_episode, self.epsilon, self.beta1, self.beta2
            )
        else:
            self.alg = QJL.AD(self.max_episode, self.epsilon)

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
        
        if self.mtype == "projection":
            raise ValueError(
                "AD is not available when mtype is projection. Supported methods are 'PSO' and 'DE'.",
            )
        else:
            super().CFIM(W)
