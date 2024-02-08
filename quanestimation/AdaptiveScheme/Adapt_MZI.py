import numpy as np
from quanestimation import QJL
from quanestimation.Common.Common import brgd, annihilation


class Adapt_MZI:
    """
    Attributes
    ----------
    > **x:** `list`
        -- The regimes of the parameters for the integral.

    > **p:** `multidimensional array`
        -- The prior distribution.

    > **rho0:** `matrix`
        -- Initial state (density matrix).

    """
    def __init__(self, x, p, rho0):

        self.x = x
        self.p = p
        self.rho0 = rho0
        self.N = int(np.sqrt(len(rho0))) - 1
        # self.a = annihilation(self.N + 1)

    def general(self):
        self.MZI_type = "general"

    def online(self, target="sharpness", output="phi"):
        """
        Parameters
        ----------
        > **target:** `string`
            -- Setting the target function for calculating the tunable phase. Options are:  
            "sharpness" (default) -- Sharpness.  
            "MI" -- Mutual information. 

        > **output:** `string`
            -- The output the class. Options are:  
            "phi" (default) -- The tunable phase.  
            "dphi" -- Phase difference. 
        """
        phi = QJL.adaptMZI_online(
            self.x, self.p, self.rho0, output, target
        )

    def offline(
        self,
        target="sharpness",
        method="DE",
        p_num=10,
        deltaphi0=[],
        c=1.0,
        cr=0.5,
        c0=1.0,
        c1=2.0,
        c2=2.0,
        seed=1234,
        max_episode=1000,
        eps=1e-8,
    ):
        """
        Parameters
        ----------
        > **target:** `string`
            -- Setting the target function for calculating the tunable phase. Options are:  
            "sharpness" (default) -- Sharpness.  
            "MI" -- Mutual information. 

        > **method:** `string`
            -- The method for the adaptive phase estimation. Options are:  
            "DE" (default) -- DE algorithm for the adaptive phase estimation.    
            "PSO" -- PSO algorithm for the adaptive phase estimation.

        If the `method=DE`, the parameters are:
        > **p_num:** `int`
            -- The number of populations.

        > **deltaphi0:** `list`
            -- Initial guesses of phase difference.

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
        
        If the `method=PSO`, the parameters are:

        > **deltaphi0:** `list`
            -- Initial guesses of phase difference.

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

        > **eps:** `float`
            -- Machine epsilon.
        """
        comb_tp = brgd(self.N)
        comb = [
            np.array([int(list(comb_tp[i])[j]) for j in range(self.N)])
            for i in range(2**self.N)
        ]
        
        if method == "DE":
            QJL.DE_deltaphiOpt(
                self.x,
                self.p,
                self.rho0,
                comb,
                p_num,
                deltaphi0,
                c,
                cr,
                seed,
                max_episode,
                target,
                eps,
            )
        elif method == "PSO":
            QJL.PSO_deltaphiOpt(
                self.x,
                self.p,
                self.rho0,
                comb,
                p_num,
                deltaphi0,
                c0,
                c1,
                c2,
                seed,
                max_episode,
                target,
                eps,
            )
        else:
            raise ValueError(
                "{!r} is not a valid value for method, supported values are 'DE' and 'PSO'.".format(
                    method
                )
            )
