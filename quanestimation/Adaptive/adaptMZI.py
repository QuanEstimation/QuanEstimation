import numpy as np
from julia import Main
from quanestimation.Common.common import brgd, annihilation


class adaptMZI:
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
        self.a = annihilation(self.N + 1)

    def general(self):
        self.MZI_type = "general"

    def online(self, output="phi"):
        """
        Parameters
        ----------
        > **output:** `string`
            -- The output the class. Options are:  
            "phi" (default) -- The tunable phase.  
            "dphi" -- Phase difference. 
        """
        phi = Main.QuanEstimation.adaptMZI_online(
            self.x, self.p, self.rho0, self.a, output
        )

    def offline(
        self,
        method="DE",
        popsize=10,
        particle_num=10,
        DeltaPhi0=[],
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
        > **method:** `string`
            -- The method for the adaptive phase estimation. Options are:  
            "DE" (default) -- DE algorithm for the adaptive phase estimation.    
            "PSO" -- PSO algorithm for the adaptive phase estimation.

        If the `method=DE`, the parameters are:
        > **popsize:** `int`
            -- The number of populations.

        > **DeltaPhi0:** `list`
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
        > **particle_num:** `int`
            -- The number of particles.

        > **DeltaPhi0:** `list`
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

        > **seed:** `int`
            -- Random seed.

        > **eps:** `float`
            -- Machine epsilon.
        """
        comb_tp = brgd(self.N)
        comb = [
            np.array([int(list(comb_tp[i])[j]) for j in range(self.N)])
            for i in range(2**self.N)
        ]
        if method == "DE":
            Main.QuanEstimation.DE_DeltaPhiOpt(
                self.x,
                self.p,
                self.rho0,
                self.a,
                comb,
                popsize,
                DeltaPhi0,
                c,
                cr,
                seed,
                max_episode,
                eps,
            )
        elif method == "PSO":
            Main.QuanEstimation.PSO_DeltaPhiOpt(
                self.x,
                self.p,
                self.rho0,
                self.a,
                comb,
                particle_num,
                DeltaPhi0,
                c0,
                c1,
                c2,
                seed,
                max_episode,
                eps,
            )
        else:
            raise ValueError(
                "{!r} is not a valid value for method, supported values are 'DE' and 'PSO'.".format(
                    method
                )
            )
