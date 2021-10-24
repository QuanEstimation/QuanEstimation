import numpy as np
from julia import Main

class StateOpt_DE():
    def __init__(self, tspan, rho_initial, H0, dH=[], Liouville_operator=[], \
                gamma=[], W=[], popsize=10, ini_population=[], c=0.5, c0=0.1, \
                c1=0.6, seed=1234, max_episodes=200):
        
        """
        --------
        inputs
        --------
        tspan:
            --description: time series.
            --type: array

        rho_initial:
            --description: initial state (density matrix).
            --type: matrix
            
        H0:
            --description: free Hamiltonian.
            --type: matrix

        dH:
            --description: derivatives of Hamiltonian on all parameters to
                                be estimated. For example, dH[0] is the derivative
                                vector on the first parameter.
            --type: list (of matrix)

        Liouville operator:
            --description: Liouville operator.
            --type: list (of matrix)

        gamma:
            --description: decay rates.
            --type: list (of float number)

        W:
            --description: weight matrix.
            --type: matrix

        popsize:
           --description: number of populations.
           --type: float
        
        """
        if type(dH) != list:
            raise TypeError('The derivative of Hamiltonian should be a list!')    
        
        if len(gamma) != len(Liouville_operator):
            raise TypeError('The length of decay rates and the length of Liouville operator should be the same!')
        
        if dH == []:
            dH = [np.zeros((len(H0), len(H0)))]
        
        self.tspan = tspan
        self.rho_initial = np.array(rho_initial,dtype=np.complex128)
        self.freeHamiltonian = np.array(H0,dtype=np.complex128)
        self.Hamiltonian_derivative = [np.array(x,dtype=np.complex128) for x in dH]
        self.Liouville_operator = [np.array(x, dtype=np.complex128) for x in Liouville_operator]
        self.gamma = gamma

        self.popsize = popsize
        self.ini_population = ini_population
        self.c = c
        self.c0 = c0
        self.c1 = c1
        self.seed = seed
        self.max_episodes = max_episodes
        if W == []:
            self.W = np.eye(len(dH))
        else:
            self.W = W

    def QFIM(self, save_file):
        if self.gamma == []:
            diffevo = Main.QuanEstimation.StateOptDE_TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                                                                               self.rho_initial, self.tspan, self.W)
            if len(self.Hamiltonian_derivative) == 1:
                Main.QuanEstimation.DiffEvo_QFI(diffevo, self.popsize, self.ini_population, self.c, self.c0, self.c1, self.seed, self.max_episodes, save_file)
            else:
                Main.QuanEstimation.DiffEvo_QFIM(diffevo, self.popsize, self.ini_population, self.c, self.c0, self.c1, self.seed, self.max_episodes, save_file)
        else:
            diffevo = Main.QuanEstimation.StateOptDE_TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho_initial, self.tspan, \
                        self.Liouville_operator, self.gamma, self.W)
            if len(self.Hamiltonian_derivative) == 1:
                Main.QuanEstimation.DiffEvo_QFI(diffevo, self.popsize, self.ini_population, self.c, self.c0, self.c1, self.seed, self.max_episodes, save_file)
            else:
                Main.QuanEstimation.DiffEvo_QFIM(diffevo, self.popsize, self.ini_population, self.c, self.c0, self.c1, self.seed, self.max_episodes, save_file)

    def CFIM(self, M, save_file):
        if self.gamma == []:
            diffevo = Main.QuanEstimation.StateOptDE_TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                                                                               self.rho_initial, self.tspan, self.W)
            if len(self.Hamiltonian_derivative) == 1:
                Main.QuanEstimation.DiffEvo_CFI(M, diffevo, self.popsize, self.ini_population, self.c, self.c0, self.c1, self.seed, self.max_episodes, save_file)
            else:
                Main.QuanEstimation.DiffEvo_CFIM(M, diffevo, self.popsize, self.ini_population, self.c, self.c0, self.c1, self.seed, self.max_episodes, save_file)
        else:
            diffevo = Main.QuanEstimation.StateOptDE_TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho_initial, self.tspan, \
                        self.Liouville_operator, self.gamma, self.W)
            if len(self.Hamiltonian_derivative) == 1:
                Main.QuanEstimation.DiffEvo_CFI(M, diffevo, self.popsize, self.ini_population, self.c, self.c0, self.c1, self.seed, self.max_episodes, save_file)
            else:
                Main.QuanEstimation.DiffEvo_CFIM(M, diffevo, self.popsize, self.ini_population, self.c, self.c0, self.c1, self.seed, self.max_episodes, save_file)
          