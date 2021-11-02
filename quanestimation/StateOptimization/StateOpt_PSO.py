import numpy as np
from julia import Main

class StateOpt_PSO():
    def __init__(self, tspan, rho_initial, H0, dH=[], Liouville_operator=[], \
                 gamma=[], W=[], particle_num=10, ini_particle=[], max_episodes=400, \
                 seed=100, c0=1.0, c1=2.0, c2=2.0, v0=0.01):
        
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
            
        particle_num:
           --description: number of particles.
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

        self.particle_num = particle_num
        self.ini_particle = ini_particle
        self.max_episodes = max_episodes
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.v0 = v0
        self.seed = seed
        if W == []:
            self.W = np.eye(len(dH))
        else:
            self.W = W
    
    def QFIM(self, save_file=False):

        if self.gamma == []:
            pso = Main.QuanEstimation.StateOptPSO_TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                                                                               self.rho_initial, self.tspan, self.W)
            if len(self.Hamiltonian_derivative) == 1:
                Main.QuanEstimation.PSO_QFI(pso, self.max_episodes, self.particle_num, self.ini_particle, self.c0, self.c1, self.c2, self.v0, \
                                        self.seed, save_file)
            else:
                Main.QuanEstimation.PSO_QFIM(pso, self.max_episodes, self.particle_num, self.ini_particle, self.c0, self.c1, self.c2, self.v0, \
                                         self.seed, save_file)
        else:
            pso = Main.QuanEstimation.StateOptPSO_TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho_initial, self.tspan, \
                        self.Liouville_operator, self.gamma, self.W)
            if len(self.Hamiltonian_derivative) == 1:
                Main.QuanEstimation.PSO_QFI(pso, self.max_episodes, self.particle_num, self.ini_particle, self.c0, self.c1, self.c2, self.v0, \
                                        self.seed, save_file)
            else:
                Main.QuanEstimation.PSO_QFIM(pso, self.max_episodes, self.particle_num, self.ini_particle, self.c0, self.c1, self.c2, self.v0, \
                                         self.seed, save_file)

    def CFIM(self, Measurement, save_file=False):
        
        if self.gamma == []:
            pso = Main.QuanEstimation.StateOptPSO_TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                                                                               self.rho_initial, self.tspan, self.W)
            if len(self.Hamiltonian_derivative) == 1:
                Main.QuanEstimation.PSO_CFI(Measurement, pso, self.max_episodes, self.particle_num, self.ini_particle, self.c0, self.c1, self.c2, self.v0, \
                                        self.seed, save_file)
            else:
                Main.QuanEstimation.PSO_CFIM(Measurement, pso, self.max_episodes, self.particle_num, self.ini_particle, self.c0, self.c1, self.c2, self.v0, \
                                         self.seed, save_file)
        else:
            pso = Main.QuanEstimation.StateOptPSO_TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho_initial, self.tspan, \
                        self.Liouville_operator, self.gamma, self.W)
            if len(self.Hamiltonian_derivative) == 1:
                Main.QuanEstimation.PSO_CFI(Measurement, pso, self.max_episodes, self.particle_num, self.ini_particle, self.c0, self.c1, self.c2, self.v0, \
                                        self.seed, save_file)
            else:
                Main.QuanEstimation.PSO_CFIM(Measurement, pso, self.max_episodes, self.particle_num, self.ini_particle, self.c0, self.c1, self.c2, self.v0, \
                                         self.seed, save_file)