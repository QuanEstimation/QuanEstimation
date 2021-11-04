import numpy as np
from julia import Main

class StateOpt_NM():
    def __init__(self, tspan, psi_initial, H0, dH=[], Liouville_operator=[], \
                gamma=[], W=[], state_num=10, ini_state=[], a_r=1.0, a_e=2.0, \
                a_c=0.5, a_s=0.5, seed=1234, max_episodes=200, epsilon=1e-3):
        """
        --------
        inputs
        --------
        tspan:
            --description: time series.
            --type: array

        psi_initial:
            --description: initial state.
            --type: array
            
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

        state_num:
           --description: number of input states.
           --type: float
        
        """

        if type(dH) != list:
            raise TypeError('The derivative of Hamiltonian should be a list!')    
        
        if len(gamma) != len(Liouville_operator):
            raise TypeError('The length of decay rates and the length of Liouville operator should be the same!')
        
        if dH == []:
            dH = [np.zeros((len(H0), len(H0)))]
        
        if ini_state == []: 
            ini_state = [psi_initial]
        
        self.tspan = tspan
        self.psi_initial = np.array(psi_initial,dtype=np.complex128)
        self.freeHamiltonian = np.array(H0,dtype=np.complex128)
        self.Hamiltonian_derivative = [np.array(x,dtype=np.complex128) for x in dH]
        self.Liouville_operator = [np.array(x, dtype=np.complex128) for x in Liouville_operator]
        self.gamma = gamma

        self.state_num = state_num
        self.ini_state = ini_state
        self.a_r = a_r
        self.a_e = a_e
        self.a_c = a_c
        self.a_s = a_s
        self.max_episodes = max_episodes
        self.epsilon = epsilon
        self.seed = seed
        if W == []:
            self.W = np.eye(len(dH))
        else:
            self.W = W

    def QFIM(self, save_file=False):
        if self.gamma == [] or self.gamma == 0.0:
            neldermead = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                                                                               self.psi_initial, self.tspan, self.W)
            if len(self.Hamiltonian_derivative) == 1:
                Main.QuanEstimation.NelderMead_QFI(neldermead, self.state_num, self.ini_state, self.a_r, self.a_e, self.a_c, self.a_s, self.epsilon, self.max_episodes, self.seed, save_file)
            else:
                Main.QuanEstimation.NelderMead_QFIM(neldermead, self.state_num, self.ini_state, self.a_r, self.a_e, self.a_c, self.a_s, self.epsilon, self.max_episodes, self.seed, save_file)
        else:
            neldermead = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi_initial, self.tspan, \
                        self.Liouville_operator, self.gamma, self.W)
            if len(self.Hamiltonian_derivative) == 1:
                Main.QuanEstimation.NelderMead_QFI(neldermead, self.state_num, self.ini_state, self.a_r, self.a_e, self.a_c, self.a_s, self.epsilon, self.max_episodes, self.seed, save_file)
            else:
                Main.QuanEstimation.NelderMead_QFIM(neldermead, self.state_num, self.ini_state, self.a_r, self.a_e, self.a_c, self.a_s, self.epsilon, self.max_episodes, self.seed, save_file)

    def CFIM(self, Measurement, save_file=False):
        if self.gamma == [] or self.gamma == 0.0:
            neldermead = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                                                                               self.psi_initial, self.tspan, self.W)
            if len(self.Hamiltonian_derivative) == 1:
                Main.QuanEstimation.NelderMead_CFI(Measurement, neldermead, self.state_num, self.ini_state, self.a_r, self.a_e, self.a_c, self.a_s, self.epsilon, self.max_episodes, self.seed, save_file)
            else:
                Main.QuanEstimation.NelderMead_CFIM(Measurement, neldermead, self.state_num, self.ini_state, self.a_r, self.a_e, self.a_c, self.a_s, self.epsilon, self.max_episodes, self.seed, save_file)
        else:
            neldermead = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi_initial, \
                                                                     self.tspan, self.Liouville_operator, self.gamma, self.W)
            if len(self.Hamiltonian_derivative) == 1:
                Main.QuanEstimation.NelderMead_CFI(Measurement, neldermead, self.state_num, self.ini_state, self.a_r, self.a_e, self.a_c, self.a_s, self.epsilon, self.max_episodes, self.seed, save_file)
            else:
                Main.QuanEstimation.NelderMead_CFIM(Measurement, neldermead, self.state_num, self.ini_state, self.a_r, self.a_e, self.a_c, self.a_s, self.epsilon, self.max_episodes, self.seed, save_file)

            