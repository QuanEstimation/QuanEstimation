from julia import Main
import quanestimation.StateOptimization.StateOptimization as stateopt

class StateOpt_DDPG(stateopt.StateOptSystem):
    def __init__(self, tspan, psi0, H0, dH=[], decay=[], W=[], max_episode=500, layer_num=3, layer_dim=200, seed=1234):

        stateopt.StateOptSystem.__init__(self, tspan, psi0, H0, dH, decay, W, accuracy=1e-8)

        """
        ----------
        Inputs
        ----------
        layer_num:
            --description: the number of layers (including the input and output layer).
            --type: int

        layer_dim:
            --description: the number of neurons in the hidden layer.
            --type: int
        
        seed:
            --description: random seed.
            --type: int
        """

        self.layer_num = layer_num
        self.layer_dim = layer_dim
        self.max_episode = max_episode
        self.seed = seed

    def QFIM(self, save_file=False):
        """
        Description: use DDPG algorithm to search the optimal initial state that maximize the 
                     QFI (1/Tr(WF^{-1} with F the QFIM).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the initial state for each episode but overwrite in the next episode and all the QFI (Tr(WF^{-1})).
                           False: save the initial states for the last episode and all the QFI (Tr(WF^{-1})).
            --type: bool
        """
        if any(self.gamma):
            DDPG = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                          self.tspan, self.decay_opt, self.gamma, self.W, self.accuracy)
            Main.QuanEstimation.DDPG_QFIM(DDPG, self.layer_num, self.layer_dim, self.seed, self.max_episode, save_file)
        else:
            DDPG = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                              self.tspan, self.W, self.accuracy)
            Main.QuanEstimation.DDPG_QFIM(DDPG, self.layer_num, self.layer_dim, self.seed, self.max_episode, save_file)
        self.load_save()
            
    def CFIM(self, Measurement, save_file=False):
        """
        Description: use DDPG algorithm to search the optimal initial state that maximize the 
                     CFI (1/Tr(WF^{-1} with F the CFIM).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the initial state for each episode but overwrite in the next episode and all the CFI (Tr(WF^{-1})).
                           False: save the initial states for the last episode and all the CFI (Tr(WF^{-1})).
            --type: bool
        """

        if any(self.gamma):
            DDPG = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                          self.tspan, self.decay_opt, self.gamma, self.W, self.accuracy)
            Main.QuanEstimation.DDPG_CFIM(Measurement, DDPG, self.layer_num, self.layer_dim, self.seed, self.max_episode, save_file)
        else:
            DDPG = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                              self.tspan, self.W, self.accuracy)
            Main.QuanEstimation.DDPG_CFIM(Measurement, DDPG, self.layer_num, self.layer_dim, self.seed, self.max_episode, save_file)
        self.load_save()