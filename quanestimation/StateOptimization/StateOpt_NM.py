from julia import Main
import quanestimation.StateOptimization.StateOptimization as stateopt
class StateOpt_NM(stateopt.StateOptSystem):
    def __init__(self, tspan, psi_initial, H0, dH=[], Liouville_operator=[], gamma=[], W=[], \
                 state_num=10, ini_state=[], max_episodes=1000, a_r=1.0, a_e=2.0, \
                 a_c=0.5, a_s=0.5, seed=1234, precision=1e-6):

        stateopt.StateOptSystem.__init__(self, tspan, psi_initial, H0, dH, Liouville_operator, gamma, W)

        """
        --------
        inputs
        --------
        state_num:
           --description: number of input states.
           --type: int
        
        ini_state:
           --description: initial states.
           --type: array

        max_episodes:
            --description: max number of training episodes.
            --type: int
        
        a_r:
            --description: reflection constant.
            --type: float

        a_e:
            --description: expansion constant.
            --type: float

        a_c:
            --description: constraction constant.
            --type: float

        a_s:
            --description: shrink constant.
            --type: float
        
        seed:
            --description: random seed.
            --type: int

        precision:
            --description: calculation precision.
            --type: float
        
        """
        
        if ini_state == []: 
            ini_state = [psi_initial]

        self.state_num = state_num
        self.ini_state = ini_state
        self.max_episodes = max_episodes
        self.a_r = a_r
        self.a_e = a_e
        self.a_c = a_c
        self.a_s = a_s
        self.seed = seed
        self.precision = precision

    def QFIM(self, save_file=False):
        """
        Description: use nelder-mead method to search the optimal initial state that maximize the 
                     QFI or Tr(WF^{-1}).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the initial state for each episode but overwrite in the nest episode and all the QFI or Tr(WF^{-1}).
                           False: save the initial states for the last episode and all the QFI or Tr(WF^{-1}).
            --type: bool
        """

        if self.gamma == [] or self.gamma == 0.0:
            neldermead = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                                                                               self.psi_initial, self.tspan, self.W)
            Main.QuanEstimation.NM_QFIM(neldermead, self.state_num, self.ini_state, self.a_r, self.a_e, self.a_c, self.a_s, self.precision, self.max_episodes, self.seed, save_file)
        else:
            neldermead = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi_initial, self.tspan, \
                        self.Liouville_operator, self.gamma, self.W)
            Main.QuanEstimation.NM_QFIM(neldermead, self.state_num, self.ini_state, self.a_r, self.a_e, self.a_c, self.a_s, self.precision, self.max_episodes, self.seed, save_file)

    def CFIM(self, Measurement, save_file=False):
        """
        Description: use nelder-mead method to search the optimal initial state that maximize the 
                     CFI or Tr(WF^{-1}).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the initial state for each episode but overwrite in the nest episode and all the CFI or Tr(WF^{-1}).
                           False: save the initial states for the last episode and all the CFI or Tr(WF^{-1}).
            --type: bool
        """
        if self.gamma == [] or self.gamma == 0.0:
            neldermead = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                                                                               self.psi_initial, self.tspan, self.W)
            Main.QuanEstimation.NM_CFIM(Measurement, neldermead, self.state_num, self.ini_state, self.a_r, self.a_e, self.a_c, self.a_s, self.precision, self.max_episodes, self.seed, save_file)
        else:
            neldermead = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi_initial, \
                                                                     self.tspan, self.Liouville_operator, self.gamma, self.W)
            Main.QuanEstimation.NM_CFIM(Measurement, neldermead, self.state_num, self.ini_state, self.a_r, self.a_e, self.a_c, self.a_s, self.precision, self.max_episodes, self.seed, save_file)

            