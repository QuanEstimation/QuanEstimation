from julia import Main
import quanestimation.StateOpt.StateStruct as State

class NM_Sopt(State.StateSystem):
    def __init__(self, tspan, H0, dH=[], decay=[], W=[], state_num=10, psi0=[], \
                 max_episode=1000, ar=1.0, ae=2.0, ac=0.5, as0=0.5, seed=1234):

        State.StateSystem.__init__(self, tspan, psi0, H0, dH, decay, W, seed, accuracy=1e-8)

        """
        --------
        inputs
        --------
        state_num:
           --description: the number of input states.
           --type: int
        
        psi0:
           --description: initial guesses of states (kets).
           --type: array

        max_episode:
            --description: max number of the training episodes.
            --type: int
        
        ar:
            --description: reflection constant.
            --type: float

        ae:
            --description: expansion constant.
            --type: float

        ac:
            --description: constraction constant.
            --type: float

        as0:
            --description: shrink constant.
            --type: float
        
        seed:
            --description: random seed.
            --type: int

        accuracy:
            --description: calculation accuracy.
            --type: float
        
        """
        
        if psi0 == []: 
            ini_state = [self.psi0]
        else:
            ini_state = psi0

        self.state_num = state_num
        self.ini_state = ini_state
        self.max_episode = max_episode
        self.ar = ar
        self.ae = ae
        self.ac = ac
        self.as0 = as0
        self.seed = seed

    def QFIM(self, save_file=False):
        """
        Description: use nelder-mead method to search the optimal initial state that maximize the 
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
            neldermead = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                                self.tspan, self.decay_opt, self.gamma, self.W, self.accuracy)
            Main.QuanEstimation.QFIM_NM_Sopt(neldermead, self.state_num, self.ini_state, self.ar, self.ae, self.ac, self.as0, \
                                        self.max_episode, self.seed, save_file)
        else:
            neldermead = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                                                                    self.psi0, self.tspan, self.W, self.accuracy)
            Main.QuanEstimation.QFIM_NM_Sopt(neldermead, self.state_num, self.ini_state, self.ar, self.ae, self.ac, self.as0, \
                                        self.max_episode, self.seed, save_file)
        self.load_save()

    def CFIM(self, Measurement, save_file=False):
        """
        Description: use nelder-mead method to search the optimal initial state that maximize the 
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
            neldermead = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                                self.tspan, self.decay_opt, self.gamma, self.W, self.accuracy)
            Main.QuanEstimation.CFIM_NM_Sopt(Measurement, neldermead, self.state_num, self.ini_state, self.ar, self.ae, self.ac, \
                                        self.as0, self.max_episode, self.seed, save_file)
        else:
            neldermead = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                                                                    self.psi0, self.tspan, self.W, self.accuracy)
            Main.QuanEstimation.CFIM_NM_Sopt(Measurement, neldermead, self.state_num, self.ini_state, self.ar, self.ae, self.ac, \
                                        self.as0, self.max_episode, self.seed, save_file)
        self.load_save()
            