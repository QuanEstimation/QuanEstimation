from julia import Main
import warnings
import quanestimation.StateOpt.StateStruct as State

class NM_Sopt(State.StateSystem):
    def __init__(self, tspan, H0, dH, Hc=[], ctrl=[], decay=[], W=[], state_num=10, psi0=[], \
                 max_episode=1000, ar=1.0, ae=2.0, ac=0.5, as0=0.5, seed=1234, load=False):

        State.StateSystem.__init__(self, tspan, psi0, H0, dH, Hc, ctrl, decay, W, seed, load, eps=1e-8)

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

        eps:
            --description: calculation eps.
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
                                                                self.tspan, self.decay_opt, self.gamma, self.W, self.eps)
            Main.QuanEstimation.QFIM_NM_Sopt(neldermead, self.state_num, self.ini_state, self.ar, self.ae, self.ac, self.as0, \
                                        self.max_episode, self.seed, save_file)
        else:
            neldermead = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                                                                    self.psi0, self.tspan, self.W, self.eps)
            Main.QuanEstimation.QFIM_NM_Sopt(neldermead, self.state_num, self.ini_state, self.ar, self.ae, self.ac, self.as0, \
                                        self.max_episode, self.seed, save_file)
        self.load_save()

    def CFIM(self, M, save_file=False):
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
                                                                self.tspan, self.decay_opt, self.gamma, self.W, self.eps)
            Main.QuanEstimation.CFIM_NM_Sopt(M, neldermead, self.state_num, self.ini_state, self.ar, self.ae, self.ac, \
                                        self.as0, self.max_episode, self.seed, save_file)
        else:
            neldermead = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                                                                    self.psi0, self.tspan, self.W, self.eps)
            Main.QuanEstimation.CFIM_NM_Sopt(M, neldermead, self.state_num, self.ini_state, self.ar, self.ae, self.ac, \
                                        self.as0, self.max_episode, self.seed, save_file)
        self.load_save()
            
    def HCRB(self, save_file=False):
        """
        Description: use nelder-mead method to search the optimal initial state that maximize the HCRB.

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the initial state for each episode but overwrite in the next episode and all the HCRB.
                           False: save the initial states for the last episode and all the HCRB.
            --type: bool
        """
        if len(self.Hamiltonian_derivative) == 1:
            warnings.warn("In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the objection function \
                           for state optimization", DeprecationWarning)
        else:
            if any(self.gamma):
                neldermead = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                                self.tspan, self.decay_opt, self.gamma, self.W, self.eps)
                Main.QuanEstimation.HCRB_NM_Sopt(neldermead, self.state_num, self.ini_state, self.ar, self.ae, self.ac, self.as0, \
                                        self.max_episode, self.seed, save_file)
            else:
                neldermead = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                                                                    self.psi0, self.tspan, self.W, self.eps)
                Main.QuanEstimation.HCRB_NM_Sopt(neldermead, self.state_num, self.ini_state, self.ar, self.ae, self.ac, self.as0, \
                                        self.max_episode, self.seed, save_file)
            self.load_save()
            