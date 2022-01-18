from julia import Main
import warnings
import quanestimation.StateOpt.StateStruct as State

class DE_Sopt(State.StateSystem):
    def __init__(self, tspan, H0, dH, Hc=[], ctrl=[], decay=[], W=[], popsize=10, \
                 psi0=[], max_episode=1000, c=1.0, cr=0.5, seed=1234):

        State.StateSystem.__init__(self, tspan, psi0, H0, dH, decay, W, seed, accuracy=1e-8)
        
        """
        --------
        inputs
        --------
        popsize:
           --description: the number of populations.
           --type: int

        psi0:
           --description: initial guesses of states (kets).
           --type: array

        max_episode:
            --description: max number of the training episodes.
            --type: int
        
        c:
            --description: mutation constant.
            --type: float

        cr:
            --description: crossover constant.
            --type: float
        
        seed:
            --description: random seed.
            --type: int
        
        """
        if psi0 == []: 
            ini_population = [self.psi0]
        else:
            ini_population = psi0

        self.popsize = popsize
        self.ini_population = ini_population
        self.c = c
        self.cr = cr
        self.seed = seed
        self.max_episode = max_episode

    def QFIM(self, save_file=False):
        """
        Description: use differential evolution algorithm to search the optimal initial state that maximize the 
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
            diffevo = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                             self.tspan, self.decay_opt, self.gamma, self.W, self.accuracy)
            Main.QuanEstimation.QFIM_DE_Sopt(diffevo, self.popsize, self.ini_population, self.c, self.cr, self.seed, \
                                        self.max_episode, save_file)
        else:
            diffevo = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                                                                 self.psi0, self.tspan, self.W, self.accuracy)
            Main.QuanEstimation.QFIM_DE_Sopt(diffevo, self.popsize, self.ini_population, self.c, self.cr, self.seed, \
                                        self.max_episode, save_file)
        self.load_save()

    def CFIM(self, Measurement, save_file=False):
        """
        Description: use differential evolution algorithm to search the optimal initial state that maximize the 
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
            diffevo = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                             self.tspan, self.decay_opt, self.gamma, self.W, self.accuracy)
            Main.QuanEstimation.CFIM_DE_Sopt(Measurement, diffevo, self.popsize, self.ini_population, self.c, self.cr, self.seed, \
                                       self.max_episode, save_file)
        else:
            diffevo = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                                                                 self.psi0, self.tspan, self.W, self.accuracy)
            Main.QuanEstimation.CFIM_DE_Sopt(Measurement, diffevo, self.popsize, self.ini_population, self.c, self.cr, self.seed, \
                                        self.max_episode, save_file)
        self.load_save()
    
    def HCRB(self, save_file=False):
        """
        Description: use differential evolution algorithm to search the optimal initial state that maximize the HCRB.

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the initial state for each episode but overwrite in the next episode and all the HCRB.
                           False: save the initial states for the last episode and all the HCRB.
            --type: bool
        """
        if len(self.Hamiltonian_derivative) == 1:
            warnings.warn('In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the objection function \
                           for state optimization', DeprecationWarning)
        else:
            if any(self.gamma):
                diffevo = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                             self.tspan, self.decay_opt, self.gamma, self.W, self.accuracy)
                Main.QuanEstimation.HCRB_DE_Sopt(diffevo, self.popsize, self.ini_population, self.c, self.cr, self.seed, \
                                       self.max_episode, save_file)
            else:
                diffevo = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, \
                                                                 self.psi0, self.tspan, self.W, self.accuracy)
                Main.QuanEstimation.HCRB_DE_Sopt(diffevo, self.popsize, self.ini_population, self.c, self.cr, self.seed, \
                                        self.max_episode, save_file)
            self.load_save()
            