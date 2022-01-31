from julia import Main
import warnings
import quanestimation.StateOpt.StateStruct as State

class PSO_Sopt(State.StateSystem):
    def __init__(self, tspan, H0, dH, Hc=[], ctrl=[], decay=[], W=[], particle_num=10, psi0=[], \
                 max_episode=[1000,100], c0=1.0, c1=2.0, c2=2.0, seed=1234, load=False):

        State.StateSystem.__init__(self, tspan, psi0, H0, dH, Hc, ctrl, decay, W, seed, load, accuracy=1e-8)
        
        """
        --------
        inputs
        --------
        particle_num:
           --description: the number of particles.
           --type: int

        psi0:
           --description: initial guesses of states (kets).
           --type: array

        max_episode:
            --description: max number of the training episodes.
            --type: int
        
        c0:
            --description: damping factor that assists convergence.
            --type: float

        c1:
            --description: exploitation weight that attract the particle to its best previous position.
            --type: float
        
        c2:
            --description: exploitation weight that attract the particle to the best position in the neighborhood.
            --type: float
        
        seed:
            --description: random seed.
            --type: int
        
        """
        
        if psi0 == []: 
            ini_particle = [self.psi0]
        else:
            ini_particle = psi0

        self.particle_num = particle_num
        self.ini_particle = ini_particle
        self.max_episode = max_episode
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.v0 = 0.1
        self.seed = seed
    
    def QFIM(self, save_file=False):
        """
        Description: use particle swarm optimizaiton algorithm to search the optimal initial state that maximize the 
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
            pso = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                         self.tspan, self.decay_opt, self.gamma, self.W, self.accuracy)
            Main.QuanEstimation.QFIM_PSO_Sopt(pso, self.max_episode, self.particle_num, self.ini_particle, self.c0, self.c1, \
                                         self.c2, self.v0, self.seed, save_file)
        else:
            pso = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                             self.tspan, self.W, self.accuracy)
            Main.QuanEstimation.QFIM_PSO_Sopt(pso, self.max_episode, self.particle_num, self.ini_particle, self.c0, self.c1, \
                                         self.c2, self.v0, self.seed, save_file)
        self.load_save()

    def CFIM(self, M, save_file=False):
        """
        Description: use particle swarm optimizaiton algorithm to search the optimal initial state that maximize the 
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
            pso = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                         self.tspan, self.decay_opt, self.gamma, self.W, self.accuracy)
            Main.QuanEstimation.CFIM_PSO_Sopt(M, pso, self.max_episode, self.particle_num, self.ini_particle, self.c0, \
                                         self.c1, self.c2, self.v0, self.seed, save_file)
        else:
            pso = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                             self.tspan, self.W, self.accuracy)
            Main.QuanEstimation.CFIM_PSO_Sopt(M, pso, self.max_episode, self.particle_num, self.ini_particle, self.c0, \
                                         self.c1, self.c2, self.v0, self.seed, save_file)
        self.load_save()

    def HCRB(self, save_file=False):
        """
        Description: use particle swarm optimizaiton algorithm to search the optimal initial state that maximize the HCRB.

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
                pso = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                         self.tspan, self.decay_opt, self.gamma, self.W, self.accuracy)
                Main.QuanEstimation.HCRB_PSO_Sopt(pso, self.max_episode, self.particle_num, self.ini_particle, self.c0, self.c1, \
                                         self.c2, self.v0, self.seed, save_file)
            else:
                pso = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, \
                                                             self.tspan, self.W, self.accuracy)
                Main.QuanEstimation.HCRB_PSO_Sopt(pso, self.max_episode, self.particle_num, self.ini_particle, self.c0, self.c1, \
                                         self.c2, self.v0, self.seed, save_file)
            self.load_save()
            