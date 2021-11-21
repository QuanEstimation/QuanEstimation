from julia import Main
import quanestimation.StateOptimization.StateOptimization as stateopt
class StateOpt_PSO(stateopt.StateOptSystem):
    def __init__(self, tspan, psi0, H0, dH=[], Decay=[], W=[], particle_num=10, ini_particle=[], \
                 max_episode=[1000,100], c0=1.0, c1=2.0, c2=2.0, seed=1234):

        stateopt.StateOptSystem.__init__(self, tspan, psi0, H0, dH, Decay, W)
        
        """
        --------
        inputs
        --------
        particle_num:
           --description: number of particles.
           --type: int

                ini_particle:
           --description: initial particles.
           --type: array

        max_episode:
            --description: max number of training episodes.
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
        
        if ini_particle == []: 
            ini_particle = [psi0]

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
                     QFI or Tr(WF^{-1}).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the initial state for each episode but overwrite in the nest episode and all the QFI or Tr(WF^{-1}).
                           False: save the initial states for the last episode and all the QFI or Tr(WF^{-1}).
            --type: bool
        """
        if self.gamma == [] or self.gamma[0] == 0.0:
            pso = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, self.tspan, self.W)
            Main.QuanEstimation.PSO_QFIM(pso, self.max_episode, self.particle_num, self.ini_particle, self.c0, self.c1, self.c2, self.v0, \
                                         self.seed, save_file)
        else:
            pso = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, self.tspan, \
                        self.Decay_opt, self.gamma, self.W)
            Main.QuanEstimation.PSO_QFIM(pso, self.max_episode, self.particle_num, self.ini_particle, self.c0, self.c1, self.c2, self.v0, \
                                         self.seed, save_file)
        self.load_save()

    def CFIM(self, Measurement, save_file=False):
        """
        Description: use particle swarm optimizaiton algorithm to search the optimal initial state that maximize the 
                     CFI or Tr(WF^{-1}).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save the initial state for each episode but overwrite in the nest episode and all the CFI or Tr(WF^{-1}).
                           False: save the initial states for the last episode and all the CFI or Tr(WF^{-1}).
            --type: bool
        """
        if self.gamma == [] or self.gamma[0] == 0.0:
            pso = Main.QuanEstimation.TimeIndepend_noiseless(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, self.tspan, self.W)
            Main.QuanEstimation.PSO_CFIM(Measurement, pso, self.max_episode, self.particle_num, self.ini_particle, self.c0, self.c1, self.c2, self.v0, \
                                         self.seed, save_file)
        else:
            pso = Main.QuanEstimation.TimeIndepend_noise(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi0, self.tspan, \
                        self.Decay_opt, self.gamma, self.W)
            Main.QuanEstimation.PSO_CFIM(Measurement, pso, self.max_episode, self.particle_num, self.ini_particle, self.c0, self.c1, self.c2, self.v0, \
                                         self.seed, save_file)
        self.load_save()