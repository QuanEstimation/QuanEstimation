import numpy as np
from julia import Main
import quanestimation.ComprehensiveOpt.ComprehensiveStruct as Comp

class PSO_Compopt(Comp.ComprehensiveSystem):
    def __init__(self, tspan, H0, dH, Hc, decay=[], ctrl_bound=[], W=[], psi0=[], measurement0=[], \
                 particle_num=10, ctrl0=[], max_episode=[1000, 100], c0=1.0, c1=2.0, c2=2.0, seed=1234):

        Comp.ComprehensiveSystem.__init__(self, tspan, psi0, measurement0, H0, Hc, dH, decay, ctrl_bound, W, ctrl0, seed, eps=1e-8)
        
        """
        --------
        inputs
        --------
        particle_num:
           --description: the number of particles.
           --type: int
        
        ctrl0:
           --description: initial guesses of controls.
           --type: array

        max_episode:
            --description: max number of the training episodes.
            --type: int or array
        
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
            self.psi0 = [self.psi0]
        else:
            self.psi0 = psi0

        if ctrl0 == []: 
            self.ctrl0 = [np.array(self.control_coefficients)]
        else:
            self.ctrl0 = ctrl0

        if measurement0 == []: 
            measurement = [np.array(self.M)]
        else:
            measurement = measurement0
        self.measurement0 = [np.array(x, dtype=np.complex128) for x in measurement0]

        self.particle_num = particle_num
        self.max_episode = max_episode
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

    def SC(self, target="QFIM", M=[], save_file=False):
        pso = Main.QuanEstimation.SC_Compopt(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi, \
                    self.tspan, self.decay_opt, self.gamma, self.control_Hamiltonian, self.control_coefficients, \
                    self.ctrl_bound, self.W, self.eps)
        if target == "QFIM":
            Main.QuanEstimation.SC_PSO_Compopt(pso, self.max_episode, self.particle_num, self.psi0, self.ctrl0, self.c0, self.c1, self.c2, \
                                         self.seed, save_file)
            self.load_save_state()
        elif target == "CFIM":
            if M==[]:
                raise ValueError("M should not be empty.")
            else: 
                M = [np.array(x, dtype=np.complex128) for x in M]
                Main.QuanEstimation.SC_PSO_Compopt(M, pso, self.max_episode, self.particle_num, self.psi0, self.ctrl0, self.c0, self.c1, self.c2, \
                                         self.seed, save_file)
                self.load_save_state()
        else:
            raise ValueError("{!r} is not a valid value for target, supported values are 'QFIM', 'CFIM'.".format(target))
    
    def CM(self, rho0, save_file=False):
        rho0 = np.array(rho0,dtype=np.complex128)
        pso = Main.QuanEstimation.CM_Compopt(self.freeHamiltonian, self.Hamiltonian_derivative, \
                    self.tspan, self.decay_opt, self.gamma, self.control_Hamiltonian, self.control_coefficients, \
                    self.ctrl_bound, self.M, self.W, self.eps)
        Main.QuanEstimation.CM_PSO_Compopt(rho0, pso, self.max_episode, self.particle_num, self.psi0, self.ctrl0, self.measurement, self.c0, self.c1, self.c2, \
                                         self.seed, save_file)
        self.load_save_meas()

    def SM(self, save_file=False):
        if len(self.control_coefficients[0]) == 1:
            H0 = np.array(self.freeHamiltonian, dtype=np.complex128)
            Hc = [np.array(x, dtype=np.complex128) for x in self.control_Hamiltonian]
            Htot = H0 + sum([Hc[i]*self.control_coefficients[i][0] for i in range(len(self.control_coefficients))])
            freeHamiltonian = np.array(Htot, dtype=np.complex128)
        else:
            H0 = np.array(self.freeHamiltonian, dtype=np.complex128)
            Hc = [np.array(x, dtype=np.complex128) for x in self.control_Hamiltonian]
            Htot = []
            for i in range(len(self.control_coefficients[0])):
                S_ctrl = sum([Hc[j]*self.control_coefficients[j][i] for j in range(len(self.control_coefficients))])
                Htot.append(H0+S_ctrl)
            freeHamiltonian = [np.array(x, dtype=np.complex128) for x in Htot] 

        pso = Main.QuanEstimation.SM_Compopt(freeHamiltonian, self.Hamiltonian_derivative, self.psi, \
                    self.tspan, self.decay_opt, self.gamma, self.M, self.W, self.eps)
        Main.QuanEstimation.SM_PSO_Compopt(pso, self.max_episode, self.particle_num, self.psi0, self.measurement, \
                self.c0, self.c1, self.c2, self.seed, save_file)
        self.load_save_meas()
        
    def SCM(self, save_file=False):
        pso = Main.QuanEstimation.SCM_Compopt(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi, \
                    self.tspan, self.decay_opt, self.gamma, self.control_Hamiltonian, self.control_coefficients, \
                    self.ctrl_bound, self.M, self.W, self.eps)
        Main.QuanEstimation.SCM_PSO_Compopt(pso, self.max_episode, self.particle_num, self.psi0, self.ctrl0, self.measurement, self.c0, self.c1, self.c2, \
                                         self.seed, save_file)
        self.load_save_state()
        self.load_save_meas()
