import numpy as np
from julia import Main
import warnings
import quanestimation.ComprehensiveOpt.ComprehensiveStruct as Comp

class DE_Compopt(Comp.ComprehensiveSystem):
    def __init__(self, option, tspan, H0, dH, Hc, decay=[], ctrl_bound=[], W=[], psi0=[], measurement0=[], \
                popsize=10, ctrl0=[], max_episode=1000, c=1.0, cr=0.5, seed=1234):

        Comp.ComprehensiveSystem.__init__(self, tspan, psi0, measurement0, H0, Hc, dH, decay, ctrl_bound, W, ctrl0, seed, accuracy=1e-8)
        
        """
        --------
        inputs
        --------
        popsize:
           --description: the number of populations.
           --type: int
        
        ctrl0:
           --description: initial guesses of controls.
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
            self.psi0 = [self.psi0]
        else:
            self.psi0 = psi0

        if ctrl0 == []: 
            self.ctrl0 = [np.array(self.control_coefficients)]
        else:
            self.ctrl0 = ctrl0

        if measurement0 == []: 
            measurement0 = [np.array(self.M)]
        else:
            measurement0 = measurement0
        self.measurement0 = measurement0

        self.popsize =  popsize
        self.max_episode = max_episode
        self.c = c
        self.cr = cr
        self.seed = seed
        self.option = option

    def SC(self, target="QFIM", M=[], save_file=False):
        diffevo = Main.QuanEstimation.Compopt_SCopt(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi, \
                    self.tspan, self.decay_opt, self.gamma, self.control_Hamiltonian, self.control_coefficients, \
                    self.ctrl_bound, self.W, self.accuracy)
        if target == "QFIM":
            Main.QuanEstimation.DE_Compopt_SCopt(diffevo, self.popsize, self.psi0, self.ctrl0, self.c, self.cr, self.seed, self.max_episode, save_file)
            self.load_save_state()
        elif target == "CFIM":
            M = [np.array(x, dtype=np.complex128) for x in M]
            Main.QuanEstimation.DE_Compopt_SCopt(M, diffevo, self.popsize, self.psi0, self.ctrl0, self.c, self.cr, self.seed, self.max_episode, save_file)
            self.load_save_state()
        else:
            raise ValueError("{!r} is not a valid value for target, supported values are 'QFIM', 'CFIM'.".format(self.option))
    
    def CM(self, rho0, save_file=False):
        rho0 = np.array(rho0,dtype=np.complex128)
        diffevo = Main.QuanEstimation.Compopt_CMopt(self.freeHamiltonian, self.Hamiltonian_derivative, \
                    self.tspan, self.decay_opt, self.gamma, self.control_Hamiltonian, self.control_coefficients, \
                    self.ctrl_bound, self.M, self.W, self.accuracy)
        Main.QuanEstimation.DE_Compopt_CMopt(rho0, diffevo, self.popsize, self.ctrl0, self.measurement0, self.c, self.cr, self.seed, self.max_episode, save_file)
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

        diffevo = Main.QuanEstimation.Compopt_SMopt(freeHamiltonian, self.Hamiltonian_derivative, self.psi, \
                    self.tspan, self.decay_opt, self.gamma, self.M, self.W, self.accuracy)
        Main.QuanEstimation.DE_Compopt_SMopt(diffevo, self.popsize, self.psi0, self.measurement0, self.c, self.cr, self.seed, self.max_episode, save_file)
        self.load_save_meas()
        
    def SCM(self, save_file=False):
        diffevo = Main.QuanEstimation.Compopt_SCMopt(self.freeHamiltonian, self.Hamiltonian_derivative, self.psi, \
                    self.tspan, self.decay_opt, self.gamma, self.control_Hamiltonian, self.control_coefficients, \
                    self.ctrl_bound, self.M, self.W, self.accuracy)
        Main.QuanEstimation.DE_Compopt_SCMopt(diffevo, self.popsize, self.psi0, self.ctrl0, self.measurement0, self.c, self.cr, self.seed, self.max_episode, save_file)
        self.load_save_state()
        self.load_save_meas()
        