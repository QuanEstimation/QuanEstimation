import numpy as np
from julia import Main
import warnings
import quanestimation.ComprehensiveOpt.ComprehensiveStruct as Comp
from quanestimation.Common.common import SIC

class AD_Compopt(Comp.ComprehensiveSystem):
    def __init__(
        self,
        save_file=False,
        Adam=False,
        psi0=[],
        ctrl0=[],
        measurement0=[],
        max_episode=300,
        epsilon=0.01,
        beta1=0.90,
        beta2=0.99,
        seed=1234,
        eps=1e-8):

        Comp.ComprehensiveSystem.__init__(self, psi0, ctrl0, measurement0, save_file, seed, eps)

        """
        ----------
        Inputs
        ----------
        save_file:
            --description: True: save the states (or controls, measurements) and the value of the 
                                 target function for each episode.
                           False: save the states (or controls, measurements) and all the value 
                                   of the target function for the last episode.
            --type: bool 
            
        Adam:
            --description: whether to use Adam to update the controls.
            --type: bool (True or False)

        psi0:
           --description: initial guesses of states (kets).
           --type: array
           
        ctrl0:
            --description: initial control coefficients.
            --type: list (of vector)
            
        measurement0:
           --description: a set of POVMs.
           --type: list (of vector)

        max_episode:
            --description: max number of the training episodes.
            --type: int

        epsilon:
            --description: learning rate.
            --type: float

        beta1:
            --description: the exponential decay rate for the first moment estimates .
            --type: float

        beta2:
            --description: the exponential decay rate for the second moment estimates .
            --type: float

        """

        self.Adam = Adam
        self.max_episode = max_episode
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0
        self.seed = seed

    def SC(self, W=[], M=[], target="QFIM", dtype="SLD"):
        """
        Description: use auto-GRAPE (GRAPE) algorithm to optimize states and control coefficients.

        ---------
        Inputs
        ---------
        M:
            --description: a set of POVM.
            --type: list of matrix
            
        W:
            --description: weight matrix.
            --type: matrix

        """
        if self.dynamics_type != "dynamics":
            raise ValueError("{!r} is not a valid type for dynamics, supported type is \
                             Lindblad dynamics.".format(self.dynamics_type))
            
        if M==[]:
            M = SIC(len(self.rho0))
        M = [np.array(x, dtype=np.complex128) for x in M]

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        AD = Main.QuanEstimation.Compopt_SCopt(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.psi0,
            self.tspan,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.ctrl_bound,
            self.W,
            self.eps)
        
        if M != []:
            warnings.warn("AD is not available when target is 'CFIM'. Supported methods \
                           are 'PSO' and 'DE'.", DeprecationWarning)
        else:
            if target=="HCRB":
                warnings.warn("GRAPE is not available when the target function is HCRB. \
                       Supported methods are 'PSO', 'DE' and 'DDPG'.", DeprecationWarning)
            elif target=="QFIM" and dtype=="SLD":
                Main.QuanEstimation.SC_AD_Compopt(
                AD,
                self.max_episode,
                self.epsilon,
                self.mt,
                self.vt,
                self.beta1,
                self.beta2,
                self.eps,
                self.Adam,
                self.save_file)
                self.load_save_state()
            elif target=="QFIM" and dtype=="RLD":
                pass #### to be done
            elif target=="QFIM" and dtype=="LLD":
                pass #### to be done
            else:
                raise ValueError("Please enter the correct values for target and dtype.\
                                  Supported target are 'QFIM', 'CFIM' and 'HCRB',  \
                                  supported dtype are 'SLD', 'RLD' and 'LLD'.") 
            