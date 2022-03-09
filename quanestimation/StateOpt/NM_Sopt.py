from julia import Main
import warnings
import numpy as np
import quanestimation.StateOpt.StateStruct as State
from quanestimation.Common.common import SIC

class NM_Sopt(State.StateSystem):
    def __init__(
        self,
        save_file=False,
        state_num=10,
        psi0=[],
        max_episode=1000,
        ar=1.0,
        ae=2.0,
        ac=0.5,
        as0=0.5,
        seed=1234,
        load=False,
        eps=1e-8):

        State.StateSystem.__init__(self, save_file, psi0, seed, load, eps)

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

        self.state_num = state_num
        self.ini_state = self.psi
        self.max_episode = max_episode
        self.ar = ar
        self.ae = ae
        self.ac = ac
        self.as0 = as0
        self.seed = seed

    def QFIM(self, W=[], dtype="SLD"):
        """
        Description: use nelder-mead method to search the optimal initial state that maximize the
                     QFI (1/Tr(WF^{-1} with F the QFIM).

        ---------
        Inputs
        ---------
        W:
            --description: weight matrix.
            --type: matrix
        """

        if self.dynamics_type == "dynamics":
            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W
            
            if any(self.gamma):
                neldermead = Main.QuanEstimation.TimeIndepend_noise(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.decay_opt,
                    self.gamma,
                    self.W,
                    self.eps)
                if dtype == "SLD":
                    Main.QuanEstimation.QFIM_NM_Sopt(
                        neldermead,
                        self.state_num,
                        self.ini_state,
                        self.ar,
                        self.ae,
                        self.ac,
                        self.as0,
                        self.max_episode,
                        self.seed,
                        self.save_file)
                elif dtype == "RLD":
                    pass #### to be done
                elif dtype == "LLD":
                    pass #### to be done
                else:
                    raise ValueError("{!r} is not a valid value for dtype, supported \
                              values are 'SLD', 'RLD' and 'LLD'.".format(dtype))
            else:
                neldermead = Main.QuanEstimation.TimeIndepend_noiseless(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.W,
                    self.eps)
                if dtype == "SLD":
                    Main.QuanEstimation.QFIM_NM_Sopt(
                        neldermead,
                        self.state_num,
                        self.ini_state,
                        self.ar,
                        self.ae,
                        self.ac,
                        self.as0,
                        self.max_episode,
                        self.seed,
                        self.save_file)
                elif dtype == "RLD":
                    pass #### to be done
                elif dtype == "LLD":
                    pass #### to be done
                else:
                    raise ValueError("{!r} is not a valid value for dtype, supported \
                              values are 'SLD', 'RLD' and 'LLD'.".format(dtype))
        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W
            
            neldermead = Main.QuanEstimation.TimeIndepend_Kraus(self.K, self.dK, self.psi0, self.W, self.eps)
            if dtype == "SLD":
                Main.QuanEstimation.QFIM_NM_Sopt(
                    neldermead,
                    self.state_num,
                    self.ini_state,
                    self.ar,
                    self.ae,
                    self.ac,
                    self.as0,
                    self.max_episode,
                    self.seed,
                    self.save_file)
            elif dtype == "RLD":
                pass #### to be done
            elif dtype == "LLD":
                pass #### to be done
            else:
                raise ValueError("{!r} is not a valid value for dtype, supported \
                                  values are 'SLD', 'RLD' and 'LLD'.".format(dtype))

        self.load_save()

    def CFIM(self, M=[], W=[]):
        """
        Description: use nelder-mead method to search the optimal initial state that maximize the
                     CFI (1/Tr(WF^{-1} with F the CFIM).

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

        if M==[]:
            M = SIC(len(self.psi0))
        M = [np.array(x, dtype=np.complex128) for x in M]
        
        if self.dynamics_type == "dynamics":
            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W
            
            if any(self.gamma):
                neldermead = Main.QuanEstimation.TimeIndepend_noise(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.decay_opt,
                    self.gamma,
                    self.W,
                    self.eps)
                Main.QuanEstimation.CFIM_NM_Sopt(
                    M,
                    neldermead,
                    self.state_num,
                    self.ini_state,
                    self.ar,
                    self.ae,
                    self.ac,
                    self.as0,
                    self.max_episode,
                    self.seed,
                    self.save_file)
            else:
                neldermead = Main.QuanEstimation.TimeIndepend_noiseless(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.W,
                    self.eps)
                Main.QuanEstimation.CFIM_NM_Sopt(
                    M,
                    neldermead,
                    self.state_num,
                    self.ini_state,
                    self.ar,
                    self.ae,
                    self.ac,
                    self.as0,
                    self.max_episode,
                    self.seed,
                    self.save_file)

        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W
            neldermead = Main.QuanEstimation.TimeIndepend_Kraus(self.K, self.dK, self.psi0, self.W, self.eps)
            Main.QuanEstimation.CFIM_NM_Sopt(
                M,
                neldermead,
                self.state_num,
                self.ini_state,
                self.ar,
                self.ae,
                self.ac,
                self.as0,
                self.max_episode,
                self.seed,
                self.save_file)

        self.load_save()

    def HCRB(self, W=[]):
        """
        Description: use nelder-mead method to search the optimal initial state that maximize the HCRB.

        ---------
        Inputs
        ---------
        W:
            --description: weight matrix.
            --type: matrix
        """
        
        if self.dynamics_type == "dynamics":
            if W == []:
                W = np.eye(self.Hamiltonian_derivative)
            self.W = W
            
            if len(self.Hamiltonian_derivative) == 1:
                warnings.warn("In single parameter scenario, HCRB is equivalent to QFI. Please \
                               choose QFIM as the target function for control optimization",\
                               DeprecationWarning)
            else:
                if any(self.gamma):
                    neldermead = Main.QuanEstimation.TimeIndepend_noise(
                        self.freeHamiltonian,
                        self.Hamiltonian_derivative,
                        self.psi0,
                        self.tspan,
                        self.decay_opt,
                        self.gamma,
                        self.W,
                        self.eps)
                    Main.QuanEstimation.HCRB_NM_Sopt(
                        neldermead,
                        self.state_num,
                        self.ini_state,
                        self.ar,
                        self.ae,
                        self.ac,
                        self.as0,
                        self.max_episode,
                        self.seed,
                        self.save_file)
                else:
                    neldermead = Main.QuanEstimation.TimeIndepend_noiseless(
                        self.freeHamiltonian,
                        self.Hamiltonian_derivative,
                        self.psi0,
                        self.tspan,
                        self.W,
                        self.eps)
                    Main.QuanEstimation.HCRB_NM_Sopt(
                        neldermead,
                        self.state_num,
                        self.ini_state,
                        self.ar,
                        self.ae,
                        self.ac,
                        self.as0,
                        self.max_episode,
                        self.seed,
                        self.save_file)

        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W
            
            neldermead = Main.QuanEstimation.TimeIndepend_Kraus(self.K, self.dK, self.psi0, self.W, self.eps)
            Main.QuanEstimation.HCRB_NM_Sopt(
                neldermead,
                self.state_num,
                self.ini_state,
                self.ar,
                self.ae,
                self.ac,
                self.as0,
                self.max_episode,
                self.seed,
                self.save_file)

        self.load_save()
