from julia import Main
import warnings
import numpy as np
import quanestimation.StateOpt.StateStruct as State
from quanestimation.Common.common import SIC

class AD_Sopt(State.StateSystem):
    def __init__(
        self,
        save_file=False,
        Adam=False,
        psi0=[],
        max_episode=300,
        epsilon=0.01,
        beta1=0.90,
        beta2=0.99,
        seed=1234,
        load=False,
        eps=1e-8):

        State.StateSystem.__init__(self, save_file, psi0, seed, load, eps)

        """
        ----------
        Inputs
        ----------
        Adam:
            --description: whether to use Adam to update the controls.
            --type: bool (True or False)

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

        eps:
            --description: calculation eps.
            --type: float

        """

        self.Adam = Adam
        self.max_episode = max_episode
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0

    def QFIM(self, W=[], dtype="SLD"):
        """
        Description: use autodifferential algorithm to search the optimal initial state that maximize the
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
                AD = Main.QuanEstimation.TimeIndepend_noise(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.decay_opt,
                    self.gamma,
                    self.W,
                    self.eps)
                if dtype=="SLD":
                    Main.QuanEstimation.QFIM_AD_Sopt(
                        AD,
                        self.mt,
                        self.vt,
                        self.epsilon,
                        self.beta1,
                        self.beta2,
                        self.max_episode,
                        self.Adam,
                        self.save_file)
                elif dtype == "RLD":
                    pass #### to be done
                elif dtype == "LLD":
                    pass #### to be done
                else:
                    raise ValueError("{!r} is not a valid value for dtype, supported \
                              values are 'SLD', 'RLD' and 'LLD'.".format(dtype))
            else:
                AD = Main.QuanEstimation.TimeIndepend_noiseless(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.W,
                    self.eps)
                if dtype == "SLD":
                    Main.QuanEstimation.QFIM_AD_Sopt(
                        AD,
                        self.mt,
                        self.vt,
                        self.epsilon,
                        self.beta1,
                        self.beta2,
                        self.max_episode,
                        self.Adam,
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

            AD = Main.QuanEstimation.TimeIndepend_Kraus(self.K, self.dK, self.psi0, self.W, self.eps)
            if dtype == "SLD":
                Main.QuanEstimation.QFIM_AD_Sopt(
                    AD,
                    self.mt,
                    self.vt,
                    self.epsilon,
                    self.beta1,
                    self.beta2,
                    self.max_episode,
                    self.Adam,
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
        Description: use autodifferential algorithm to search the optimal initial state that maximize the
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
                AD = Main.QuanEstimation.TimeIndepend_noise(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.decay_opt,
                    self.gamma,
                    self.W,
                    self.eps)
                Main.QuanEstimation.CFIM_AD_Sopt(
                    M,
                    AD,
                    self.mt,
                    self.vt,
                    self.epsilon,
                    self.beta1,
                    self.beta2,
                    self.max_episode,
                    self.Adam,
                    self.save_file)
            else:
                AD = Main.QuanEstimation.TimeIndepend_noiseless(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.W,
                    self.eps)
                Main.QuanEstimation.CFIM_AD_Sopt(
                    M,
                    AD,
                    self.mt,
                    self.vt,
                    self.epsilon,
                    self.beta1,
                    self.beta2,
                    self.max_episode,
                    self.Adam,
                    self.save_file)
        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W

            AD = Main.QuanEstimation.TimeIndepend_Kraus(self.K, self.dK, self.psi0, self.W, self.eps)
            Main.QuanEstimation.CFIM_AD_Sopt(
                M,
                AD,
                self.mt,
                self.vt,
                self.epsilon,
                self.beta1,
                self.beta2,
                self.max_episode,
                self.Adam,
                self.save_file)

        self.load_save()

    def HCRB(self, W=[]):
        warnings.warn("AD is not available when the objective function is HCRB. \
                       Supported methods are 'PSO', 'DE', 'NM' and 'DDPG'.",\
                       DeprecationWarning)
