from julia import Main
import warnings
import numpy as np
import quanestimation.StateOpt.StateStruct as State
from quanestimation.Common.common import SIC


class DDPG_Sopt(State.StateSystem):
    def __init__(
        self,
        save_file=False,
        max_episode=500,
        layer_num=3,
        layer_dim=200,
        seed=1234,
        psi0=[],
        load=False,
        eps=1e-8,
    ):

        State.StateSystem.__init__(self, save_file, psi0, seed, load, eps)
        """
        ----------
        Inputs
        ----------
        layer_num:
            --description: the number of layers (including the input and output layer).
            --type: int

        layer_dim:
            --description: the number ofP neurons in the hidden layer.
            --type: int
        
        seed:
            --description: random seed.
            --type: int
        """

        self.layer_num = layer_num
        self.layer_dim = layer_dim
        self.max_episode = max_episode
        self.seed = seed

    def QFIM(self, W=[], dtype="SLD"):
        """
        Description: use DDPG algorithm to search the optimal initial state that maximize the
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
                DDPG = Main.QuanEstimation.TimeIndepend_noise(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.decay_opt,
                    self.gamma,
                    self.W,
                    self.eps,
                )
                if dtype == "SLD":
                    Main.QuanEstimation.QFIM_DDPG_Sopt(
                        DDPG,
                        self.layer_num,
                        self.layer_dim,
                        self.seed,
                        self.max_episode,
                        self.save_file,
                    )
                elif dtype == "RLD":
                    pass  #### to be done
                elif dtype == "LLD":
                    pass  #### to be done
                else:
                    raise ValueError(
                        "{!r} is not a valid value for dtype, supported \
                              values are 'SLD', 'RLD' and 'LLD'.".format(
                            dtype
                        )
                    )
            else:
                DDPG = Main.QuanEstimation.TimeIndepend_noiseless(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.W,
                    self.eps,
                )
                if dtype == "SLD":
                    Main.QuanEstimation.QFIM_DDPG_Sopt(
                        DDPG,
                        self.layer_num,
                        self.layer_dim,
                        self.seed,
                        self.max_episode,
                        self.save_file,
                    )
                elif dtype == "RLD":
                    pass  #### to be done
                elif dtype == "LLD":
                    pass  #### to be done
                else:
                    raise ValueError(
                        "{!r} is not a valid value for dtype, supported \
                              values are 'SLD', 'RLD' and 'LLD'.".format(
                            dtype
                        )
                    )

        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W

            DDPG = Main.QuanEstimation.TimeIndepend_Kraus(
                self.K, self.dK, self.psi0, self.W, self.eps
            )
            if dtype == "SLD":
                Main.QuanEstimation.QFIM_DDPG_Sopt(
                    DDPG,
                    self.layer_num,
                    self.layer_dim,
                    self.seed,
                    self.max_episode,
                    self.save_file,
                )
            elif dtype == "RLD":
                pass  #### to be done
            elif dtype == "LLD":
                pass  #### to be done
            else:
                raise ValueError(
                    "{!r} is not a valid value for dtype, supported \
                                  values are 'SLD', 'RLD' and 'LLD'.".format(
                        dtype
                    )
                )

        self.load_save()

    def CFIM(self, M=[], W=[]):
        """
        Description: use DDPG algorithm to search the optimal initial state that maximize the
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
        if M == []:
            M = SIC(len(self.psi0))
        M = [np.array(x, dtype=np.complex128) for x in M]

        if self.dynamics_type == "dynamics":
            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W

            if any(self.gamma):
                DDPG = Main.QuanEstimation.TimeIndepend_noise(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.decay_opt,
                    self.gamma,
                    self.W,
                    self.eps,
                )
                Main.QuanEstimation.CFIM_DDPG_Sopt(
                    M,
                    DDPG,
                    self.layer_num,
                    self.layer_dim,
                    self.seed,
                    self.max_episode,
                    self.save_file,
                )
            else:
                DDPG = Main.QuanEstimation.TimeIndepend_noiseless(
                    self.freeHamiltonian,
                    self.Hamiltonian_derivative,
                    self.psi0,
                    self.tspan,
                    self.W,
                    self.eps,
                )
                Main.QuanEstimation.CFIM_DDPG_Sopt(
                    M,
                    DDPG,
                    self.layer_num,
                    self.layer_dim,
                    self.seed,
                    self.max_episode,
                    self.save_file,
                )

        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W

            DDPG = Main.QuanEstimation.TimeIndepend_Kraus(
                self.K, self.dK, self.psi0, self.W, self.eps
            )
            Main.QuanEstimation.CFIM_DDPG_Sopt(
                M,
                DDPG,
                self.layer_num,
                self.layer_dim,
                self.seed,
                self.max_episode,
                self.save_file,
            )

        self.load_save()

    def HCRB(self, W=[]):
        """
        Description: use DDPG algorithm to search the optimal initial state that maximize the
                     HCRB.

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

            if len(self.Hamiltonian_derivative) == 1:
                warnings.warn(
                    "In single parameter scenario, HCRB is equivalent to QFI. Please \
                               choose QFIM as the target function for control optimization",
                    DeprecationWarning,
                )
            else:
                if any(self.gamma):
                    DDPG = Main.QuanEstimation.TimeIndepend_noise(
                        self.freeHamiltonian,
                        self.Hamiltonian_derivative,
                        self.psi0,
                        self.tspan,
                        self.decay_opt,
                        self.gamma,
                        self.W,
                        self.eps,
                    )
                    Main.QuanEstimation.HCRB_DDPG_Sopt(
                        DDPG,
                        self.layer_num,
                        self.layer_dim,
                        self.seed,
                        self.max_episode,
                        self.save_file,
                    )
                else:
                    DDPG = Main.QuanEstimation.TimeIndepend_noiseless(
                        self.freeHamiltonian,
                        self.Hamiltonian_derivative,
                        self.psi0,
                        self.tspan,
                        self.W,
                        self.eps,
                    )
                    Main.QuanEstimation.HCRB_DDPG_Sopt(
                        DDPG,
                        self.layer_num,
                        self.layer_dim,
                        self.seed,
                        self.max_episode,
                        self.save_file,
                    )

        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W

            DDPG = Main.QuanEstimation.TimeIndepend_Kraus(
                self.K, self.dK, self.psi0, self.W, self.eps
            )
            Main.QuanEstimation.HCRB_DDPG_Sopt(
                DDPG,
                self.layer_num,
                self.layer_dim,
                self.seed,
                self.max_episode,
                self.save_file,
            )

        self.load_save()
