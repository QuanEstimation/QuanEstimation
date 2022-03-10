import numpy as np
import warnings
from julia import Main
import quanestimation.ControlOpt.ControlStruct as Control
from quanestimation.Common.common import SIC


class DDPG_Copt(Control.ControlSystem):
    def __init__(
        self,
        tspan,
        rho0,
        H0,
        dH,
        Hc,
        decay=[],
        ctrl_bound=[],
        save_file=False,
        max_episode=500,
        layer_num=3,
        layer_dim=200,
        seed=1234,
        ctrl0=[],
        load=False,
        eps=1e-8,
    ):

        Control.ControlSystem.__init__(
            self,
            tspan,
            rho0,
            H0,
            Hc,
            dH,
            decay,
            ctrl_bound,
            save_file,
            ctrl0,
            load,
            eps,
        )

        """                                           
        ----------
        Inputs
        ----------
        max_episode:
            --description: max number of the training episodes.
            --type: int

        layer_num:
            --description: the number of layers (including the input and output layer).
            --type: int

        layer_dim:
            --description: the number of neurons in the hidden layer.
            --type: int
        
        seed:
            --description: random seed.
            --type: int

        """
        self.ctrl_interval = (len(self.tspan) - 1) // len(self.control_coefficients[0])
        self.layer_num = layer_num
        self.layer_dim = layer_dim
        self.max_episode = max_episode
        self.seed = seed

    def QFIM(self, W=[], dtype="SLD"):

        """
        Description: use DDPG algorithm to update the control coefficients that maximize the
                     QFI (1/Tr(WF^{-1} with F the QFIM).

        ---------
        Inputs
        ---------
        W:
            --description: weight matrix.
            --type: matrix
        """
        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        ddpg = Main.QuanEstimation.DDPG_Copt(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.rho0,
            self.tspan,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.ctrl_bound,
            self.W,
            self.ctrl_interval,
            len(self.rho0),
            self.eps,
        )
        if dtype == "SLD":
            Main.QuanEstimation.QFIM_DDPG_Copt(
                ddpg,
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

    def CFIM(self, M=[], W=[]):
        """
        Description: use DDPG algorithm to update the control coefficients that maximize the
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
            M = SIC(len(self.rho0))
        M = [np.array(x, dtype=np.complex128) for x in M]

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        ddpg = Main.QuanEstimation.DDPG_Copt(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.rho0,
            self.tspan,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.ctrl_bound,
            self.W,
            self.ctrl_interval,
            len(self.rho0),
            self.eps,
        )
        Main.QuanEstimation.CFIM_DDPG_Copt(
            M,
            ddpg,
            self.layer_num,
            self.layer_dim,
            self.seed,
            self.max_episode,
            self.save_file,
        )

    def HCRB(self, W=[]):
        """
        Description: use DDPG algorithm to update the control coefficients that maximize the
                     HCRB.

        ---------
        Inputs
        ---------
        W:
            --description: weight matrix.
            --type: matrix
        """
        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        if len(self.Hamiltonian_derivative) == 1:
            warnings.warn(
                "In single parameter scenario, HCRB is equivalent to QFI. \
                           Please choose QFIM as the target function for control optimization",
                DeprecationWarning,
            )
        else:
            ddpg = Main.QuanEstimation.DDPG_Copt(
                self.freeHamiltonian,
                self.Hamiltonian_derivative,
                self.rho0,
                self.tspan,
                self.decay_opt,
                self.gamma,
                self.control_Hamiltonian,
                self.control_coefficients,
                self.ctrl_bound,
                self.W,
                self.ctrl_interval,
                len(self.rho0),
                self.eps,
            )
            Main.QuanEstimation.HCRB_DDPG_Copt(
                ddpg,
                self.layer_num,
                self.layer_dim,
                self.seed,
                self.max_episode,
                self.save_file,
            )

    def mintime(self, f, W=[], M=[], method="binary", target="QFIM", dtype="SLD"):
        if len(self.Hamiltonian_derivative) > 1:
            f = 1 / f

        if M == []:
            M = SIC(len(self.rho0))
        M = [np.array(x, dtype=np.complex128) for x in M]

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        ddpg = Main.QuanEstimation.DDPG_Copt(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.rho0,
            self.tspan,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.ctrl_bound,
            self.W,
            self.ctrl_interval,
            len(self.rho0),
            self.eps,
        )

        if not (method == "binary" or method == "forward"):
            raise ValueError(
                "{!r} is not a valid value for method, supported \
                             values are 'binary' and 'forward'.".format(
                    method
                )
            )

        if M != []:
            Main.QuanEstimation.mintime(
                Main.eval("Val{:" + method + "}()"),
                "CFIM_DDPG_Copt",
                ddpg,
                f,
                M,
                self.layer_num,
                self.layer_dim,
                self.seed,
                self.max_episode,
            )
        else:
            if target == "HCRB":
                if len(self.Hamiltonian_derivative) == 1:
                    warnings.warn(
                        "In single parameter scenario, HCRB is equivalent to QFI. Please \
                                   choose QFIM as the target function for control optimization",
                        DeprecationWarning,
                    )
                else:
                    Main.QuanEstimation.mintime(
                        Main.eval("Val{:" + method + "}()"),
                        "HCRB_DDPG_Copt",
                        ddpg,
                        f,
                        self.layer_num,
                        self.layer_dim,
                        self.seed,
                        self.max_episode,
                    )
            elif target == "QFIM" and dtype == "SLD":
                Main.QuanEstimation.mintime(
                    Main.eval("Val{:" + method + "}()"),
                    "QFIM_DDPG_Copt",
                    ddpg,
                    f,
                    self.layer_num,
                    self.layer_dim,
                    self.seed,
                    self.max_episode,
                )
            elif target == "QFIM" and dtype == "RLD":
                pass  #### to be done
            elif target == "QFIM" and dtype == "LLD":
                pass  #### to be done
            else:
                raise ValueError(
                    "Please enter the correct values for target and dtype.\
                                  Supported target are 'QFIM', 'CFIM' and 'HCRB',  \
                                  supported dtype are 'SLD', 'RLD' and 'LLD'."
                )
