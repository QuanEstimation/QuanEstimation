import numpy as np
import warnings
from julia import Main
import quanestimation.ControlOpt.ControlStruct as Control
from quanestimation.Common.common import SIC

class GRAPE_Copt(Control.ControlSystem):
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
        Adam=True,
        ctrl0=[],
        max_episode=300,
        epsilon=0.01,
        beta1=0.90,
        beta2=0.99,
        load=False,
        eps=1e-8,
        auto=True):

        Control.ControlSystem.__init__(
            self, tspan, rho0, H0, Hc, dH, decay, ctrl_bound, save_file, ctrl0, load, eps)

        """
        ----------
        Inputs
        ----------
        auto:
            --description: True: use autodifferential to calculate the gradient.
                                  False: calculate the gradient with analytical method.
            --type: bool (True or False)

        Adam:
            --description: whether to use Adam to update the controls.
            --type: bool (True or False)

        ctrl0:
           --description: initial guess of controls.
           --type: array

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

        self.auto = auto
        self.Adam = Adam
        self.max_episode = max_episode
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0

    def QFIM(self, W=[], dtype="SLD"):
        """
        Description: use auto-GRAPE (GRAPE) algorithm to update the control coefficients that maximize the
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
        
        grape = Main.QuanEstimation.GRAPE_Copt(
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
            self.mt,
            self.vt,
            self.epsilon,
            self.beta1,
            self.beta2,
            self.eps)
        
        if self.auto == True:
            if dtype == "SLD":
                Main.QuanEstimation.QFIM_autoGRAPE_Copt(grape, self.max_episode, self.Adam, self.save_file)
            elif dtype == "RLD":
                pass #### to be done
            elif dtype == "LLD":
                pass #### to be done
            else:
                raise ValueError("{!r} is not a valid value for dtype, supported \
                                 values are 'SLD', 'RLD' and 'LLD'.".format(dtype))
        else:
            if (len(self.tspan) - 1) != len(self.control_coefficients[0]):
                warnings.warn("GRAPE is not available when the length of each control is not \
                               equal to the length of time, and is replaced by auto-GRAPE.",
                               DeprecationWarning)
                if dtype == "SLD":
                    Main.QuanEstimation.QFIM_autoGRAPE_Copt(
                        grape, self.max_episode, self.Adam, self.save_file)
                elif dtype == "RLD":
                    pass #### to be done
                elif dtype == "LLD":
                    pass #### to be done
                else:
                    raise ValueError("{!r} is not a valid value for dtype, supported \
                                     values are 'SLD', 'RLD' and 'LLD'.".format(dtype))
            else:
                if dtype == "SLD":
                    Main.QuanEstimation.QFIM_GRAPE_Copt(grape, self.max_episode, self.Adam, self.save_file)
                elif dtype == "RLD":
                    pass #### to be done
                elif dtype == "LLD":
                    pass #### to be done
                else:
                    raise ValueError("{!r} is not a valid value for dtype, supported \
                                     values are 'SLD', 'RLD' and 'LLD'.".format(dtype))

    def CFIM(self, M=[], W=[]):
        """
        Description: use auto-GRAPE (GRAPE) algorithm to update the control coefficients that maximize the
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
            M = SIC(len(self.rho0))
        M = [np.array(x, dtype=np.complex128) for x in M]
        
        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W
        
        grape = Main.QuanEstimation.GRAPE_Copt(
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
            self.mt,
            self.vt,
            self.epsilon,
            self.beta1,
            self.beta2,
            self.eps)
        
        if self.auto == True:
            Main.QuanEstimation.CFIM_autoGRAPE_Copt(M, grape, self.max_episode, self.Adam, self.save_file)
        else:
            if (len(self.tspan) - 1) != len(self.control_coefficients[0]):
                warnings.warn("GRAPE is not available when the length of each control is not equal to \
                               the length of time, and is replaced by auto-GRAPE.", DeprecationWarning)
                Main.QuanEstimation.CFIM_autoGRAPE_Copt(M, grape, self.max_episode, self.Adam, self.save_file)
            else:
                Main.QuanEstimation.CFIM_GRAPE_Copt(M, grape, self.max_episode, self.Adam, self.save_file)

    def HCRB(self, W=[]):
        warnings.warn("GRAPE is not available when the target function is HCRB. \
                       Supported methods are 'PSO', 'DE' and 'DDPG'.", DeprecationWarning)

    def mintime(self, f, W=[], M=[], method="binary", target="QFIM", dtype="SLD"):
        if len(self.Hamiltonian_derivative) > 1:
            f = 1 / f
            
        if M==[]:
            M = SIC(len(self.rho0))
        M = [np.array(x, dtype=np.complex128) for x in M]
        
        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W
        
        grape = Main.QuanEstimation.GRAPE_Copt(
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
            self.mt,
            self.vt,
            self.epsilon,
            self.beta1,
            self.beta2,
            self.eps)

        if not (method == "binary" or method == "forward"):
            raise ValueError("{!r} is not a valid value for method, supported \
                             values are 'binary' and 'forward'.".format(method))
        if M != []:
            if self.auto:
                Main.QuanEstimation.mintime(
                    Main.eval("Val{:" + method + "}()"),
                    "CFIM_autoGRAPE_Copt",
                    grape,
                    f,
                    M,
                    self.max_episode,
                    self.Adam)
            else:
                Main.QuanEstimation.mintime(
                    Main.eval("Val{:" + method + "}()"),
                    "CFIM_GRAPE_Copt",
                    grape,
                    f,
                    M,
                    self.max_episode,
                    self.Adam)
        else:
            if target == "HCRB":
                warnings.warn("GRAPE is not available when the target function is HCRB. \
                       Supported methods are 'PSO', 'DE' and 'DDPG'.", DeprecationWarning)
            elif target=="QFIM" and dtype=="SLD":
                if self.auto:
                    Main.QuanEstimation.mintime(
                        Main.eval("Val{:" + method + "}()"),
                        "QFIM_autoGRAPE_Copt",
                        grape,
                        f,
                        self.max_episode,
                        self.Adam)
                else:
                    Main.QuanEstimation.mintime(
                        Main.eval("Val{:" + method + "}()"),
                        "QFIM_GRAPE_Copt",
                        grape,
                        f,
                        self.max_episode,
                        self.Adam)
            elif target=="QFIM" and dtype=="RLD":
                pass #### to be done
            elif target=="QFIM" and dtype=="LLD":
                pass #### to be done
            else:
                raise ValueError("Please enter the correct values for target and dtype.\
                                  Supported target are 'QFIM', 'CFIM' and 'HCRB',  \
                                  supported dtype are 'SLD', 'RLD' and 'LLD'.") 
                