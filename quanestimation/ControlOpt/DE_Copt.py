import numpy as np
import warnings
from julia import Main
import quanestimation.ControlOpt.ControlStruct as Control
from quanestimation.Common.common import SIC

class DE_Copt(Control.ControlSystem):
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
        popsize=10,
        ctrl0=[],
        max_episode=1000,
        c=1.0,
        cr=0.5,
        seed=1234,
        load=False,
        eps=1e-8):

        Control.ControlSystem.__init__(
            self, tspan, rho0, H0, Hc, dH, decay, ctrl_bound, save_file, ctrl0, load, eps)

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
        if ctrl0 == []:
            ini_population = [np.array(self.control_coefficients)]
        else:
            ini_population = ctrl0

        self.popsize = popsize
        self.ini_population = ini_population
        self.max_episode = max_episode
        self.c = c
        self.cr = cr
        self.seed = seed

    def QFIM(self, W=[], dtype="SLD"):
        """
        Description: use differential evolution algorithm to update the control coefficients that maximize the
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
        
        diffevo = Main.QuanEstimation.DE_Copt(
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
            self.eps)
        if dtype == "SLD":
            Main.QuanEstimation.QFIM_DE_Copt(
                diffevo,
                self.popsize,
                self.ini_population,
                self.c,
                self.cr,
                self.seed,
                self.max_episode,
                self.save_file)
        elif dtype == "RLD":
            pass #### to be done
        elif dtype == "LLD":
            pass #### to be done
        else:
            raise ValueError("{!r} is not a valid value for dtype, supported \
                             values are 'SLD', 'RLD' and 'LLD'.".format(dtype))

    def CFIM(self, M=[], W=[]):

        """
        Description: use differential evolution algorithm to update the control coefficients that maximize the
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
        
        diffevo = Main.QuanEstimation.DE_Copt(
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
            self.eps)
        Main.QuanEstimation.CFIM_DE_Copt(
            M,
            diffevo,
            self.popsize,
            self.ini_population,
            self.c,
            self.cr,
            self.seed,
            self.max_episode,
            self.save_file)

    def HCRB(self, W=[]):
        """
        Description: use differential evolution algorithm to update the control coefficients that maximize the
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
            warnings.warn("In single parameter scenario, HCRB is equivalent to QFI. \
                           Please choose QFIM as the target function for control optimization",\
                           DeprecationWarning)
        else:

            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W

            diffevo = Main.QuanEstimation.DE_Copt(
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
                self.eps)
            Main.QuanEstimation.HCRB_DE_Copt(
                diffevo,
                self.popsize,
                self.ini_population,
                self.c,
                self.cr,
                self.seed,
                self.max_episode,
                self.save_file)

    def mintime(self, f, W=[], M=[], method="binary", target="QFIM", dtype="SLD"):
        if len(self.Hamiltonian_derivative) > 1:
            f = 1 / f
            
        if M==[]:
            M = SIC(len(self.rho0))
        M = [np.array(x, dtype=np.complex128) for x in M]
        
        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W
        
        diffevo = Main.QuanEstimation.DE_Copt(
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
            self.eps)

        if not (method == "binary" or method == "forward"):
            raise ValueError("{!r} is not a valid value for method, supported \
                             values are 'binary' and 'forward'.".format(method))
            
        if M != []:
            Main.QuanEstimation.mintime(
                Main.eval("Val{:" + method + "}()"),
                "CFIM_DE_Copt",
                diffevo,
                f,
                M,
                self.popsize,
                self.ini_population,
                self.c,
                self.cr,
                self.seed,
                self.max_episode)
        else:
            if target == "HCRB":
                if len(self.Hamiltonian_derivative) == 1:
                    warnings.warn("In single parameter scenario, HCRB is equivalent to QFI. Please \
                                   choose QFIM as the target function for control optimization",\
                                   DeprecationWarning)
                else:
                    Main.QuanEstimation.mintime(
                        Main.eval("Val{:" + method + "}()"),
                        "HCRB_DE_Copt",
                        diffevo,
                        f,
                        self.popsize,
                        self.ini_population,
                        self.c,
                        self.cr,
                        self.seed,
                        self.max_episode)
            elif target=="QFIM" and dtype=="SLD":
                Main.QuanEstimation.mintime(
                    Main.eval("Val{:" + method + "}()"),
                    "QFIM_DE_Copt",
                    diffevo,
                    f,
                    self.popsize,
                    self.ini_population,
                    self.c,
                    self.cr,
                    self.seed,
                    self.max_episode)
            elif target=="QFIM" and dtype=="RLD":
                pass #### to be done
            elif target=="QFIM" and dtype=="LLD":
                pass #### to be done
            else:
                raise ValueError("Please enter the correct values for target and dtype.\
                                  Supported target are 'QFIM', 'CFIM' and 'HCRB',  \
                                  supported dtype are 'SLD', 'RLD' and 'LLD'.") 
                