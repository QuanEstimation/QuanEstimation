import numpy as np
from julia import Main
import warnings
import quanestimation.ControlOpt.ControlStruct as Control
from quanestimation.Common.common import SIC

class PSO_Copt(Control.ControlSystem):
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
        particle_num=10,
        ctrl0=[],
        max_episode=[1000, 100],
        c0=1.0,
        c1=2.0,
        c2=2.0,
        seed=1234,
        load=False,
        eps=1e-8):

        Control.ControlSystem.__init__(
            self, tspan, rho0, H0, Hc, dH, decay, ctrl_bound, save_file, ctrl0, load, eps)

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

        if ctrl0 == []:
            ini_particle = [np.array(self.control_coefficients)]
        else:
            ini_particle = ctrl0

        self.particle_num = particle_num
        self.ini_particle = ini_particle
        self.max_episode = max_episode
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

    def QFIM(self, W=[], dtype="SLD"):
        """
        Description: use particle swarm optimization algorithm to update the control coefficients
                     that maximize the QFI (1/Tr(WF^{-1} with F the QFIM).

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
        
        pso = Main.QuanEstimation.PSO_Copt(
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
            Main.QuanEstimation.QFIM_PSO_Copt(
                pso,
                self.max_episode,
                self.particle_num,
                self.ini_particle,
                self.c0,
                self.c1,
                self.c2,
                self.seed,
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
        Description: use particle swarm optimization algorithm to update the control coefficients
                     that maximize the CFI (1/Tr(WF^{-1} with F the CFIM).

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
        
        pso = Main.QuanEstimation.PSO_Copt(
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
        Main.QuanEstimation.CFIM_PSO_Copt(
            M,
            pso,
            self.max_episode,
            self.particle_num,
            self.ini_particle,
            self.c0,
            self.c1,
            self.c2,
            self.seed,
            self.save_file)
        
    def HCRB(self, W=[]):

        """
        Description: use particle swarm optimization algorithm to update the control coefficients
                     that maximize the HCRB.

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

            pso = Main.QuanEstimation.PSO_Copt(
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
            Main.QuanEstimation.HCRB_PSO_Copt(
                pso,
                self.max_episode,
                self.particle_num,
                self.ini_particle,
                self.c0,
                self.c1,
                self.c2,
                self.seed,
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
        
        if not (method == "binary" or method == "forward"):
            raise ValueError("{!r} is not a valid value for method, supported \
                             values are 'binary' and 'forward'.".format(method))
            
        pso = Main.QuanEstimation.PSO_Copt(
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
        
        if M != []:
            Main.QuanEstimation.mintime(
                Main.eval("Val{:" + method + "}()"),
                "CFIM_PSO_Copt",
                pso,
                f,
                M,
                self.max_episode,
                self.particle_num,
                self.ini_particle,
                self.c0,
                self.c1,
                self.c2,
                self.seed)
        else:
            if target == "HCRB":
                if len(self.Hamiltonian_derivative) == 1:
                    warnings.warn("In single parameter scenario, HCRB is equivalent to QFI. Please \
                                   choose QFIM as the target function for control optimization",\
                                   DeprecationWarning)
                else:
                    Main.QuanEstimation.mintime(
                        Main.eval("Val{:" + method + "}()"),
                        "HCRB_PSO_Copt",
                        pso,
                        f,
                        self.max_episode,
                        self.particle_num,
                        self.ini_particle,
                        self.c0,
                        self.c1,
                        self.c2,
                        self.seed)
            elif target=="QFIM" and dtype=="SLD":
                Main.QuanEstimation.mintime(
                    Main.eval("Val{:" + method + "}()"),
                    "QFIM_PSO_Copt",
                    pso,
                    f,
                    self.max_episode,
                    self.particle_num,
                    self.ini_particle,
                    self.c0,
                    self.c1,
                    self.c2,
                    self.seed)
            elif target=="QFIM" and dtype=="RLD":
                pass #### to be done
            elif target=="QFIM" and dtype=="LLD":
                pass #### to be done
            else:
                raise ValueError("Please enter the correct values for target and dtype.\
                                  Supported target are 'QFIM', 'CFIM' and 'HCRB',  \
                                  supported dtype are 'SLD', 'RLD' and 'LLD'.") 
            