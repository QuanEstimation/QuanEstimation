import numpy as np
import warnings
from julia import Main
import quanestimation.ControlOpt.ControlStruct as Control
from quanestimation.Common.common import SIC


class DE_Copt(Control.ControlSystem):
    def __init__(
        self,
        save_file=False,
        popsize=10,
        ctrl0=[],
        max_episode=1000,
        c=1.0,
        cr=0.5,
        seed=1234,
        load=False,
        eps=1e-8,
    ):

        Control.ControlSystem.__init__(self, save_file, ctrl0, load, eps)

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

        self.popsize = popsize
        self.max_episode = max_episode
        self.c = c
        self.cr = cr
        self.seed = seed

    def QFIM(self, W=[], dtype="SLD"):
        ini_population = Main.vec(self.ctrl0)
        self.alg = Main.QuanEstimation.DE(
            self.max_episode,
            self.popsize,
            ini_population,
            self.c,
            self.cr,
            self.seed,
        )

        super().QFIM(W, dtype)

    def CFIM(self, M=[], W=[]):
        ini_population = Main.vec(self.ctrl0)
        self.alg = Main.QuanEstimation.DE(
            self.max_episode,
            self.popsize,
            ini_population,
            self.c,
            self.cr,
            self.seed,
        )

        super().CFIM(M, W)

    def HCRB(self, W=[]):
        ini_population = Main.vec(self.ctrl0)
        self.alg = Main.QuanEstimation.DE(
            self.max_episode,
            self.popsize,
            ini_population,
            self.c,
            self.cr,
            self.seed,
        )
        
        super().HCRB(W)

    ## FIXME: mintime
    def mintime(self, f, W=[], M=[], method="binary", target="QFIM", dtype="SLD"):
        if len(self.Hamiltonian_derivative) > 1:
            f = 1 / f

        if M == []:
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
                "CFIM_DE_Copt",
                diffevo,
                f,
                M,
                self.popsize,
                Main.vec(self.ctrl0),
                self.c,
                self.cr,
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
                        "HCRB_DE_Copt",
                        diffevo,
                        f,
                        self.popsize,
                        Main.vec(self.ctrl0),
                        self.c,
                        self.cr,
                        self.seed,
                        self.max_episode,
                    )
            elif target == "QFIM" and dtype == "SLD":
                Main.QuanEstimation.mintime(
                    Main.eval("Val{:" + method + "}()"),
                    "QFIM_DE_Copt",
                    diffevo,
                    f,
                    self.popsize,
                    Main.vec(self.ctrl0),
                    self.c,
                    self.cr,
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
