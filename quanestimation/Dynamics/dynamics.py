import numpy as np
import warnings
import math
from julia import Main


class Lindblad:
    """
    General dynamics of density matrices in the form of time local Lindblad master equation.
    {\partial_t \rho} = -i[H, \rho] + \sum_n {\gamma_n} {Ln.rho.Ln^{\dagger}
                 -0.5(rho.Ln^{\dagger}.Ln+Ln^{\dagger}.Ln.rho)}.
    """

    def __init__(self, tspan, rho0, H0, dH, decay=[], Hc=[], ctrl=[]):
        """
        ----------
        Inputs
        ----------
        tspan:
           --description: time series.
           --type: array

        rho0:
           --description: initial state (density matrix).
           --type: matrix

        H0:
           --description: free Hamiltonian.
           --type: matrix

        Hc:
           --description: control Hamiltonian.
           --type: list (of matrix)

        dH:
           --description: derivatives of Hamiltonian on all parameters to
                          be estimated. For example, dH[0] is the derivative
                          vector on the first parameter.
           --type: list (of matrix)

        ctrl:
           --description: control coefficients.
           --type: list (of array)

        decay:
           --description: decay operators and the corresponding decay rates.
                          decay[0] represent a list of decay operators and
                          decay[1] represent the corresponding decay rates.
           --type: list
        """
        self.tspan = tspan
        self.rho0 = np.array(rho0, dtype=np.complex128)

        if type(H0) == np.ndarray:
            self.freeHamiltonian = np.array(H0, dtype=np.complex128)
        else:
            self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0]

        if type(dH[0]) != np.ndarray:
            raise TypeError("The derivative of Hamiltonian should be a list!")

        if dH == []:
            dH = [np.zeros((len(self.rho0), len(self.rho0)))]
        self.Hamiltonian_derivative = [np.array(x, dtype=np.complex128) for x in dH]

        if decay == []:
            decay_opt = [np.zeros((len(self.rho0), len(self.rho0)))]
            self.gamma = [0.0]
        else:
            decay_opt = [decay[i][0] for i in range(len(decay))]
            self.gamma = [decay[i][1] for i in range(len(decay))]
        self.decay_opt = [np.array(x, dtype=np.complex128) for x in decay_opt]

        if Hc == []:
            Hc = [np.zeros((len(self.rho0), len(self.rho0)))]
            ctrl = [np.zeros(len(self.tspan) - 1)]
            self.control_Hamiltonian = [np.array(x, dtype=np.complex128) for x in Hc]
            self.control_coefficients = ctrl
        elif ctrl == []:
            ctrl = [np.zeros(len(self.tspan) - 1) for j in range(len(Hc))]
            self.control_Hamiltonian = Hc
            self.control_coefficients = ctrl
        else:
            ctrl_length = len(ctrl)
            ctrlnum = len(Hc)
            if ctrlnum < ctrl_length:
                raise TypeError(
                    "There are %d control Hamiltonians but %d coefficients sequences: \
                                too many coefficients sequences"
                    % (ctrlnum, ctrl_length)
                )
            elif ctrlnum > ctrl_length:
                warnings.warn(
                    "Not enough coefficients sequences: there are %d control Hamiltonians \
                            but %d coefficients sequences. The rest of the control sequences are\
                            set to be 0."
                    % (ctrlnum, ctrl_length),
                    DeprecationWarning,
                )

            number = math.ceil((len(self.tspan) - 1) / len(ctrl[0]))
            if len(self.tspan) - 1 % len(ctrl[0]) != 0:
                tnum = number * len(ctrl[0])
                self.tspan = np.linspace(self.tspan[0], self.tspan[-1], tnum + 1)
            self.control_Hamiltonian = Hc
            self.control_coefficients = ctrl

    def expm(self):
        rho, drho = Main.QuanEstimation.expm(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.rho0,
            self.tspan,
            self.decay_opt,
            self.gamma,
        )
        return rho, drho

    def secondorder_derivative(self, d2H):
        d2H = [np.array(x, dtype=np.complex128) for x in d2H]
        rho, drho, d2rho = Main.QuanEstimation.secondorder_derivative(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            d2H,
            self.rho0,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.tspan,
        )
        return rho, drho, d2rho
