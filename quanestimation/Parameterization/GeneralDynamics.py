import numpy as np
import warnings
import math
from quanestimation import QJL


class Lindblad:
    r"""
    Class for simulating quantum dynamics governed by the Lindblad master equation.

    The dynamics of a density matrix is described by the Lindblad master equation:
    \begin{align}
    \partial_t\rho &=\mathcal{L}\rho \nonumber \\
    &=-i[H,\rho]+\sum_i \gamma_i\left(\Gamma_i\rho\Gamma^{\dagger}_i-\frac{1}{2}
    \left\{\rho,\Gamma^{\dagger}_i \Gamma_i \right\}\right),
    \end{align}
    where:
        - $\rho$ is the evolved density matrix
        - $H$ is the Hamiltonian of the system
        - $\Gamma_i$ are the decay operators
        - $\gamma_i$ are the corresponding decay rates

    ## Attributes
    **tspan** : array  
        Time points for the evolution.

    **rho0** : matrix  
        Initial state (density matrix).

    **H0** : matrix or list  
        Free Hamiltonian. It is a matrix when time-independent, or a list of matrices 
        (with length equal to `tspan`) when time-dependent.

    **dH** : list  
        Derivatives of the free Hamiltonian with respect to the unknown parameters.  
        Each element is a matrix representing the partial derivative with respect to 
        one parameter. For example, `dH[0]` is the derivative with respect to the 
        first parameter.

    **decay** : list  
        Decay operators and corresponding decay rates. Input format:  
        `decay=[[Γ₁, γ₁], [Γ₂, γ₂], ...]`  
        where Γ₁, Γ₂ are decay operators and γ₁, γ₂ are the corresponding decay rates.

    **Hc** : list  
        Control Hamiltonians.

    **ctrl** : list of arrays  
        Control coefficients for each control Hamiltonian.
    """

    def __init__(self, tspan, rho0, H0, dH, decay=[], Hc=[], ctrl=[]):
        
        self.tspan = tspan
        self.rho0 = np.array(rho0, dtype=np.complex128)

        if type(H0) == np.ndarray:
            self.freeHamiltonian = np.array(H0, dtype=np.complex128)
        else:
            self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0]

        if type(dH[0]) != np.ndarray:
            raise TypeError("The derivative of Hamiltonian should be a list!")

        if not dH:
            dH = [np.zeros((len(self.rho0), len(self.rho0)))]
        self.Hamiltonian_derivative = [np.array(x, dtype=np.complex128) for x in dH]

        if not decay:
            decay_opt = [np.zeros((len(self.rho0), len(self.rho0)))]
            self.gamma = [0.0]
        else:
            decay_opt = [decay[i][0] for i in range(len(decay))]
            self.gamma = [decay[i][1] for i in range(len(decay))]
        self.decay_opt = [np.array(x, dtype=np.complex128) for x in decay_opt]

        if not Hc:
            Hc = [np.zeros((len(self.rho0), len(self.rho0)))]
            ctrl = [np.zeros(len(self.tspan) - 1)]
            self.control_Hamiltonian = [np.array(x, dtype=np.complex128) for x in Hc]
            self.control_coefficients = ctrl
        elif not ctrl:
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
        r"""
        Calculate the density matrix and its derivatives using the matrix exponential method.

        The density matrix at the $j$th time interval is obtained by:
        \begin{align}
        \rho_j = e^{\Delta t \mathcal{L}} \rho_{j-1}
        \end{align}
        where $\Delta t$ is the time interval and $\rho_{j-1}$ is the density matrix 
        at the previous time step.

        The derivative $\partial_{\textbf{x}}\rho_j$ is calculated as:
        \begin{align}
        \partial_{\textbf{x}}\rho_j = \Delta t (\partial_{\textbf{x}}\mathcal{L}) \rho_j
        + e^{\Delta t \mathcal{L}} (\partial_{\textbf{x}}\rho_{j-1})
        \end{align}

        ## Returns
        **rho** : list of matrices  
            Density matrices at each time point in `tspan`.

        **drho** : list of lists of matrices  
            Derivatives of the density matrices with respect to the unknown parameters.  
            `drho[i][j]` is the derivative of the density matrix at the i-th time point 
            with respect to the j-th parameter.
        """

        rho, drho = QJL.expm_py(
            self.tspan,
            self.rho0,
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
        )
        rho = [np.array(rho_i) for rho_i in rho]
        drho = [[np.array(drho_ij) for drho_ij in drho_i] for drho_i in drho]
        
        return rho, drho

    def ode(self):
        r"""
        Calculate the density matrix and its derivatives using an ODE solver.

        The density matrix at the $j$th time interval is obtained by:
        \begin{align}
        \rho_j = e^{\Delta t \mathcal{L}} \rho_{j-1}
        \end{align}
        where $\Delta t$ is the time interval and $\rho_{j-1}$ is the density matrix 
        at the previous time step.

        The derivative $\partial_{\textbf{x}}\rho_j$ is calculated as:
        \begin{align}
        \partial_{\textbf{x}}\rho_j = \Delta t (\partial_{\textbf{x}}\mathcal{L}) \rho_j
        + e^{\Delta t \mathcal{L}} (\partial_{\textbf{x}}\rho_{j-1})
        \end{align}

        ## Returns
        **rho** : list of matrices  
            Density matrices at each time point in `tspan`.

        **drho** : list of lists of matrices  
            Derivatives of the density matrices with respect to the unknown parameters.  
            `drho[i][j]` is the derivative of the density matrix at the i-th time point 
            with respect to the j-th parameter.
        """

        rho, drho = QJL.ode_py(
            self.tspan,
            self.rho0,
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
        )
        rho = [np.array(rho_i) for rho_i in rho]
        drho = [[np.array(drho_ij) for drho_ij in drho_i] for drho_i in drho]
        
        return rho, drho
        
    def secondorder_derivative(self, d2H):
        r"""
        Calculate the density matrix, its first derivatives, and second derivatives 
        with respect to the unknown parameters.

        The density matrix at the $j$th time interval is obtained by:
        \begin{align}
        \rho_j = e^{\Delta t \mathcal{L}} \rho_{j-1}
        \end{align}

        The first derivative $\partial_{\textbf{x}}\rho_j$ is calculated as:
        \begin{align}
        \partial_{\textbf{x}}\rho_j = \Delta t (\partial_{\textbf{x}}\mathcal{L}) \rho_j
        + e^{\Delta t \mathcal{L}} (\partial_{\textbf{x}}\rho_{j-1})
        \end{align}

        The second derivative $\partial_{\textbf{x}}^2\rho_j$ is calculated as:
        \begin{align}
        \partial_{\textbf{x}}^2\rho_j =& \Delta t (\partial_{\textbf{x}}^2\mathcal{L}) \rho_j \\
        &+ \Delta t (\partial_{\textbf{x}}\mathcal{L}) \partial_{\textbf{x}}\rho_j \\
        &+ \Delta t (\partial_{\textbf{x}}\mathcal{L}) e^{\Delta t \mathcal{L}} \partial_{\textbf{x}}\rho_{j-1} \\
        &+ e^{\Delta t \mathcal{L}} (\partial_{\textbf{x}}^2\rho_{j-1})
        \end{align}

        ## Parameters
        **d2H** : list  
            Second-order derivatives of the free Hamiltonian with respect to the unknown parameters.  
            Each element is a matrix representing the second partial derivative with respect to 
            two parameters. For example, `d2H[0]` might be the second derivative with respect to 
            the first parameter twice, or a mixed partial derivative.

        ## Returns
        **rho** : list of matrices  
            Density matrices at each time point in `tspan`.

        **drho** : list of lists of matrices  
            First derivatives of the density matrices with respect to the unknown parameters.

        **d2rho** : list of lists of matrices  
            Second derivatives of the density matrices with respect to the unknown parameters.
        """

        d2H = [np.array(x, dtype=np.complex128) for x in d2H]
        rho, drho, d2rho = QJL.secondorder_derivative(
            self.tspan,
            self.rho0,
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            d2H,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
        )
        rho = [np.array(rho_i) for rho_i in rho]
        drho = [[np.array(drho_ij) for drho_ij in drho_i] for drho_i in drho]
        #d2rho = 
        return rho, drho, d2rho
