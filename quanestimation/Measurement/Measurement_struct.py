import numpy as np
import quanestimation.Measurement as Measure
from quanestimation.Common.common import gramschmidt

class MeasurementSystem:
    def __init__(self, tspan, rho0, H0, dH, decay, W, ini_measurement, seed, accuracy):
        
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
           --type: matrix or a list of matrix
        
        dH: 
           --description: derivatives of Hamiltonian on all parameters to
                          be estimated. For example, dH[0] is the derivative
                          vector on the first parameter.
           --type: list (of matrix)
           
        decay:
           --description: decay operators and the corresponding decay rates.
                          decay[0][0] represent the first decay operator and
                          decay[0][1] represent the corresponding decay rate.
           --type: list 
        
        ini_measurement:
           --description: a set of POVMs.
           --type: list (of vector)

        W:
            --description: weight matrix.
            --type: matrix

        accuracy:
            --description: calculation accuracy.
            --type: float
        
        """   
        self.tspan = tspan
        self.rho0 = np.array(rho0, dtype=np.complex128)

        if type(H0) == np.ndarray:
            self.freeHamiltonian = np.array(H0, dtype=np.complex128)
        else:
            self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0] 

        if type(dH) != list:
            raise TypeError('The derivative of Hamiltonian should be a list!') 

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

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        if ini_measurement == []: 
            np.random.seed(seed)
            M = [[] for i in range(len(self.rho0))]
            for i in range(len(self.rho0)):
                r_ini = 2*np.random.random(len(self.rho0))-np.ones(len(self.rho0))
                r = r_ini/np.linalg.norm(r_ini)
                phi = 2*np.pi*np.random.random(len(self.rho0))
                M[i] = [r[i]*np.exp(1.0j*phi[i]) for i in range(len(self.rho0))]
            self.Measurement = gramschmidt(np.array(M))
        else:
            self.Measurement = ini_measurement

        self.accuracy = accuracy

    def load_save(self):
        file_load = open('measurements.csv', 'r')
        file_load = ''.join([i for i in file_load]).replace("im", "j")
        file_load = ''.join([i for i in file_load]).replace(" ", "")
        file_save = open("measurements.csv","w")
        file_save.writelines(file_load)
        file_save.close()

def MeasurementOpt(*args, method = 'AD', **kwargs):

    if method == 'AD':
        return Measure.AD_Meas(*args, **kwargs)
    elif method == 'PSO':
        return Measure.PSO_Meas(*args, **kwargs)
    elif method == 'DE':
        return Measure.DiffEvo_Meas(*args, **kwargs)
    else:
        raise ValueError("{!r} is not a valid value for method, supported values are 'AD', 'PSO', 'DE'.".format(method))
