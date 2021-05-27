# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 09:22:01 2020

@author: JL MZ
"""
import numpy as np
from quanestimation.AsymptoticBound.CramerRao import QFIM

class Holevo:
    def __init__(self, precision=1e-10):
        pass
        
    def Holevo_bound_trace(self, rho, drho, cost_function):
        """
        Description: Calculation classical Fisher information matrix (CFIM)
                     for a density matrix.

        ---------
        Inputs
        ---------
        rho:
           --description: parameterized density matrix.
           --type: matrix

        drho:
           --description: derivatives of density matrix on all parameters to
                          be estimated. For example, drho[0] is the derivative
                          vector on the first parameter.
           --type: list (of matrix)

        M:
           --description: cost_function in the Holevo bound.
           --type: matrix

        ----------
        Returns
        ----------
        CFIM:
            --description: trace of Holevo Cramer Rao bound. 
            --type: float number

        """

        if type(drho) != list:
            raise TypeError('Multiple derivatives of density matrix are required for Holevo bound')

        dim = len(rho)
        para_dim = len(drho)
        CostG = cost_function

        QFIM_temp, SLD_temp = QFIM(rho, drho, rho_type='density_matrix', dtype='SLD', rep='original', exportLD=True)
        QFIMinv = np.linalg.inv(QFIM_temp)

        V = np.array([[0.+0.j for i in range(0,para_dim)] for k in range(0,para_dim)])
        for para_i in range(0,para_dim):
            for para_j in range(0,para_dim):
                Vij_temp = 0.+0.j
                for ki in range(0,para_dim):
                    for mi in range(0,para_dim):
                        SLD_ki = SLD_temp[ki]
                        SLD_mi = SLD_temp[mi]
                        Vij_temp = Vij_temp+QFIMinv[para_i][ki]*QFIMinv[para_j][mi]\
                                 *np.trace(np.dot(np.dot(rho, SLD_ki), SLD_mi))
                V[para_i][para_j] = Vij_temp

        real_part = np.dot(CostG,np.real(V))
        imag_part = np.dot(CostG,np.imag(V))
        Holevo_trace = np.trace(real_part)+np.trace(scylin.sqrtm(np.dot(imag_part,np.conj(np.transpose(imag_part)))))

        return Holevo_trace
