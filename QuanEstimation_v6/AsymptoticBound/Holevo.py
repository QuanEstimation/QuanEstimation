# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 09:22:01 2020

@author: JL MZ
"""
class Holevo:
    def Holevo_bound_trace(self,rho_input,drho_input,cost_function):
         '''
         Input:
            1) rho_input: density matrix
            2) drho_input: list of derivative of density matrix on all parameters.
         Notice: drho here contains the derivative on all parameters, for example,
               drho[0] is the derivative vector on the first parameter.
           3) cost_function: cost_function in the Holevo bound.
         '''
        if type(drho_input) != list:
            raise TypeError('Multiple derivatives of density matrix are required for Holevo bound')
        else: pass

        dim = self.dimension
        para_dim = len(drho_input)
        CostG = cost_function
        if len(rho_input) == dim*dim:
            rho_matrix = np.reshape(rho_input,(dim,dim))
        else:
            rho_matrix = rho_input

        QFIM_temp = self.QFIM(rho_input,drho_input)
        QFIMinv = nplin.inv(QFIM_temp)
        SLD_temp = self.SLD(rho_input,drho_input)

        V = np.array([[0.+0.j for i in range(0,para_dim)] for k in range(0,para_dim)])
        for para_i in range(0,para_dim):
            for para_j in range(0,para_dim):
                Vij_temp = 0.+0.j
                for ki in range(0,para_dim):
                    for mi in range(0,para_dim):
                        SLD_ki = SLD_temp[ki]
                        SLD_mi = SLD_temp[mi]
                        Vij_temp = Vij_temp+QFIMinv[para_i][ki]*QFIMinv[para_j][mi]\
                                 *np.trace(np.dot(np.dot(rho_matrix,SLD_ki),SLD_mi))
                V[para_i][para_j] = Vij_temp

        real_part = np.dot(CostG,np.real(V))
        imag_part = np.dot(CostG,np.imag(V))
        Holevo_trace = np.trace(real_part)+np.trace(scylin.sqrtm(np.dot(imag_part,np.conj(np.transpose(imag_part)))))

        return Holevo_trace
