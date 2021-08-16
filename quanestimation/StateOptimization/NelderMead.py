import numpy as np
from quanestimation.AsymptoticBound.CramerRao import QFIM
from quanestimation.Dynamics.dynamics import Lindblad
from quanestimation.Common.common import mat_vec_convert

class NelderMead(Lindblad):
    def __init__(self, a_r, a_c, a_e, tspan, rho_initial, H0, Hc, dH, ctrl_initial, Liouville_operator, gamma, control_option=True):
        
        Lindblad.__init__(self, tspan, rho_initial, H0, Hc, dH, ctrl_initial, Liouville_operator, \
                        gamma, control_option)
        """
        --------
        inputs
        --------
        a_r:
           --description: reflection parameter.
           --type: float
        
        a_c:
           --description: contraction parameter.
           --type: float
           
        a_e:
           --description: expansion parameter.
           --type: float
        
        """
        if type(rho_initial) != list:
            raise TypeError('Please make sure rho is a list!')
            
        self.len_state = len(rho_initial)
        self.a_r = a_r
        self.a_c = a_c
        self.a_e = a_e
        
        # self.rho, self.F = self.sort(rho, drho)
        
        
        # def sort(self, rho, drho):
        #     """
        #     Order the states according to their value.
        #     """
        #     F = np.zeros(len(rho))
        #     for si in range(len(rho)):
        #         F[si] = QFIM(rho[si], drho[si])
        #     ind = np.argsort(F)
        #     F_sort = F[ind]
        #     rho_sort = rho[ind]
            
        #     return rho_sort, F_sort
        
        # def reflection(self):
        #     """
        #     Reflection-extension step.
        #     refl: refl = 1 is a standard reflection
        #     """
        #     # reflected point and score
        #     cr = self.rho[0] + self.a_r*(self.rho[0] - self.rho[-1])
        #     rscore = QFIM(cr)   #calculate the target function

        #     return rscore
        
        # def expansion(self, res, x0, ext):
        #     """
        #     ext: the amount of the expansion; ext=0 means no expansion
        #     """
        #     xr, rscore = res[-1]
        #     # if it is the new best point, we try to expand
        #     if rscore < res[0][1]:
        #         xe = xr + ext*(xr - x0)
        #         escore = self.f(xe)
        #         if escore < rscore:
        #             new_res = res[:]
        #             new_res[-1] = (xe, escore)
        #             return new_res
        #     return None
            