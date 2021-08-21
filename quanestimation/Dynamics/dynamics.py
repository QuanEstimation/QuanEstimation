
import numpy as np
import warnings
import math
from scipy import linalg as scylin
from qutip import *
from julia import Main
from quanestimation.Common.common import dRHO
# Main.include('./'+'Julia'+'/'+'src'+'/'+'QuanEstimation.jl')

# Main.include('./'+'Common'+'/'+'Liouville.jl')

class Lindblad:
    """
    General dynamics of density matrices in the form of time local Lindblad master equation.
    {\partial_t \rho} = -i[H, \rho] + \sum_n {\gamma_n} {Ln.rho.Ln^{\dagger}
                 -0.5(rho.Ln^{\dagger}.Ln+Ln^{\dagger}.Ln.rho)}.
    """

    def __init__(self, tspan, rho_initial, H0, dH, Liouville_operator=[], gamma=[], Hc=[], ctrl_initial=[], control_option=True):
        """
        ----------
        Inputs
        ----------
        tspan: 
           --description: time series.
           --type: array
        
        rho_initial: 
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
           
        ctrl_initial: 
           --description: control coefficients.
           --type: list (of array)
           
        Liouville operator:
           --description: Liouville operator in Lindblad master equation.
           --type: list (of matrix)    
           
        gamma:
           --description: decay rates.
           --type: list (of float number)
           
        control_option:   
           --description: if True, add controls to physical system.
           --type: bool
        """
        
        self.freeHamiltonian = H0
        self.Liouville_operator = Liouville_operator
        self.gamma = gamma
        self.rho_initial = rho_initial
        self.tspan = tspan
        self.dt = self.tspan[1]-self.tspan[0]
        self.T = self.tspan[-1]
        self.tnum = len(self.tspan)
        self.Hamiltonian_derivative = dH
        self.control_Hamiltonian = Hc
        self.control_coefficients = ctrl_initial
        self.control_option = control_option

        self.dim = len(self.freeHamiltonian)
        self.Liouvillenumber = len(self.Liouville_operator)
        
        self.ctrlnum = len(self.control_Hamiltonian)
        self.ctrlnum_total = self.ctrlnum
        self.control_coeff_total = self.control_coefficients
        self.rho = None
        self.rho_derivative = None
        self.propagator_save = None
        self.environment_assisted_order = None
        self.environmentstate = False
        
        ctrl_length = len(self.control_coefficients)
        ctrlnum = len(self.control_Hamiltonian)
        if ctrlnum < ctrl_length:
            raise TypeError('There are %d control Hamiltonians but %d coefficients sequences: \
                            too many coefficients sequences'%(ctrlnum,ctrl_length))
        elif ctrlnum > ctrl_length:
            warnings.warn('Not enough coefficients sequences: there are %d control Hamiltonians \
                           but %d coefficients sequences. The rest of the control sequences are\
                           set to be 0.'%(ctrlnum,ctrl_length), DeprecationWarning)
        else: pass

        
        if len(self.gamma) != self.Liouvillenumber:
            raise TypeError('The length of decay rates and Liouville operators should be the same')
        
        if type(self.Hamiltonian_derivative) != list:
            raise TypeError('The derivative of Hamiltonian should be a list!')
        else:
            self.freeHamiltonian_derivative_Liou = []
            for para_i in range(0,len(self.Hamiltonian_derivative)):
                dH0_temp = Main.QuanEstimation.liouville_commu_py(self.Hamiltonian_derivative[para_i])
                self.freeHamiltonian_derivative_Liou.append(dH0_temp)

        self.ctrlH_Liou = []
        for hi in range(0,self.ctrlnum):
            Htemp = Main.QuanEstimation.liouville_commu_py(self.control_Hamiltonian[hi])
            self.ctrlH_Liou.append(Htemp)

    def general_information(self):
        print('==================================')
        print('General information:')
        show_dimension = 'dimension of Hamiltonian: '+str(self.dim)
        print(show_dimension)
        show_Liou = 'number of Liouville operators: '+str(self.Liouvillenumber)
        print(show_Liou)
        show_num = 'number of time step: '+str(self.tnum)
        print(show_num)
        show_ctrl = 'number of controls: '+str(self.ctrlnum_total)
        print(show_ctrl)
        show_cswitch = 'Control switch is '+str(self.control)
        print(show_cswitch)
        print('==================================')


    def Dicoherence_Liouville(self,tj):

        ga = [0. for i in range(0,self.Liouvillenumber)]
        for gi in range(0,self.Liouvillenumber):
            gtest = self.gamma[gi]
            if type(gtest) == float:
                ga[gi] = gtest
            elif type(gtest) != float:
                ga[gi] = gtest[tj]

        result = [[0. for i in range(0,self.dim*self.dim)] for k in range(0,self.dim*self.dim)]
        for bi in range(0,self.dim):
            for bj in range(0,self.dim):
                ni = self.dim*bi+bj
                for bk in range(0,self.dim):
                    for bl in range(0,self.dim):
                        nj = self.dim*bk+bl
                        L_temp = 0.
                        for Ln in range(0,self.Liouvillenumber):
                            Lc = self.Liouville_operator[Ln]
                            L_temp = L_temp+ga[Ln]*Lc[bi][bk]*np.conj(Lc[bj][bl])
                            for bp in range(0,self.dim):
                                L_temp = L_temp-0.5*ga[Ln]*float(bk==bi)*Lc[bp][bj]*np.conj(Lc[bp][bl])\
                                       -0.5*ga[Ln]*float(bl==bj)*Lc[bp][bk]*np.conj(Lc[bp][bi])
                        result[ni][nj] = L_temp

        result = np.array(result)
        result[np.abs(result) < 1e-10] = 0.
        return result

    def Hamiltonian_Liouville(self,tj):
        if self.control_option == False:
            freeHamiltonian_Liouville = -1.j*Main.QuanEstimation.liouville_commu_py(self.freeHamiltonian)
            Ld = freeHamiltonian_Liouville+self.Dicoherence_Liouville(tj)
            result = scylin.expm(self.dt*Ld)

        elif self.control_option == True:
            if type(tj) != int:
                raise TypeError('input variable has to be the number of time point')
            else:
                Htot = self.freeHamiltonian
                for hn in range(0,self.ctrlnum):
                    Hc_temp = None
                    Hc_temp = self.control_coefficients[hn]
                    Htot = Htot+self.control_Hamiltonian[hn]*Hc_temp[tj]
                freepart = Main.QuanEstimation.liouville_commu_py(Htot)
                Ld = -1.j*freepart+self.Dicoherence_Liouville(tj)
                result = scylin.expm(self.dt*Ld)
        return result

    def Propagator(self,tstart,tend):
        if type(tstart) != int and type(tend) != int:
            raise TypeError('inputs are the number of time interval')
        else:
            if tstart > tend:
                result = np.eye(self.dim*self.dim)
            elif tstart == tend:
                result = self.Hamiltonian_Liouville(tstart)
            else:
                result = self.Hamiltonian_Liouville(tstart)
                for pi in range(tstart+1,tend-tstart):
                    Ltemp = self.Hamiltonian_Liouville(pi)
                    result = np.dot(Ltemp,result)
            return result

    def evolved_state(self,tj):
        rho_temp = np.reshape(self.rho_initial,(self.dim*self.dim,1))
        propa = self.Propagator(0,tj)
        rho_tvec = np.dot(propa,rho_temp)
        return rho_tvec

    def data_generation(self):
        """
        Description: This function will save all the propators during the evolution,
                     which may be memory consuming.
        ----------
        outputs
        ----------
        rho: 
           --description: parameterized density matrix.
           --type: list (of matrix)
           
        rho_derivative: 
           --description: derivatives of density matrix on all parameters to
                          be estimated.
           --type: list (of matrix)
           
        propagator_save: 
           --description: propagating superoperator.
           --type: list (of matrix)  
        """ 
        tnum = self.tnum
        dim = self.dim
        dH = self.Hamiltonian_derivative 
        para_len = len(dH)
        dL = [[] for i in range(0,para_len)]
        for para_i in range(0,para_len):
            dL_temp = -1.j*Main.QuanEstimation.liouville_commu_py(dH[para_i])
            dL[para_i] = dL_temp
        dt = self.dt
        D = [[[] for i in range(0,tnum+1)] for i in range(0,tnum+1)]
        rhovec = [[] for i in range(0,tnum)]
        drhovec = [[[] for k in range(0,para_len)] for i in range(0,tnum)]
        rhomat = [[] for i in range(0,tnum)]
        drhomat = [[[] for k in range(0,para_len)] for i in range(0,tnum)]

        rhovec[0] = self.evolved_state(0)
        for para_i in range(0,para_len):
            drhovec_temp = dt*np.dot(dL[para_i],rhovec[0])
            drhovec[0][para_i] = drhovec_temp
            drhomat[0][para_i] = np.reshape(drhovec_temp,(dim,dim))
        D[0][0] = self.Hamiltonian_Liouville(0)
        D[1][0] = np.eye(dim*dim)

        for di in range(1,tnum):
            D[di+1][di] = np.eye(dim*dim)
            D[di][di] = self.Hamiltonian_Liouville(di)
            D[0][di] = np.dot(D[di][di],D[0][di-1])
            rhovec[di] = np.array(np.dot(D[di][di],rhovec[di-1]))
            rhomat[di] = np.reshape(rhovec[di],(dim,dim))

            for para_i in range(0,para_len):
                drho_temp = dt*np.dot(dL[para_i],rhovec[di])
                for dj in range(1,di):
                    D[di-dj][di] = np.dot(D[di-dj+1][di],D[di-dj][di-dj])
                    drho_temp = drho_temp+dt*np.dot(D[di-dj+1][di],np.dot(dL[para_i],rhovec[di-dj]))
                drhovec[di][para_i] = np.array(drho_temp)
                drhomat[di][para_i] = np.reshape(np.array(drho_temp),(dim,dim))
        self.rho = rhovec
        self.rho_derivative = drhovec
        self.propagator_save = D

    def environment_assisted_state(self,statement,Dissipation_order):
        '''
        If the dissipation coefficient can be manually manipulated, it can be updated via GRAPE.
        This function is used to clarify which dissipation parameter can be updated.
        Input: 1) statement: True: the dissipation parameter is involved in the GRAPE.
                       False: the dissipation parameter is not involved in the GRAPE.
             2) Dissipation_order: number list contains the number of dissipation parameter to be updated.
                            For example, [3] means the 3rd Liouville operator can be updated and
                            [3, 5] means the 3rd and 5th Liouville operators can be updated.
        '''
        if  statement == True:
            newnum = int(self.ctrlnum+len(Dissipation_order))
            Hk_Liou = [[] for i in range(0,newnum)] 
            for hi in range(0,self.ctrlnum):
                Hk_Liou[hi] = Main.QuanEstimation.liouville_commu_py(self.control_Hamiltonian[hi])
            for hi in range(0,len(Dissipation_order)):
                hj = int(self.ctrlnum+hi)
                hnum = Dissipation_order[hi]
                Hk_Liou[hj] = 1.j*Main.QuanEstimation.liouville_dissip_py(self.Liouville_operator[hnum])
                ga = self.gamma[hnum]
                ctrl_coeff = self.control_coeff_total
                ctrl_coeff.append(ga)
                self.control_coeff_total = ctrl_coeff
            self.ctrlnum_total = newnum
            self.ctrlH_Liou = Hk_Liou
            self.environment_assisted_order = Dissipation_order
            self.environmentstate = statement

    def expm(self):
        tnum = self.tnum
        dim = self.dim
        dt = self.dt
        para_num = len(self.Hamiltonian_derivative)
        dH_L = self.freeHamiltonian_derivative_Liou

        rhovec = [[] for i in range(tnum)]
        rhovec[0] = self.rho_initial.reshape(dim*dim,1)
        # rhovec[0] = np.dot(self.Hamiltonian_Liouville(0), self.rho_initial.reshape(dim*dim,1))
        drhovec = [[[] for pj in range(para_num)] for j in range(tnum)]
        for para_i in range(para_num):
            drhovec[0][para_i] = np.zeros(dim*dim, dtype=np.complex128).reshape(dim*dim,1)
            # drhovec[0][para_i] = -1.j*dt*np.dot(dH_L[para_i], rhovec[0])
            
        for ti in range(1, tnum):
            Liouville_tot = self.Hamiltonian_Liouville(ti)
            rhovec[ti] = np.dot(Liouville_tot, rhovec[ti-1])
            for para_k in range(para_num):
                drhovec[ti][para_k] = -1.j*dt*np.dot(dH_L[para_k], rhovec[ti])+np.dot(Liouville_tot, drhovec[ti-1][para_k])

        rho = [(rhovec[i]).reshape(dim, dim) for i in range(tnum)]

        drho = [[] for i in range(tnum)]
        for i in range(tnum):
            drho[i] = [(drhovec[i][para_l]).reshape(dim, dim) for para_l in range(para_num)]

        return rho, drho

    def ODE(self):
        # no control
        tnum = self.tnum
        dim = self.dim
        dt = self.dt
        dH_L = self.freeHamiltonian_derivative_Liou
        H0_L = Main.QuanEstimation.liouville_commu_py(self.freeHamiltonian)

        state_pre = self.rho_initial.reshape(dim**2, 1)
        dstate = [np.array([[0.+0.*1.j] for i in range(0,dim**2)]) for i in range(len(dH_L))]

        Liouv_tot = -1.0j*H0_L
        c_ops = []
        for gi in range(len(self.gamma)):
            Liouv_tot = Liouv_tot + self.gamma[gi]*Main.QuanEstimation.liouville_dissip_py(self.Liouville_operator[gi])
            c_ops.append(np.sqrt(self.gamma[gi])*self.Liouville_operator[gi])

        if len(self.gamma) == 1:
            rho_res = mesolve(Qobj(self.freeHamiltonian), Qobj(self.rho_initial), self.tspan, Qobj(c_ops[0]), [])
        else:
            rho_res = mesolve(Qobj(self.freeHamiltonian), Qobj(self.rho_initial), self.tspan, Qobj(c_ops), [])

        rho, drho = [],[]
        for ti in range(tnum):
            rho_tp = rho_res.states[ti].full()
            state = rho_tp.reshape(dim**2, 1)
            # state = np.dot(scylin.expm(dt*Liouv_tot), state_pre)
            # rho = state.reshape(dim, dim)
            drho_tp = []
            for pi in range(len(dH_L)):
                A = -1j*np.dot(dH_L[pi], state_pre)
                dstate[pi] = dRHO(dstate[pi], Liouv_tot, A, dt)
                drho_tp.append(dstate[pi].reshape(dim, dim))
            state_pre = state
            rho.append(rho_tp)
            drho.append(drho_tp)
        return rho, drho

