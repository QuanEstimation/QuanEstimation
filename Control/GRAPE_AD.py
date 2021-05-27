
import numpy as np
from AsymptoticBound.CramerRao import CramerRao
from Dynamics.dynamics_AD import Lindblad_AD
from Common.common import mat_vec_convert, Adam


class GRAPE_AD(Lindblad_AD, CramerRao):
    def __init__(self, tspan, rho_initial, H0, Hc=[], dH=[], ctrl_initial=[], Liouville_operator=[], \
             gamma=[], control_option=True, precision=1e-8):
        
        Lindblad_AD.__init__(self, tspan, rho_initial, H0, Hc, dH, ctrl_initial, Liouville_operator, \
                    gamma, control_option)
        CramerRao.__init__(self)
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
           --description: Liouville operator.
           --type: list (of matrix)    
           
        gamma:
           --description: decay rates.
           --type: list (of float number)
           
        control_option:   
           --description: if True, add controls to physical system.
           --type: bool
           
        epsilon:
           --description: step to update the control coefficients.
           --type: float number
           
        precision:
           --description: calculation precision.
           --type: float number
        
        """         
        #self.epsilon = epsilon #for test
        self.precision = precision

        self.rho = None
        self.rho_derivative = None
        self.propagator = None
        self.Final = None
        self.m_t, self.v_t = precision, precision

        
    def GRAPE_QFI(self, env_assisted=False):
        """
        Description: Calculation the sum of diagnal elements of classical Fisher information matrix (CFIM)
                     for a density matrix.
        
        ---------
        Inputs
        ---------
        env_assisted:
           --description: whether involved environment parameter.
                          True: environment parameter is involved in GRAPE, and the function 
                          'environment_assisted_state()' has to run first.
                          False: environment parameter is not involved in GRAPE.
           --type: string {'True', 'False'}  
           
        ----------
        Returns
        ----------
           --description: updated values of control coefficients.
           --type: list (of matrix)

        ----------
        Notice
        ----------
           1) To run this funtion, the function 'Data_generation_multiparameter('vector')' has to be run first.
           2) maximize is always more accurate than the minimize in this code.
        
        """
        #----------------------------------------
        self.data_generation()
        #---------------------------------------
        dH0 = self.freeHamiltonian_derivative_Liou[0]
        Hc_coeff = self.control_coeff_total
        D = self.propagator_save
        
        rho_final = mat_vec_convert(self.rho[self.tnum-1])
        drho_final = mat_vec_convert(self.rho_derivative[self.tnum-1][0])
        self.Ffinal, SLD_final = self.QFIM(rho_final,[drho_final], rho_type='density_matrix',dtype='SLD', rep='original', exportLD=True)

        for ti in range(0,self.tnum):
            for ki in range(0,self.ctrlnum_total):
                Hk = self.ctrlH_Liou[ki]
                Mj1 = 1.j*np.dot(D[ti+1][self.tnum-1],np.dot(Hk,self.rho[ti]))
                #-------------------------------------------------------------------
                Mj2 = np.zeros((self.dim*self.dim,1))
                for ri in range(0,ti+1):
                    Mj2_temp = np.dot(D[ri+1][ti],np.dot(dH0,self.rho[ri]))
                    Mj2_temp = np.dot(D[ti+1][self.tnum-1],np.dot(Hk,Mj2_temp))
                    Mj2 = Mj2+Mj2_temp
                #-------------------------------------------------------------------
                Mj3 = np.zeros((self.dim*self.dim,1))
                for ri in range(ti+1,self.tnum):
                    Mj3_temp = np.dot(D[ti+1][ri],np.dot(Hk,self.rho[ti]))
                    Mj3_temp = np.dot(D[ri+1][self.tnum-1],np.dot(dH0,Mj3_temp))
                    Mj3 = Mj3+Mj3_temp
                Mj1mat = np.reshape(Mj1,(self.dim,self.dim))
                Mj2mat = np.reshape(Mj2,(self.dim,self.dim))
                Mj3mat = np.reshape(Mj3,(self.dim,self.dim))

                SLD2 = np.dot(SLD_final,SLD_final)
                term1 = self.dt*np.trace(np.dot(SLD2,Mj1mat))
                term2 = -2*(self.dt**2)*np.trace(np.dot(SLD_final,Mj2mat+Mj3mat))
                delta = np.real(term1+term2)
                #---------------------------------
                # update the control coefficients:
                #---------------------------------
                Hc_kiti = Hc_coeff[ki]
                Hc_kiti[ti], self.m_t, self.v_t = Adam(delta, ti, Hc_kiti[ti], self.m_t, self.v_t, \
                                          alpha=0.01, beta1=0.90, beta2=0.99, epsilon=self.precision)
                Hc_coeff[ki] = Hc_kiti
                
        if self.environmentstate == False:
            self.control_coeff_total = Hc_coeff
            self.control_coefficients = self.control_coeff_total
        elif  self.environmentstate == True:
            self.control_coefficients = Hc_coeff[0:self.ctrlnum]
            for ei in range(0,len(self.environment_assisted_order)):
                gam_num = self.environment_assisted_order[ei]
                self.gamma[gam_num] = Hc_coeff[self.ctrlnum+ei]

    def GRAPE_CFIM(self, M, obj_fun):
        """
        Description: Calculation the sum of diagnal elements of classical Fisher information matrix (CFIM)
                     for a density matrix.
                     
        ---------
        Inputs
        ---------
        M:
           --description: a set of POVM. It takes the form [M1, M2, ...].
           --type: list (of matrix)
           
        rep:
           --description: setting the objective function of GRAPE.
                          None: calculate CFI.
                          'f0': $f_{0}=\sum_a 1/F_{aa}$.
                          'f1': the lower bound $f_{1}=d^2/TrF$.
                          'exact': exact gradient for TrF^{-1}, however, it is ONLY valid for two-parameter systems.
           --type: string {None, 'f0', 'f1', 'exact'}  
        
        ----------
        Returns
        ----------
           --description: updated values of control coefficients..
           --type: list (of matrix)

        ----------
        Notice
        ----------
           1) To run this funtion, the function 'Data_generation_multiparameter('vector')' has to be run first.
           2) maximize is always more accurate than the minimize in this code.
        
        """
        #----------------------------------------
        self.data_generation()
        #---------------------------------------

        paralen = len(self.Hamiltonian_derivative)
        dH0 = self.freeHamiltonian_derivative_Liou
        Hc_coeff = self.control_coeff_total
        Hc = self.control_Hamiltonian
        D = self.propagator_save

        rhoT_vec = self.rho[self.tnum-1]
        rhoT = np.reshape(rhoT_vec,(self.dim,self.dim))
        drhoT_vec = self.rho_derivative[self.tnum-1]
        drhoT = [[] for i in range(0,paralen)]
        for para_i in range(0,paralen):
            drhoT[para_i] = np.reshape(drhoT_vec[para_i],(self.dim,self.dim))
        #------------------------------------------------------------------------------
        #Generation of L1 and L2 for diagonal entries of CFIM (i.e., alpha = beta):
        #------------------------------------------------------------------------------
        L1 = [[] for i in range(0,paralen)]
        L2 = [[] for i in range(0,paralen)]
        for para_i in range(0,paralen):
            L1[para_i] = np.zeros((self.dim,self.dim))
            L2[para_i] = np.zeros((self.dim,self.dim))
        for para_j in range(0,paralen):
            for mi in range(0,len(M)):
                ptemp = np.trace(np.dot(rhoT,M[mi]))
                dptemp = np.trace(np.dot(drhoT[para_j],M[mi]))
                if ptemp > self.precision:
                    L1[para_j] = L1[para_j]+(dptemp/ptemp)*M[mi]
                    L2[para_j] = L2[para_j]+((dptemp/ptemp)**2)*M[mi]
                elif ptemp < self.precision:
                    L1[para_j] = L1[para_j]
                    L2[para_j] = L2[para_j]
         #------------------------------------------------------------------------------
        #Generation L2 for off-diagonal entries of CFIM in two-parameter estimation:
        #------------------------------------------------------------------------------
        if paralen == 2:
            L2_offdiag = np.zeros((self.dim,self.dim))
            for mi in range(0,len(M)):
                ptp_2para = np.trace(np.dot(rhoT,M[mi]))
                dptp0_2para = np.trace(np.dot(drhoT[0],M[mi]))
                dptp1_2para = np.trace(np.dot(drhoT[1],M[mi]))
                L2_offdiag = L2_offdiag+(dptp0_2para*dptp1_2para/ptp_2para/ptp_2para)*M[mi]

        #--------------------------------------------------
        # Generation of CFIM at the target time
        #--------------------------------------------------
        CFIM_temp = self.CFIM(rhoT,drhoT,M)
        norm_f0 = 0.
        #--------------------------------------------------
        if paralen==1:
            norm_f0 = norm_f0+1/CFIM_temp
        else:
            for ci in range(0,paralen):
                norm_f0 = norm_f0+1/CFIM_temp[ci][ci]
        #--------------------------------------------------
        norm_f0 = norm_f0**2
        M2_2para = [[] for i in range(0,2)]
        M3_2para = [[] for i in range(0,2)]

        for ti in range(0,self.tnum):
            #--------------------------------------------
            #calculation of gradient
            #--------------------------------------------
            for ki in range(0,self.ctrlnum_total):
                Hk = self.ctrlH_Liou[ki]
                Mj1 = 1.j*np.dot(D[ti+1][self.tnum-1],np.dot(Hk,self.rho[ti]))
                Mj1mat = np.reshape(Mj1,(self.dim,self.dim))
                delta = 0.

                for para_i in range(0,paralen):
                    #----------------------------------------------------------------------
                    Mj2 = np.zeros((self.dim*self.dim,1))
                    for ri in range(0,ti+1):
                        Mj2_temp = np.dot(D[ri+1][ti],np.dot(dH0[para_i],self.rho[ri]))
                        Mj2_temp = np.dot(D[ti+1][self.tnum-1],np.dot(Hk,Mj2_temp))
                        Mj2 = Mj2+Mj2_temp
                    #----------------------------------------------------------------------
                    Mj3 = np.zeros((self.dim*self.dim,1))
                    for ri in range(ti+1,self.tnum):
                        Mj3_temp = np.dot(D[ti+1][ri],np.dot(Hk,self.rho[ti]))
                        Mj3_temp = np.dot(D[ri+1][self.tnum-1],np.dot(dH0[para_i],Mj3_temp))
                        Mj3 = Mj3+Mj3_temp

                    Mj2mat = np.reshape(Mj2,(self.dim,self.dim))
                    Mj3mat = np.reshape(Mj3,(self.dim,self.dim))

                    term1 = self.dt*np.trace(np.dot(L2[para_i],Mj1mat))
                    term2 = -2*(self.dt**2)*np.trace(np.dot(L1[para_i],Mj2mat+Mj3mat))
                    #---------------------------------------------------------------------
                    if paralen==1:
                        self.Ffinal = CFIM_temp
                        delta = np.real(term1+term2)
                    #---------------------------------------------------------------------
                    else:
                        if obj_fun == 'f0':
                            delta = delta+np.real(term1+term2)/((CFIM_temp[para_i][para_i])**2)/norm_f0
                        elif obj_fun == 'f1':
                            delta = delta+np.real(term1+term2)/float(paralen)/float(paralen)
                        elif obj_fun == 'exact':
                            if paralen > 2:
                                raise TypeError('the "exact" mode is only valid for two-parameter systems, the current\
                                              parameter number is '+str(paralen))
                            elif paralen == 2:
                                delta = delta+np.real(term1+term2)*((CFIM_temp[1-para_i][1-para_i])**2\
                                      +(CFIM_temp[0][1])**2)/((np.trace(CFIM_temp))**2)
                                M2_2para[para_i] = Mj2mat
                                M3_2para[para_i] = Mj3mat

                if paralen>1 and obj_fun == 'exact':
                    gradient_offdiag = self.dt*np.trace(np.dot(L2_offdiag, Mj1mat))-(self.dt**2)*np.trace(np.dot(L1[1], \
                                 M2_2para[0]+M3_2para[0]))-(self.dt**2)*np.trace(np.dot(L1[0], M2_2para[1]+M3_2para[1]))
                    delta = delta-np.real(2*gradient_offdiag*CFIM_temp[0][1]/np.trace(CFIM_temp))

                 #------------------------------------------------------------
                 #update the control coefficients:
                 #------------------------------------------------------------
                Hc_kiti = Hc_coeff[ki]
                Hc_kiti[ti], self.m_t, self.v_t = Adam(delta, ti, Hc_kiti[ti], self.m_t, self.v_t, \
                                          alpha=0.01, beta1=0.90, beta2=0.99, epsilon=self.precision)
                Hc_coeff[ki] = Hc_kiti
                
        if   self.environmentstate == False:
            self.control_coeff_total = Hc_coeff
            self.control_coefficients = self.control_coeff_total
        elif  self.environmentstate == True:
            self.control_coefficients = Hc_coeff[0:self.ctrlnum]
            for ei in range(0,len(self.environment_assisted_order)):
                gam_num = self.environment_assisted_order[ei]
                self.gamma[gam_num] = Hc_coeff[self.ctrlnum+ei]
