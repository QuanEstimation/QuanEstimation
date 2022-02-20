
import numpy as np
import os
from scipy import interpolate
from scipy.integrate import simps, solve_bvp
from itertools import product
from quanestimation.AsymptoticBound.CramerRao import CFIM, QFIM, SLD
from quanestimation.Common.common import load_M, extract_ele

def BCRB(x, p, rho, drho, M=[], b=[], db=[], btype=1, eps=1e-8):
    para_num = len(x)
    if para_num==1:
        #### singleparameter senario ####
        p_num = len(p)
        if b==[]:
            b = np.zeros(p_num)
            db = np.zeros(p_num)
        if b!=[] and db==[]:
            db = np.zeros(p_num)

        if M==[]: 
            M = load_M(len(rho[0]))
        else:
            if type(M) != list:
                raise TypeError("Please make sure M is a list!")

        if type(drho[0]) == list:
            drho = [drho[i][0] for i in range(p_num)]
        if type(b[0]) == list or type(b[0]) == np.ndarray:
            b = b[0]
        if type(db[0]) == list or type(db[0]) == np.ndarray:
            db = db[0]

        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = CFIM(rho[m], [drho[m]], M=M, eps=eps)

        if btype == 1:
            arr = [p[i]*((1+db[i])**2/F_tp[i]+b[i]**2) for i in range(p_num)]
            F = simps(arr, x[0])
            return F
        elif btype == 2:
            arr = [p[i]*F_tp[i] for i in range(p_num)]
            F1 = simps(arr, x[0])
            arr2 = [p[j]*(1+db[j]) for j in range(p_num)]
            B = simps(arr2, x[0])
            arr3 = [p[k]*b[k]**2 for k in range(p_num)]
            bb = simps(arr3, x[0])
            F = B**2/F1+bb
            return F
        else:
            raise NameError("NameError: btype should be choosen in {1, 2}")
    else:
        #### multiparameter senario ####
        if b==[]:
            b, db = [], []
            for i in range(para_num):
                b.append(np.zeros(len(x[i])))
                db.append(np.zeros(len(x[i])))
        if b!=[] and db==[]:
            db = []
            for i in range(para_num):
                db.append(np.zeros(len(x[i])))

        p_shape = np.shape(p)
        p_ext = extract_ele(p, para_num)
        rho_ext = extract_ele(rho, para_num)
        drho_ext = extract_ele(drho, para_num)
        b_pro = product(*b)
        db_pro = product(*db)

        p_list, rho_list, drho_list = [], [], []
        for p_ele, rho_ele, drho_ele in zip(p_ext, rho_ext, drho_ext):
            p_list.append(p_ele)
            rho_list.append(rho_ele)
            drho_list.append(drho_ele) 

        b_list, db_list = [], []
        for b_ele, db_ele in zip(b_pro, db_pro):
            b_list.append([b_ele[i] for i in range(para_num)])
            db_list.append([db_ele[j] for j in range(para_num)])

        dim = len(rho_list[0])
        if M==[]: 
            M = load_M(dim)
        else:
            if type(M) != list:
                raise TypeError("Please make sure M is a list!")
        if btype == 1:
            F_list = [[[0.0 for i in range(len(p_list))] for j in range(para_num)] for k in range(para_num)]
            for i in range(len(p_list)):
                F_tp = CFIM(rho_list[i], drho_list[i], M=M, eps=eps)
                F_inv = np.linalg.pinv(F_tp)
                B = np.diag([(1.0+db_list[i][j]) for j in range(para_num)])
                term1 = np.dot(B, np.dot(F_inv, B))
                term2 = np.dot(np.array(b_list[i]).reshape(para_num, 1), np.array(b_list[i]).reshape(1, para_num))
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = term1[pj][pk]+term2[pj][pk]

            res = np.zeros([para_num, para_num])
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                    arr = p*F_ij
                    for si in reversed(range(para_num)):
                        arr = simps(arr, x[si])
                    res[para_i][para_j] = arr
                    res[para_j][para_i] = arr
            return res
        elif btype == 2:
            F_list = [[[0.0 for i in range(len(p_list))] for j in range(para_num)] for k in range(para_num)]
            B_list = [[[0.0 for i in range(len(p_list))] for j in range(para_num)] for k in range(para_num)]
            bb_list = [[[0.0 for i in range(len(p_list))] for j in range(para_num)] for k in range(para_num)]
            for i in range(len(p_list)):
                F_tp = CFIM(rho_list[i], drho_list[i], M=M, eps=eps)
                B_tp = np.diag([(1.0+db_list[i][j]) for j in range(para_num)])
                bb_tp = np.dot(np.array(b_list[i]).reshape(para_num, 1), np.array(b_list[i]).reshape(1, para_num))
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = F_tp[pj][pk]
                        B_list[pj][pk][i] = B_tp[pj][pk]
                        bb_list[pj][pk][i] = bb_tp[pj][pk]

            F_res = np.zeros([para_num, para_num])
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                    arr = p*F_ij
                    for si in reversed(range(para_num)):
                        arr = simps(arr, x[si])
                    F_res[para_i][para_j] = arr
                    F_res[para_j][para_i] = arr
            B_res = np.zeros([para_num, para_num])
            bb_res = np.zeros([para_num, para_num])
            for para_m in range(para_num):
                for para_n in range(para_num):
                    B_mn = np.array(B_list[para_m][para_n]).reshape(p_shape)
                    bb_mn = np.array(bb_list[para_m][para_n]).reshape(p_shape)
                    arr2 = p*B_mn
                    arr3 = p*bb_mn
                    for sj in reversed(range(para_num)):
                        arr2 = simps(arr2, x[sj])
                        arr3 = simps(arr3, x[sj])
                    B_res[para_m][para_n] = arr2
                    bb_res[para_m][para_n] = arr3
            res = np.dot(B_res, np.dot(np.linalg.pinv(F_res), B_res)) + bb_res
            return res
        else:
            raise NameError("NameError: btype should be choosen in {1, 2}")

def BQCRB(x, p, rho, drho, b=[], db=[], btype=1, dtype="SLD", eps=1e-8):
    para_num = len(x)

    if para_num==1:
        #### singleparameter senario ####
        p_num = len(p)
    
        if b==[]:
            b = np.zeros(p_num)
            db = np.zeros(p_num)
        if b!=[] and db==[]:
            db = np.zeros(p_num)

        if type(drho[0]) == list:
            drho = [drho[i][0] for i in range(p_num)]
        if type(b[0]) == list or type(b[0]) == np.ndarray:
            b = b[0]
        if type(db[0]) == list or type(db[0]) == np.ndarray:
            db = db[0]

        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = QFIM(rho[m], [drho[m]], dtype=dtype, eps=eps)

        if btype == 1:
            arr = [p[i]*((1+db[i])**2/F_tp[i]+b[i]**2) for i in range(p_num)]
            F = simps(arr, x[0])
            return F
        elif btype == 2:
            arr2 = [p[i]*F_tp[i] for i in range(p_num)]
            F2 = simps(arr2, x[0])
            arr2 = [p[j]*(1+db[j]) for j in range(p_num)]
            B = simps(arr2, x[0])
            arr3 = [p[k]*b[k]**2 for k in range(p_num)]
            bb = simps(arr3, x[0])
            F = B**2/F2+bb
            return F
        else:
            raise NameError("NameError: btype should be choosen in {1, 2}")
    else:
        #### multiparameter senario ####
        if b==[]:
            b, db = [], []
            for i in range(para_num):
                b.append(np.zeros(len(x[i])))
                db.append(np.zeros(len(x[i])))
        if b!=[] and db==[]:
            db = []
            for i in range(para_num):
                db.append(np.zeros(len(x[i])))

        p_shape = np.shape(p)
        p_ext = extract_ele(p, para_num)
        rho_ext = extract_ele(rho, para_num)
        drho_ext = extract_ele(drho, para_num)
        b_pro = product(*b)
        db_pro = product(*db)

        p_list, rho_list, drho_list = [], [], []
        for p_ele, rho_ele, drho_ele in zip(p_ext, rho_ext, drho_ext):
            p_list.append(p_ele)
            rho_list.append(rho_ele)
            drho_list.append(drho_ele) 

        b_list, db_list = [], []
        for b_ele, db_ele in zip(b_pro, db_pro):
            b_list.append([b_ele[i] for i in range(para_num)])
            db_list.append([db_ele[j] for j in range(para_num)])

        if btype == 1:
            F_list = [[[0.0 for i in range(len(p_list))] for j in range(para_num)] for k in range(para_num)]
            for i in range(len(p_list)):
                F_tp = QFIM(rho_list[i], drho_list[i], dtype=dtype, eps=eps)
                F_inv = np.linalg.pinv(F_tp)
                B = np.diag([(1.0+db_list[i][j]) for j in range(para_num)])
                term1 = np.dot(B, np.dot(F_inv, B))
                term2 = np.dot(np.array(b_list[i]).reshape(para_num, 1), np.array(b_list[i]).reshape(1, para_num))
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = term1[pj][pk]+term2[pj][pk]

            res = np.zeros([para_num,para_num])
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                    arr = p*F_ij
                    for si in reversed(range(para_num)):
                        arr = simps(arr, x[si])
                    res[para_i][para_j] = arr
                    res[para_j][para_i] = arr
            return res
        elif btype == 2:
            F_list = [[[0.0 for i in range(len(p_list))] for j in range(para_num)] for k in range(para_num)]
            B_list = [[[0.0 for i in range(len(p_list))] for j in range(para_num)] for k in range(para_num)]
            bb_list = [[[0.0 for i in range(len(p_list))] for j in range(para_num)] for k in range(para_num)]
            for i in range(len(p_list)):
                F_tp = QFIM(rho_list[i], drho_list[i], dtype=dtype, eps=eps)
                B_tp = np.diag([(1.0+db_list[i][j]) for j in range(para_num)])
                bb_tp = np.dot(np.array(b_list[i]).reshape(para_num, 1), np.array(b_list[i]).reshape(1, para_num))
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = F_tp[pj][pk]
                        B_list[pj][pk][i] = B_tp[pj][pk]
                        bb_list[pj][pk][i] = bb_tp[pj][pk]

            F_res = np.zeros([para_num,para_num])
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                    arr = p*F_ij
                    for si in reversed(range(para_num)):
                        arr = simps(arr, x[si])
                    F_res[para_i][para_j] = arr
                    F_res[para_j][para_i] = arr
            B_res = np.zeros([para_num,para_num])
            bb_res = np.zeros([para_num,para_num])
            for para_m in range(para_num):
                for para_n in range(para_num):
                    B_mn = np.array(B_list[para_m][para_n]).reshape(p_shape)
                    bb_mn = np.array(bb_list[para_m][para_n]).reshape(p_shape)
                    arr2 = p*B_mn
                    arr3 = p*bb_mn
                    for sj in reversed(range(para_num)):
                        arr2 = simps(arr2, x[sj])
                        arr3 = simps(arr3, x[sj])
                    B_res[para_m][para_n] = arr2
                    bb_res[para_m][para_n] = arr3
            res = np.dot(B_res, np.dot(np.linalg.pinv(F_res), B_res)) + bb_res
            return res
        else:
            raise NameError("NameError: btype should be choosen in {1, 2}")
    
def VTB(x, p, dp, rho, drho, M=[], btype=1, eps=1e-8):
    para_num = len(x)
    p_num = len(p)

    if para_num == 1:
        #### singleparameter senario ####
        if M==[]: 
            M = load_M(len(rho[0]))
        else:
            if type(M) != list:
                raise TypeError("Please make sure M is a list!")

        if type(drho[0]) == list:
            drho = [drho[i][0] for i in range(p_num)]
        if type(dp[0]) == list or type(dp[0]) == np.ndarray:
            dp = [dp[i][0] for i in range(p_num)]

        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = CFIM(rho[m], [drho[m]], M=M, eps=eps)

        if btype == 1:
            I_tp = [np.real(dp[i]*dp[i]/p[i]**2) for i in range(p_num)]
            arr = [p[j]/(I_tp[j]+F_tp[j]) for j in range(p_num)]
            return simps(arr, x[0])
        elif btype == 2:
            arr1 = [np.real(dp[i]*dp[i]/p[i]) for i in range(p_num)]  
            I = simps(arr1, x[0])
            arr2 = [np.real(F_tp[j]*p[j]) for j in range(p_num)]
            F = simps(arr2, x[0])
            return 1.0/(I+F)
        else:
            raise NameError("NameError: btype should be choosen in {1, 2}")
    else:
        #### multiparameter senario ####
        p_shape = np.shape(p)
        p_ext = extract_ele(p, para_num)
        dp_ext = extract_ele(dp, para_num)
        rho_ext = extract_ele(rho, para_num)
        drho_ext = extract_ele(drho, para_num)
        
        p_list, dp_list, rho_list, drho_list = [], [], [], []
        for p_ele, dp_ele, rho_ele, drho_ele in zip(p_ext, dp_ext, rho_ext, drho_ext):
            p_list.append(p_ele)
            dp_list.append(dp_ele)
            rho_list.append(rho_ele)
            drho_list.append(drho_ele)
   
        dim = len(rho_list[0])
        if M==[]: 
            M = load_M(dim)
        else:
            if type(M) != list:
                raise TypeError("Please make sure M is a list!")
        if btype == 1:
            F_list = [[[0.0 for i in range(len(p_list))] for j in range(para_num)] for k in range(para_num)]
            for i in range(len(p_list)):
                F_tp = CFIM(rho_list[i], drho_list[i], M=M, eps=eps)
                I_tp = np.zeros((para_num, para_num))
                for pm in range(para_num):
                    for pn in range(para_num):
                        I_tp[pm][pn] = dp_list[i][pm]*dp_list[i][pn]/p_list[i]**2

                F_tot = np.linalg.pinv(F_tp+I_tp)
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = F_tot[pj][pk]
            
            res = np.zeros([para_num,para_num])
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                    arr = p*F_ij
                    for si in reversed(range(para_num)):
                        arr = simps(arr, x[si])
                    res[para_i][para_j] = arr
                    res[para_j][para_i] = arr
            return res
        elif btype == 2:
            F_list = [[[0.0 for i in range(len(p_list))] for j in range(para_num)] for k in range(para_num)]
            I_list = [[[0.0 for i in range(len(p_list))] for j in range(para_num)] for k in range(para_num)]
            for i in range(len(p_list)):
                F_tp = CFIM(rho_list[i], drho_list[i], M=M, eps=eps)
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = F_tp[pj][pk]
                        I_list[pj][pk][i] = dp_list[i][pj]*dp_list[i][pk]/p_list[i]**2

            F_res = np.zeros([para_num,para_num])
            I_res = np.zeros([para_num,para_num])
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                    I_ij = np.array(I_list[para_i][para_j]).reshape(p_shape)
                    arr1 = p*F_ij
                    arr2 = p*I_ij
                    for si in reversed(range(para_num)):
                        arr1 = simps(arr1, x[si])
                        arr2 = simps(arr2, x[si])
                    F_res[para_i][para_j] = arr1
                    F_res[para_j][para_i] = arr1
                    I_res[para_i][para_j] = arr2
                    I_res[para_j][para_i] = arr2
            return np.linalg.pinv(F_res+I_res)
        else:
            raise NameError("NameError: btype should be choosen in {1, 2}")

def QVTB(x, p, dp, rho, drho, btype=1, dtype="SLD", eps=1e-8):
    para_num = len(x)
    p_num = len(p)
    
    if para_num == 1:
        if type(drho[0]) == list:
            drho = [drho[i][0] for i in range(p_num)]
        if type(dp[0]) == list  or type(dp[0]) == np.ndarray:
            dp = [dp[i][0] for i in range(p_num)]

        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = QFIM(rho[m], [drho[m]], dtype=dtype, eps=eps)
            
        if btype == 1:
            I_tp = [np.real(dp[i]*dp[i]/p[i]**2) for i in range(p_num)]
            arr = [p[j]/(I_tp[j]+F_tp[j]) for j in range(p_num)]
            return simps(arr, x[0])
        elif btype == 2:
            arr1 = [np.real(dp[i]*dp[i]/p[i]) for i in range(p_num)]  
            I = simps(arr1, x[0])
            arr2 = [np.real(F_tp[j]*p[j]) for j in range(p_num)]
            F = simps(arr2, x[0])
            return 1.0/(I+F)
        else:
            raise NameError("NameError: btype should be choosen in {1, 2}")
    else:
        #### multiparameter senario ####
        p_shape = np.shape(p)
        p_ext = extract_ele(p, para_num)
        dp_ext = extract_ele(dp, para_num)
        rho_ext = extract_ele(rho, para_num)
        drho_ext = extract_ele(drho, para_num)

        p_list, dp_list, rho_list, drho_list,  = [], [], [], []
        for p_ele, dp_ele, rho_ele, drho_ele  in zip(p_ext, dp_ext, rho_ext, drho_ext):
            p_list.append(p_ele)
            dp_list.append(dp_ele)
            rho_list.append(rho_ele)
            drho_list.append(drho_ele)
   
        if btype == 1:
            F_list = [[[0.0 for i in range(len(p_list))] for j in range(para_num)] for k in range(para_num)]
            I_list = [[[0.0 for i in range(len(p_list))] for j in range(para_num)] for k in range(para_num)]
            for i in range(len(p_list)):
                F_tp = QFIM(rho_list[i], drho_list[i], dtype=dtype, eps=eps)
                I_tp = np.zeros((para_num, para_num))
                for pm in range(para_num):
                    for pn in range(para_num):
                        I_tp[pm][pn] = dp_list[i][pm]*dp_list[i][pn]/p_list[i]**2

                F_tot = np.linalg.pinv(F_tp+I_tp)
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = F_tot[pj][pk]
            
            res = np.zeros([para_num,para_num])
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                    arr = p*F_ij
                    for si in reversed(range(para_num)):
                        arr = simps(arr, x[si])
                    res[para_i][para_j] = arr
                    res[para_j][para_i] = arr
            return res
        elif btype == 2:
            F_list = [[[0.0 for i in range(len(p_list))] for j in range(para_num)] for k in range(para_num)]
            I_list = [[[0.0 for i in range(len(p_list))] for j in range(para_num)] for k in range(para_num)]
            for i in range(len(p_list)):
                F_tp = QFIM(rho_list[i], drho_list[i], dtype=dtype, eps=eps)
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = F_tp[pj][pk]
                        I_list[pj][pk][i] = dp_list[i][pj]*dp_list[i][pk]/p_list[i]**2

            F_res = np.zeros([para_num,para_num])
            I_res = np.zeros([para_num,para_num])
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                    I_ij = np.array(I_list[para_i][para_j]).reshape(p_shape)
                    arr1 = p*F_ij
                    arr2 = p*I_ij
                    for si in reversed(range(para_num)):
                        arr1 = simps(arr1, x[si])
                        arr2 = simps(arr2, x[si])
                    F_res[para_i][para_j] = arr1
                    F_res[para_j][para_i] = arr1
                    I_res[para_i][para_j] = arr2
                    I_res[para_j][para_i] = arr2
            return np.linalg.pinv(F_res+I_res)
        else:
            raise NameError("NameError: btype should be choosen in {1, 2}")

def OBB_func(x, y, t, J, F):
    interp_J = interpolate.interp1d(t, (J))
    interp_F = interpolate.interp1d(t, (F))
    J_tp, F_tp = interp_J(x), interp_F(x)
    return np.vstack((y[1],-J_tp*y[1]+F_tp*y[0]-J_tp))

def boundary_condition(ya, yb):
    return np.array([ya[1]+1.0, yb[1]+1.0])

def OBB(x, p, dp, rho, drho, d2rho, dtype="SLD", eps=1e-8):
    #### single parameter senario ####
    p_num = len(p)

    if type(drho[0]) == list:
        drho = [drho[i][0] for i in range(p_num)]
    if type(d2rho[0]) == list:
        d2rho = [d2rho[i][0] for i in range(p_num)]
    if type(dp[0]) == list  or type(dp[0]) == np.ndarray:
        dp = [dp[i][0] for i in range(p_num)]
    if type(x[0]) != float or type(x[0]) != int:
        x = x[0]
    
    F, J = np.zeros(p_num), np.zeros(p_num)
    bias, dbias = np.zeros(p_num), np.zeros(p_num)
    for m in range(p_num):
        f, LD = QFIM(rho[m], [drho[m]], dtype=dtype, exportLD=True, eps=eps)
        F[m] = f
        term1 = np.dot(np.dot(d2rho[m], d2rho[m]), LD)
        term2 = np.dot(np.dot(LD, LD), drho[m])
        dF = np.real(np.trace(2*term1-term2))
        J[m] = dp[m]/p[m] - dF/f

    y_guess = np.zeros((2, x.size))
    fun = lambda m, n: OBB_func(m, n, x, J, F)
    result = solve_bvp(fun, boundary_condition, x, y_guess)
    res = result.sol(x)
    bias, dbias = res[0], res[1]
    
    value = [p[i]*((1+dbias[i])**2/F[i] + bias[i]**2) for i in range(p_num)]
    return simps(value, x)
