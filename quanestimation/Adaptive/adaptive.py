import numpy as np
from julia import Main
from quanestimation.Common.common import extract_ele
from quanestimation.MeasurementOpt.MeasurementStruct import MeasurementOpt

class adaptive():
    def __init__(self, x, p, tspan, rho0, H, dH, decay=[], W=[], max_episode=1000, eps=1e-8):

        self.x = x
        self.p = p
        self.tspan = tspan
        self.rho0 = np.array(rho0, dtype=np.complex128)
        self.decay = decay

        if decay == []:
            decay_opt = [np.zeros((len(self.rho0), len(self.rho0)))]
            self.gamma = [0.0]
        else:
            decay_opt = [decay[i][0] for i in range(len(decay))]
            self.gamma = [decay[i][1] for i in range(len(decay))]
        self.decay_opt = [np.array(x, dtype=np.complex128) for x in decay_opt]

        self.max_episode = max_episode
        self.eps = eps

        para_num = len(x)
        if para_num == 1:
            H = [np.array(x, dtype=np.complex128) for x in H]
            dH = [np.array(x, dtype=np.complex128) for x in dH]
            self.H, self.dH = H, dH
        else:
            H_ext = extract_ele(H, para_num)
            dH_ext = extract_ele(dH, para_num)

            H_list, dH_list = [], []
            for H_ele, dH_ele in zip(H_ext, dH_ext):
                H_list.append(H_ele)
                dH_list.append(dH_ele)
            H_list = [np.array(x, dtype=np.complex128) for x in H_list]
            dH_list = [np.array(x, dtype=np.complex128) for x in dH_list]

            dH = []
            for i in range(para_num):
                dH.append([dH_list[j][i] for j in range(len(H_list))])

            self.H = zip(H_list)
            self.dH = zip(dH)

    def CFIM(self, M, W=[], save_file=False):
        if W == []:
            W = np.eye(len(self.x))
        self.W = W

        apt = Main.QuanEstimation.adapt(self.x, self.H, self.dH, self.rho0, self.tspan, self.decay_opt, \
                                     self.gamma, self.W, self.eps)
        pout, xout = Main.QuanEstimation.adaptive(apt, M, self.p, self.max_episode, save_file) 

        return pout, xout

    def Mopt(self, H0, dH0, W=[]):
        m = MeasurementOpt(self.tspan, self.rho0, H0, dH0, decay=self.decay, mtype="projection", minput=[], method="DE")
        m.CFIM(save_file=False)
