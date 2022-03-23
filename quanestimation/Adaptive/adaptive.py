import numpy as np
from scipy.integrate import simps
from itertools import product
import quanestimation.Adaptive as apt
from quanestimation.Common.common import extract_ele, SIC
from quanestimation.MeasurementOpt.MeasurementStruct import MeasurementOpt
from quanestimation.Dynamics.dynamics import Lindblad
from quanestimation.AsymptoticBound.CramerRao import QFIM, CFIM


class adaptive:
    def __init__(self, x, p, rho0, max_episode=1000, eps=1e-8):

        self.x = x
        self.p = p
        self.rho0 = np.array(rho0, dtype=np.complex128)
        self.max_episode = max_episode
        self.eps = eps
        self.para_num = len(x)

    def dynamics(self, tspan, H, dH, Hc=[], ctrl=[], decay=[]):
        self.tspan = tspan
        self.H = H
        self.dH = dH
        self.Hc = Hc
        self.ctrl = ctrl
        self.decay = decay

        self.dynamic_type = "dynamics"

    def kraus(self, K, dK):
        self.K = K
        self.dK = dK

        self.dynamic_type = "kraus"

    def CFIM(self, M=[], W=[], savefile=False):
        if M == []:
            M = SIC(len(self.rho0))
        if W == []:
            W = np.eye(len(self.x))
        self.W = W

        if self.dynamic_type == "dynamics":
            adaptive_dynamics(
                self.x,
                self.p,
                M,
                self.tspan,
                self.rho0,
                self.H,
                self.dH,
                self.decay,
                self.Hc,
                self.ctrl,
                W,
                self.max_episode,
                self.eps,
                savefile,
            )
        elif self.dynamic_type == "kraus":
            adaptive_kraus(
                self.x,
                self.p,
                M,
                self.rho0,
                self.K,
                self.dK,
                W,
                self.max_episode,
                self.eps,
                savefile,
            )
        else:
            raise ValueError(
                "{!r} is not a valid value for type of dynamics, supported values are 'dynamics' and 'kraus'.".format(
                    self.dynamic_type
                )
            )

    def Mopt(self, W=[]):
        if W == []:
            W = np.identity(self.para_num)
        else:
            W = W

        if self.dynamic_type == "dynamics":
            if self.para_num == 1:
                F = []
                for i in range(len(self.H)):
                    dynamics = Lindblad(
                        self.tspan,
                        self.rho0,
                        self.H[i],
                        self.dH[i],
                        decay=self.decay,
                        Hc=self.Hc,
                        ctrl=self.ctrl,
                    )
                    rho_tp, drho_tp = dynamics.expm()
                    rho, drho = rho_tp[-1], drho_tp[-1]
                    F_tp = QFIM(rho, drho)
                    F.append(F_tp)
                idx = np.argmax(F)
                H_res, dH_res = self.H[idx], self.dH[idx]
            else:
                p_ext = extract_ele(self.p, self.para_num)
                H_ext = extract_ele(self.H, self.para_num)
                dH_ext = extract_ele(self.dH, self.para_num)

                p_list, H_list, dH_list = [], [], []
                for p_ele, H_ele, dH_ele in zip(p_ext, H_ext, dH_ext):
                    p_list.append(p_ele)
                    H_list.append(H_ele)
                    dH_list.append(dH_ele)

                F = []
                for i in range(len(p_list)):
                    dynamics = Lindblad(
                        self.tspan,
                        self.rho0,
                        self.H_list[i],
                        self.dH_list[i],
                        decay=self.decay,
                        Hc=self.Hc,
                        ctrl=self.ctrl,
                    )
                    rho_tp, drho_tp = dynamics.expm()
                    rho, drho = rho_tp[-1], drho_tp[-1]
                    F_tp = QFIM(rho, drho)
                    if np.linalg.det(F_tp) < self.eps:
                        F.append(self.eps)
                    else:
                        F.append(1.0 / np.trace(np.dot(W, np.linalg.inv(F_tp))))
                idx = np.argmax(F)
                H_res, dH_res = self.H_list[idx], self.dH_list[idx]
            m = MeasurementOpt(mtype="projection", minput=[], method="DE")
            m.dynamics(
                self.tspan,
                self.rho0,
                H_res,
                dH_res,
                Hc=self.Hc,
                ctrl=self.ctrl,
                decay=self.decay,
            )
            m.CFIM(W=W, savefile=False)
        elif self.dynamic_type == "kraus":
            if self.para_num == 1:
                F = []
                for hi in range(len(self.K)):
                    rho_tp = sum(
                        [np.dot(Ki, np.dot(self.rho0, Ki.conj().T)) for Ki in K[hi]]
                    )
                    drho_tp = sum(
                        [
                            np.dot(dKi, np.dot(self.rho0, Ki.conj().T))
                            + np.dot(Ki, np.dot(self.rho0, dKi.conj().T))
                            for (Ki, dKi) in zip(self.K[hi], self.dK[hi])
                        ]
                    )
                    F_tp = QFIM(rho_tp, drho_tp)
                    F.append(F_tp)

                idx = np.argmax(F)
                K_res, dK_res = self.K[idx], self.dK[idx]
            else:
                p_shape = np.shape(self.p)

                p_ext = extract_ele(self.p, self.para_num)
                K_ext = extract_ele(self.K, self.para_num)
                dK_ext = extract_ele(self.dK, self.para_num)

                p_list, K_list, dK_list = [], [], []
                for K_ele, dK_ele in zip(K_ext, dK_ext):
                    p_list.append(p_ele)
                    K_list.append(K_ele)
                    dK_list.append(dK_ele)
                F = []
                for hi in range(len(p_list)):
                    rho_tp = sum(
                        [np.dot(Ki, np.dot(self.rho0, Ki.conj().T)) for Ki in K_list[hi]]
                    )
                    dK_reshape = [
                        [dK_list[hi][i][j] for i in range(self.k_num)]
                        for j in range(self.para_num)
                    ]
                    drho_tp = [
                        sum(
                            [
                                np.dot(dKi, np.dot(self.rho0, Ki.conj().T))
                                + np.dot(Ki, np.dot(self.rho0, dKi.conj().T))
                                for (Ki, dKi) in zip(K_list[hi], dKj)
                            ]
                        )
                        for dKj in dK_reshape
                    ]
                    F_tp = QFIM(rho_tp, drho_tp)
                    if np.linalg.det(F_tp) < self.eps:
                        F.append(self.eps)
                    else:
                        F.append(1.0 / np.trace(np.dot(W, np.linalg.inv(F_tp))))
                F = np.array(F).reshape(p_shape)
                idx = np.where(np.array(F) == np.max(np.array(F)))
                K_res, dK_res = self.K_list[idx], self.dK_list[idx]
            m = MeasurementOpt(mtype="projection", minput=[], method="DE")
            m.kraus(self.rho0, K_res, dK_res, decay=self.decay)
            m.CFIM(W=W, savefile=False)
        else:
            raise ValueError(
                "{!r} is not a valid value for type of dynamics, supported values are 'dynamics' and 'kraus'.".format(
                    self.dynamic_type
                )
            )


def adaptive_dynamics(
    x, p, M, tspan, rho0, H, dH, decay, Hc, ctrl, W, max_episode, eps, savefile
):

    para_num = len(x)
    dim = np.shape(rho0)[0]
    if para_num == 1:
        #### singleparameter senario ####
        p_num = len(p)

        F = []
        for hi in range(p_num):
            dynamics = Lindblad(
                tspan, rho0, H[hi], dH[hi], decay=decay, Hc=Hc, ctrl=ctrl
            )
            rho_tp, drho_tp = dynamics.expm()
            F_tp = CFIM(rho_tp[-1], drho_tp[-1], M)
            F.append(F_tp)

        idx = np.argmax(F)
        x_opt = x[0][idx]
        print("The optimal parameter is %f" % x_opt)

        u = 0.0
        y, xout, pout = [], [], []
        for ei in range(max_episode):
            rho = [np.zeros((dim, dim), dtype=np.complex128) for i in range(p_num)]
            for hj in range(p_num):
                x_idx = np.argmin(np.abs(x[0] - (x[0][hj] + u)))
                H_tp = H[x_idx]
                dH_tp = dH[x_idx]
                dynamics = Lindblad(
                    tspan, rho0, H_tp, dH_tp, decay=decay, Hc=Hc, ctrl=ctrl
                )
                rho_tp, drho_tp = dynamics.expm()
                rho[hj] = rho_tp[-1]
            print("The tunable parameter is %f" % u)
            res_exp = input("Please enter the experimental result: ")
            res_exp = int(res_exp)
            pyx = np.zeros(p_num)
            for xi in range(p_num):
                pyx[xi] = np.real(np.trace(np.dot(rho[xi], M[res_exp])))

            arr = [pyx[m] * p[m] for m in range(p_num)]
            py = simps(arr, x[0])
            p_update = pyx * p / py
            p = p_update
            p_idx = np.argmax(p)
            x_out = x[0][p_idx]
            print("The estimator is %s (%d episodes)" % (x_out, ei))
            u = x_opt - x_out

            if (ei + 1) % 50 == 0:
                if (x_out + u) > x[0][-1] and (x_out + u) < x[0][0]:
                    raise ValueError("please increase the regime of the parameters.")

            xout.append(x_out)
            y.append(res_exp)
            pout.append(p)
        #### save file ####
        if savefile == False:
            np.save("xout", xout)
            np.save("y", y)
            np.save("pout", p)
        else:
            np.save("xout", xout)
            np.save("y", y)
            np.save("pout", pout)
    else:
        #### miltiparameter senario ####
        p_shape = np.shape(p)
        x_list = []
        for x_tp in product(*x):
            x_list.append([x_tp[i] for i in range(para_num)])

        p_ext = extract_ele(p, para_num)
        H_ext = extract_ele(H, para_num)
        dH_ext = extract_ele(dH, para_num)

        p_list, H_list, dH_list = [], [], []
        for p_ele, H_ele, dH_ele in zip(p_ext, H_ext, dH_ext):
            p_list.append(p_ele)
            H_list.append(H_ele)
            dH_list.append(dH_ele)

        F = []
        for hi in range(len(p_list)):
            dynamics = Lindblad(
                tspan, rho0, H_list[hi], dH_list[hi], decay=decay, Hc=Hc, ctrl=ctrl
            )
            rho_tp, drho_tp = dynamics.expm()
            F_tp = CFIM(rho_tp[-1], drho_tp[-1], M)
            if np.linalg.det(F_tp) < eps:
                F.append(eps)
            else:
                F.append(1.0 / np.trace(np.dot(W, np.linalg.inv(F_tp))))
        F = np.array(F).reshape(p_shape)
        idx = np.unravel_index(F.argmax(), F.shape)
        x_opt = [x[i][idx[i]] for i in range(para_num)]
        print("The optimal parameter are %s" % (x_opt))

        u = [0.0 for i in range(para_num)]
        y, xout, pout = [], [], []
        for ei in range(max_episode):
            rho = [
                np.zeros((dim, dim), dtype=np.complex128) for i in range(len(p_list))
            ]
            for hj in range(len(p_list)):
                idx_list = [
                    np.argmin(np.abs(x[i] - (x_list[hj][i] + u[i])))
                    for i in range(para_num)
                ]
                x_idx = int(
                    sum(
                        [
                            idx_list[i] * np.prod(np.array(p_shape[(i + 1) :]))
                            for i in range(para_num)
                        ]
                    )
                )
                H_tp = H_list[x_idx]
                dH_tp = dH_list[x_idx]
                dynamics = Lindblad(
                    tspan, rho0, H_tp, dH_tp, decay=decay, Hc=Hc, ctrl=ctrl
                )
                rho_tp, drho_tp = dynamics.expm()
                rho[hj] = rho_tp[-1]
            print("The tunable parameter are %s" % (u))
            res_exp = input("Please enter the experimental result: ")
            res_exp = int(res_exp)
            pyx_list = np.zeros(len(p_list))
            for xi in range(len(p_list)):
                pyx_list[xi] = np.real(np.trace(np.dot(rho[xi], M[res_exp])))
            pyx = pyx_list.reshape(p_shape)
            arr = p * pyx
            for si in reversed(range(para_num)):
                arr = simps(arr, x[si])
            py = arr
            p_update = p * pyx / py
            p = p_update
            p_idx = np.unravel_index(p.argmax(), p.shape)
            x_out = [x[i][p_idx[i]] for i in range(para_num)]

            print("The estimator is %s (%d episodes)" % (x_out, ei))
            u = np.array(x_opt) - np.array(x_out)

            if (ei + 1) % 50 == 0:
                for un in range(para_num):
                    if (x_out[un] + u[un]) > x[un][-1] and (x_out[un] + u[un]) < x[un][
                        0
                    ]:
                        raise ValueError(
                            "please increase the regime of the parameters."
                        )
            xout.append(x_out)
            y.append(res_exp)
            pout.append(p)
        #### save file ####
        if savefile == False:
            np.save("xout", xout)
            np.save("y", y)
            np.save("pout", p)
        else:
            np.save("xout", xout)
            np.save("y", y)
            np.save("pout", pout)


def adaptive_kraus(x, p, M, rho0, K, dK, W, max_episode, eps, savefile):
    para_num = len(x)
    dim = np.shape(rho0)[0]
    if para_num == 1:
        #### singleparameter senario ####
        p_num = len(p)
        F = []
        for hi in range(p_num):
            rho_tp = sum([np.dot(Ki, np.dot(rho0, Ki.conj().T)) for Ki in K[hi]])
            drho_tp = [
                sum(
                    [
                        (
                            np.dot(dKi, np.dot(rho0, Ki.conj().T))
                            + np.dot(Ki, np.dot(rho0, dKi.conj().T))
                        )
                        for (Ki, dKi) in zip(K[hi], dKj)
                    ]
                )
                for dKj in dK[hi]
            ]
            F_tp = CFIM(rho_tp, drho_tp, M)
            F.append(F_tp)

        idx = np.argmax(F)
        x_opt = x[0][idx]
        print("The optimal parameter is %s" % x_opt)

        u = 0.0
        y, xout, pout = [], [], []
        for ei in range(max_episode):
            rho = [np.zeros((dim, dim), dtype=np.complex128) for i in range(p_num)]
            for hj in range(p_num):
                x_idx = np.argmin(np.abs(x[0] - (x[0][hj] + u)))
                rho_tp = sum([np.dot(Ki, np.dot(rho0, Ki.conj().T)) for Ki in K[x_idx]])
                rho[hj] = rho_tp
            print("The tunable parameter is %s" % u)
            res_exp = input("Please enter the experimental result: ")
            res_exp = int(res_exp)
            pyx = np.zeros(p_num)
            for xi in range(p_num):
                pyx[xi] = np.real(np.trace(np.dot(rho[xi], M[res_exp])))

            arr = [pyx[m] * p[m] for m in range(p_num)]
            py = simps(arr, x[0])
            p_update = pyx * p / py
            p = p_update
            p_idx = np.argmax(p)
            x_out = x[0][p_idx]
            print("The estimator is %s (%d episodes)" % (x_out, ei))
            u = x_opt - x_out

            if (ei + 1) % 50 == 0:
                if (x_out + u) > x[0][-1] and (x_out + u) < x[0][0]:
                    raise ValueError("please increase the regime of the parameters.")

            xout.append(x_out)
            y.append(res_exp)
            pout.append(p)
        #### save file ####
        if savefile == False:
            np.save("xout", xout)
            np.save("y", y)
            np.save("pout", p)
        else:
            np.save("xout", xout)
            np.save("y", y)
            np.save("pout", pout)
    else:
        #### miltiparameter senario ####
        p_shape = np.shape(p)
        x_list = []
        for x_tp in product(*x):
            x_list.append([x_tp[i] for i in range(para_num)])

        p_ext = extract_ele(p, para_num)
        K_ext = extract_ele(K, para_num)
        dK_ext = extract_ele(dK, para_num)

        p_list, K_list, dK_list = [], [], []
        for p_ele, K_ele, dK_ele in zip(p_ext, K_ext, dK_ext):
            p_list.append(p_ele)
            K_list.append(K_ele)
            dK_list.append(dK_ele)
        k_num = len(K_list[0])
        F = []
        for hi in range(len(p_list)):
            rho_tp = sum([np.dot(Ki, np.dot(rho0, Ki.conj().T)) for Ki in K_list[hi]])
            dK_reshape = [
                        [dK_list[hi][i][j] for i in range(k_num)]
                        for j in range(para_num)
                    ]
            drho_tp = [
                sum(
                    [
                        np.dot(dKi, np.dot(rho0, Ki.conj().T))
                        + np.dot(Ki, np.dot(rho0, dKi.conj().T))
                        for (Ki, dKi) in zip(K_list[hi], dKj)
                    ]
                )
                for dKj in dK_reshape
            ]
            F_tp = CFIM(rho_tp, drho_tp, M)
            if np.linalg.det(F_tp) < eps:
                F.append(eps)
            else:
                F.append(1.0 / np.trace(np.dot(W, np.linalg.inv(F_tp))))
        F = np.array(F).reshape(p_shape)
        idx = np.unravel_index(F.argmax(), F.shape)
        x_opt = [x[i][idx[i]] for i in range(para_num)]
        print("The optimal parameter is %s" % (x_opt))
        u = [0.0 for i in range(para_num)]
        y, xout, pout = [], [], []
        for ei in range(max_episode):
            rho = [
                np.zeros((dim, dim), dtype=np.complex128) for i in range(len(p_list))
            ]
            for hj in range(len(p_list)):
                idx_list = [
                    np.argmin(np.abs(x[i] - (x_list[hj][i] + u[i])))
                    for i in range(para_num)
                ]
                x_idx = int(
                    sum(
                        [
                            idx_list[i] * np.prod(np.array(p_shape[(i + 1) :]))
                            for i in range(para_num)
                        ]
                    )
                )
                rho[hj] = sum(
                    [np.dot(Ki, np.dot(rho0, Ki.conj().T)) for Ki in K_list[x_idx]]
                )
            print("The tunable parameter are %s" % (u))
            res_exp = input("Please enter the experimental result: ")
            res_exp = int(res_exp)
            pyx_list = np.zeros(len(p_list))
            for xi in range(len(p_list)):
                pyx_list[xi] = np.real(np.trace(np.dot(rho[xi], M[res_exp])))
            pyx = pyx_list.reshape(p_shape)
            arr = p * pyx
            for si in reversed(range(para_num)):
                arr = simps(arr, x[si])
            py = arr
            p_update = p * pyx / py
            p = p_update
            p_idx = np.unravel_index(p.argmax(), p.shape)
            x_out = [x[i][p_idx[i]] for i in range(para_num)]

            print("The estimator are %s (%d episodes)" % (x_out, ei))
            u = np.array(x_opt) - np.array(x_out)

            if (ei + 1) % 50 == 0:
                for un in range(para_num):
                    if (x_out[un] + u[un]) > x[un][-1] and (x_out[un] + u[un]) < x[un][
                        0
                    ]:
                        raise ValueError(
                            "please increase the regime of the parameters."
                        )
            xout.append(x_out)
            y.append(res_exp)
            pout.append(p)
        #### save file ####
        if savefile == False:
            np.save("xout", xout)
            np.save("y", y)
            np.save("pout", p)
        else:
            np.save("xout", xout)
            np.save("y", y)
            np.save("pout", pout)
