from quanestimation import *
import numpy as np
import os

# initial state
rho0 = np.zeros((6, 6), dtype=np.complex128)
rho0[0][0], rho0[0][4], rho0[4][0], rho0[4][4] = 0.5, 0.5, 0.5, 0.5
# Hamiltonian
sx = np.array([[0.0, 1.0], [1.0, 0.0]])
sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
sz = np.array([[1.0, 0.0], [0.0, -1.0]])
ide2 = np.array([[1.0, 0.0], [0.0, 1.0]])
s1 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]) / np.sqrt(2)
s2 = np.array([[0.0, -1.0j, 0.0], [1.0j, 0.0, -1.0j], [0.0, 1.0j, 0.0]]) / np.sqrt(2)
s3 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
ide3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
I1, I2, I3 = np.kron(ide3, sx), np.kron(ide3, sy), np.kron(ide3, sz)
S1, S2, S3 = np.kron(s1, ide2), np.kron(s2, ide2), np.kron(s3, ide2)
B1, B2, B3 = 5.0e-4, 5.0e-4, 5.0e-4
cons = 100
D = (2 * np.pi * 2.87 * 1000) / cons
gS = (2 * np.pi * 28.03 * 1000) / cons
gI = (2 * np.pi * 4.32) / cons
A1 = (2 * np.pi * 3.65) / cons
A2 = (2 * np.pi * 3.03) / cons
H0 = (
    D * np.kron(np.dot(s3, s3), ide2)
    + gS * (B1 * S1 + B2 * S2 + B3 * S3)
    + gI * (B1 * I1 + B2 * I2 + B3 * I3)
    + +A1 * (np.kron(s1, sx) + np.kron(s2, sy))
    + A2 * np.kron(s3, sz)
)
dH = [gS * S1 + gI * I1, gS * S2 + gI * I2, gS * S3 + gI * I3]
Hc = [S1, S2, S3]
# dissipation
decay = [[S3, 2 * np.pi / cons]]
# dynamics
tspan = np.linspace(0.0, 2.0, 4000)
# initial control coefficients
cnum = 10
Hc_coeff = [np.zeros(cnum), np.zeros(cnum), np.zeros(cnum)]
ctrl0 = [np.array(Hc_coeff)]
ctrl_opt = Hc_coeff
psi_opt = []

episodes = 5
for ei in range(episodes):
    # state optimization
    DE_paras = {
        "popsize": 10,
        "psi0": psi_opt,
        "max_episode": 50,
        "c": 1.0,
        "cr": 0.5,
        "seed": 1234,
    }
    state = StateOpt(savefile=True, method="DE", **DE_paras)
    state.dynamics(
        tspan,
        H0,
        dH,
        Hc=Hc,
        ctrl=ctrl_opt,
        decay=decay,
    )
    state.QFIM()
    #  load f and rename
    f_load = open("f.csv", "r")
    f_load = "".join([i for i in f_load])
    f_save = open("f_Sopt%d.csv" % ei, "w")
    f_save.writelines(f_load)
    f_save.close()

    s_load = open("states.csv", "r")
    s_load = "".join([i for i in s_load])
    s_save = open("states_Sopt%d.csv" % ei, "w")
    s_save.writelines(s_load)
    s_save.close()
    if os.path.exists("f.csv"):
        os.remove("f.csv")

    # control optimization
    psi_save = np.genfromtxt("states.csv", dtype=np.complex128)
    csv2npy_states(psi_save)
    psi_opt = np.load("states.npy")
    psi_opt = psi_opt.reshape(1, len(rho0))[0]
    rho_opt = np.dot(
        psi_opt.reshape(len(rho0), 1), psi_opt.reshape(1, len(rho0)).conj()
    )
    psi_opt = [psi_opt]

    DE_paras = {
        "popsize": 10,
        "ctrl0": ctrl0,
        "max_episode": 50,
        "c": 1.0,
        "cr": 0.5,
        "seed": 1234,
    }
    control = ControlOpt(
        tspan,
        rho_opt,
        H0,
        dH,
        Hc,
        decay=decay,
        ctrl_bound=[-0.2, 0.2],
        savefile=True,
        method="DE",
        **DE_paras
    )
    control.QFIM()
    f_load = open("f.csv", "r")
    f_load = "".join([i for i in f_load])
    f_save = open("f_Copt%d.csv" % ei, "w")
    f_save.writelines(f_load)
    f_save.close()

    c_load = open("controls.csv", "r")
    c_load = "".join([i for i in c_load])
    c_save = open("controls_Copt%d.csv" % ei, "w")
    c_save.writelines(c_load)
    c_save.close()
    if os.path.exists("f.csv"):
        os.remove("f.csv")

    ctrl_save = np.genfromtxt("controls.csv")
    csv2npy_controls(ctrl_save, len(Hc))
    ctrl_opt = np.load("controls.npy")[0]
    ctrl_opt = [ctrl_opt[i] for i in range(len(Hc))]
    ctrl0 = [np.array(ctrl_opt)]

# measurement optimization
psi_save = np.genfromtxt("states.csv", dtype=np.complex128)
csv2npy_states(psi_save)
psi_opt = np.load("states.npy")
rho_opt = np.dot(psi_opt.reshape(len(rho0), 1), psi_opt.reshape(1, len(rho0)).conj())

ctrl_save = np.genfromtxt("controls.csv")
csv2npy_controls(ctrl_save, len(Hc))
ctrl_opt = np.load("controls.npy")[0]

DE_paras = {
    "popsize": 10,
    "measurement0": [],
    "max_episode": 1000,
    "c": 1.0,
    "cr": 0.5,
    "seed": 1234,
}
m = MeasurementOpt(
    mtype="projection", minput=[], savefile=True, method="DE", **DE_paras
)
m.dynamics(
    tspan,
    rho_opt,
    H0,
    dH,
    Hc=Hc,
    ctrl=ctrl_opt,
    decay=decay,
)
m.CFIM()
