using QuanEstimation
using LinearAlgebra

# initial state
psi0 = [1., 0., 0., 1.]/sqrt(2)
rho0 = psi0*psi0'
# free Hamiltonian
omega1, omega2, g = 1.0, 1.0, 0.1
sx = [0. 1.; 1. 0.0im]
sy = [0. -im; im 0.]
sz = [1. 0.0im; 0. -1.]
H0 = omega1*kron(sz, I(2)) + omega2*kron(I(2), sz) + g*kron(sx, sx)
# derivatives of the free Hamiltonian with respect to omega2 and g
dH = [kron(I(2), sz), kron(sx, sx)]
# dissipation
decay = [[kron(sz, I(2)), 0.05], [kron(I(2), sz), 0.05]]
# measurement
m1 = [1., 0., 0., 0.]
M1 = 0.85*m1*m1'
M2 = 0.1*ones(4, 4)
M = [M1, M2, I(4)-M1-M2]
# weight matrix
W = one(zeros(2, 2))
# time length for the evolution
tspan = range(0., 5., length=200)
# dynamics
rho, drho = QuanEstimation.expm(tspan, rho0, H0, dH, decay)
# calculation of the CFIM, QFIM and HCRB
f_HCRB, f_NHB = [], []
for ti in 2:length(tspan)
    # HCRB
    f_tp1 = QuanEstimation.HCRB(rho[ti], drho[ti], W)
    append!(f_HCRB, f_tp1)
    # NHB
    f_tp2 = QuanEstimation.NHB(rho[ti], drho[ti], W)
    append!(f_NHB, f_tp2)
end
