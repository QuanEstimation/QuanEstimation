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
# time length for the evolution
tspan = range(0., 10., length=1000)
# dynamics
rho, drho = QuanEstimation.expm(tspan, rho0, H0, dH, decay)
# weight matrix
W = one(zeros(2, 2))
# calculation of the CFIM, QFIM and HCRB
Im, F, f = [], [], Float64[]
for ti in 2:length(tspan)
    #CFIM
    I_tp = QuanEstimation.CFIM(rho[ti], drho[ti], M)
    append!(Im, [I_tp])
    #QFIM
    F_tp = QuanEstimation.QFIM(rho[ti], drho[ti])
    append!(F, [F_tp])
    #HCRB
    f_tp = QuanEstimation.HCRB(rho[ti], drho[ti], W)
    append!(f, f_tp)
end
