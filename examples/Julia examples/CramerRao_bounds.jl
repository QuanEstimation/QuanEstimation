using QuanEstimation

# initial state
rho0 = 0.5*ones(2, 2)
# free Hamiltonian
omega = 1.0
sx = [0. 1.; 1. 0.0im]
sy = [0. -im; im 0.]
sz = [1. 0.0im; 0. -1.]
H0 = 0.5*omega*sz
# derivative of the free Hamiltonian on omega
dH = [0.5*sz]
# dissipation
sp = [0. 1.; 0. 0.0im]
sm = [0. 0.; 1. 0.0im]
decay = [[sp, 0.0], [sm, 0.1]]
# measurement
M1 = 0.5*[1.0+0.0im  1.; 1.  1.]
M2 = 0.5*[1.0+0.0im -1.; -1.  1.]
M = [M1, M2]
# time length for the evolution
tspan = range(0., 50., length=2000)
# dynamics
rho, drho = QuanEstimation.expm(tspan, rho0, H0, dH, decay)
# calculation of the CFI and QFI
Im, F = Float64[], Float64[]
for ti in 2:length(tspan)
    # CFI
    I_tp = QuanEstimation.CFIM(rho[ti], drho[ti], M)
    append!(Im, I_tp)
    # QFI
    F_tp = QuanEstimation.QFIM(rho[ti], drho[ti])
    append!(F, F_tp)
end
