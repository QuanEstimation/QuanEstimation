using LinearAlgebra
include("../src/QuanEstimation.jl")

#### spin squeezing ####

rho_CSS = [0.25 -0.35355339im -0.25; 0.35355339im 0.5 -0.35355339im; -0.25 0.35355339im 0.25]

xi = QuanEstimation.SpinSqueezing(rho_CSS; basis="Dicke", output="KU")
println(xi)

#### Target time ####
# initial state
rho0 = 0.5*[1.0 1.0+0.0im; 1.0 1.0]
# free Hamiltonian
omega0 = 1.0
sz = [1.0 0.0im; 0.0 -1.0]
H0 = 0.5 * omega0 * sz
dH = [0.5 * sz]
# measurement
M1 = 0.5*[1.0+0.0im  1.0; 1.0  1.0]
M2 = 0.5*[1.0+0.0im -1.0; -1.0  1.0]
M = [M1, M2]
# dissipation
sp = [0.0im 1.0; 0.0 0.0]
sm = [0.0im 0.0; 1.0 0.0]
decay_opt = [sp, sm] 
gamma = [0.0, 0.1]
# dynamics
tspan = range(0.0, stop=50.0, length=2000) |>Vector
rho, drho = QuanEstimation.expm(H0, dH, [zeros(ComplexF64,size(rho0)[1],size(rho0)[1])], [zeros(length(tspan)-1)], rho0, tspan, decay_opt, gamma)
drho = [drho[i][1] for i in 1:2000]
QFI = []
for ti in 2:2000
    QFI_tp = QuanEstimation.QFIM_SLD(rho[ti], drho[ti])
    append!(QFI, QFI_tp)
end
println(QFI[243])
println(tspan[243])
t = QuanEstimation.TargetTime(20.0, tspan, QuanEstimation.QFIM_SLD, rho, drho, eps=1e-8)
println(t)
