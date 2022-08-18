using QuanEstimation
using Trapz

# free Hamiltonian
function H0_func(x)
    return 0.5*B*omega0*(sx*cos(x)+sz*sin(x))
end
# derivative of the free Hamiltonian on x
function dH_func(x)
    return [0.5*B*omega0*(-sx*sin(x)+sz*cos(x))]
end
# prior distribution
function p_func(x, mu, eta)
    return exp(-(x-mu)^2/(2*eta^2))/(eta*sqrt(2*pi))
end
function dp_func(x, mu, eta)
    return -(x-mu)*exp(-(x-mu)^2/(2*eta^2))/(eta^3*sqrt(2*pi))
end

B, omega0 = 0.5*pi, 1.0
sx = [0. 1.; 1. 0.0im]
sy = [0. -im; im 0.]
sz = [1. 0.0im; 0. -1.]
# initial state
rho0 = 0.5*ones(2, 2)
# prior distribution
x = range(-0.5*pi, stop=0.5*pi, length=100) |>Vector
mu, eta = 0.0, 0.2
p_tp = [p_func(x[i], mu, eta) for i in 1:length(x)]
dp_tp = [dp_func(x[i], mu, eta) for i in 1:length(x)]
# normalization of the distribution
c = trapz(x, p_tp)
p = p_tp/c
dp = dp_tp/c
# time length for the evolution
tspan = range(0., stop=1., length=1000)
# dynamics
rho = Vector{Matrix{ComplexF64}}(undef, length(x))
drho = Vector{Vector{Matrix{ComplexF64}}}(undef, length(x))
for i = 1:length(x) 
    H0_tp = H0_func(x[i])
    dH_tp = dH_func(x[i])
    rho_tp, drho_tp = QuanEstimation.expm(tspan, rho0, H0_tp, dH_tp)
    rho[i], drho[i] = rho_tp[end], drho_tp[end]
end

# Classical Bayesian bounds
f_BCRB1 = QuanEstimation.BCRB([x], p, [], rho, drho, btype=1)
f_BCRB2 = QuanEstimation.BCRB([x], p, [], rho, drho, btype=2)
f_BCRB3 = QuanEstimation.BCRB([x], p, dp, rho, drho, btype=3)
f_VTB = QuanEstimation.VTB([x], p, dp, rho, drho)

# Quantum Bayesian bounds
f_BQCRB1 = QuanEstimation.BQCRB([x], p, [], rho, drho, btype=1)
f_BQCRB2 = QuanEstimation.BQCRB([x], p, [], rho, drho, btype=2)
f_BQCRB3 = QuanEstimation.BQCRB([x], p, dp, rho, drho, btype=3)
f_QVTB = QuanEstimation.QVTB([x], p, dp, rho, drho)
f_QZZB = QuanEstimation.QZZB([x], p, rho)
