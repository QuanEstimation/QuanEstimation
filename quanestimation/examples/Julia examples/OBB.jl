using Trapz
include("../src/QuanEstimation.jl")

# initial state
rho0 = 0.5*[1.0 1.0+0.0im; 1.0 1.0]
# free Hamiltonian
B = pi/2.0
sx = [0.0 1.0; 1.0 0.0im]
sy = [0.0 -1.0im; 1.0im 0.0]
sz = [1.0 0.0im; 0.0 -1.0]
function H0_func(x)
    return 0.5*B*(sx*cos(x)+sz*sin(x))
end
function dH_func(x)
    return [0.5*B*(-sx*sin(x)+sz*cos(x))]
end
function d2H_func(x)
    return [0.5*B*(-sx*cos(x)-sz*sin(x))]
end
# dynamics
decay_opt = [zeros(ComplexF64,size(rho0)[1],size(rho0)[1])] 
gamma = [0.0]
tspan = range(0.0, stop=1.0, length=1000)
# prior distribution
function p_func(x, mu, eta)
    return exp(-(x-mu)^2/(2*eta^2))/(eta*sqrt(2*pi))
end
function dp_func(x, mu, eta)
    return -(x-mu)*exp(-(x-mu)^2/(2*eta^2))/(eta^3*sqrt(2*pi))
end
x = [range(-pi/2.0, stop=pi/2.0, length=100)].|>Vector
mu, eta = 0.0, 0.5
p_tp = [p_func(x[1][i], mu, eta) for i in 1:length(x[1])]
dp_tp = [dp_func(x[1][i], mu, eta) for i in 1:length(x[1])]
c = trapz(x[1], p_tp)
p, dp = p_tp/c, dp_tp/c

rho = Vector{Matrix{ComplexF64}}(undef, length(x[1]))
drho = Vector{Vector{Matrix{ComplexF64}}}(undef, length(x[1]))
d2rho = Vector{Vector{Matrix{ComplexF64}}}(undef, length(x[1]))
for i = 1:length(x[1]) 
    H0_tp = H0_func(x[1][i])
    dH_tp = dH_func(x[1][i])
    d2H_tp = d2H_func(x[1][i])
    rho_tp, drho_tp, d2rho_tp = QuanEstimation.secondorder_derivative(H0_tp, dH_tp, d2H_tp, rho0, decay_opt, gamma, 
                      [zeros(ComplexF64,size(rho0)[1],size(rho0)[1])], [zeros(length(tspan)-1)],tspan)
    rho[i] = rho_tp
    drho[i] = drho_tp
    d2rho[i] = d2rho_tp
end
f_OBB = QuanEstimation.OBB(x, p, dp, rho, drho, d2rho)
println(f_OBB)