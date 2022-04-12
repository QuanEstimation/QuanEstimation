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
mu = 0.0
eta = 0.2
p_tp = [p_func(x[1][i], mu, eta) for i in 1:length(x[1])]
dp_tp = [dp_func(x[1][i], mu, eta) for i in 1:length(x[1])]
c = trapz(x[1], p_tp)
p = p_tp/c
dp = dp_tp/c

rho = Vector{Matrix{ComplexF64}}(undef, length(x[1]))
drho = Vector{Vector{Matrix{ComplexF64}}}(undef, length(x[1]))
for i = 1:length(x[1]) 
    H0_tp = H0_func(x[1][i])
    dH_tp = dH_func(x[1][i])
    rho_tp, drho_tp = QuanEstimation.expm(H0_tp, dH_tp, [zeros(ComplexF64,size(rho0)[1],size(rho0)[1])], [zeros(length(tspan)-1)], rho0, tspan, decay_opt, gamma)
    rho[i] = rho_tp[end]
    drho[i] = drho_tp[end]
end

f_BCRB1 = QuanEstimation.BCRB(x, p, rho, drho, M=nothing, btype=1)
f_BCRB2 = QuanEstimation.BCRB(x, p, rho, drho, M=nothing, btype=2)
f_VTB1 = QuanEstimation.VTB(x, p, dp, rho, drho, M=nothing, btype=1)
f_VTB2 = QuanEstimation.VTB(x, p, dp, rho, drho, M=nothing, btype=2)

f_BQCRB1 = QuanEstimation.BQCRB(x, p, rho, drho, btype=1)
f_BQCRB2 = QuanEstimation.BQCRB(x, p, rho, drho, btype=2)
f_QVTB1 = QuanEstimation.QVTB(x, p, dp, rho, drho, btype=1)
f_QVTB2 = QuanEstimation.QVTB(x, p, dp, rho, drho, btype=2)
f_QZZB = QuanEstimation.QZZB(x, p, rho)

for f in [f_BCRB1,f_BCRB2,f_VTB1,f_VTB2,f_BQCRB1,f_BQCRB2,f_QVTB1,f_QVTB2,f_QZZB]
    println(f)
end