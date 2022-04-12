using Trapz
include("../src/QuanEstimation.jl")

# initial state
rho0 = 0.5*[1.0 1.0+0.0im; 1.0 1.0]
# free Hamiltonian
sx = [0.0 1.0; 1.0 0.0im]
sy = [0.0 -1.0im; 1.0im 0.0]
sz = [1.0 0.0im; 0.0 -1.0]
function H0_func(x)
    return 0.5*x[2]*(sx*cos(x[1])+sz*sin(x[1]))
end
function dH_func(x)
    return [0.5*x[2]*(-sx*sin(x[1])+sz*cos(x[1])), 0.5*(sx*cos(x[1])+sz*sin(x[1]))]
end
# dynamics
tspan = range(0.0, stop=1.0, length=1000)
# dissipation
decay_opt = [zeros(ComplexF64,size(rho0)[1],size(rho0)[1])] 
gamma = [0.0]
# prior distribution
function p_func(x, y, mu_x, mu_y, sigmax, sigmay, r)
    term1 = ((x-mu_x)/sigmax)^2-2*r*(((x-mu_x)/sigmax))*(((y-mu_y)/sigmay))+(((y-mu_y)/sigmay))^2
    term2 = exp(-term1/2.0/(1-r^2))
    term3 = 2*pi*sigmax*sigmay*sqrt(1-r^2)
    return term2/term3
end
function dp_func(x, y, mu_x, mu_y, sigmax, sigmay, r)
    term1 = -(2*((x-mu_x)/sigmax^2)-2*r*((y-mu_y)/sigmay)/sigmax)/2.0/(1-r^2)
    term2 = -(2*((y-mu_y)/sigmay^2)-2*r*((x-mu_x)/sigmax)/sigmay)/2.0/(1-r^2)
    p = p_func(x, y, mu_x, mu_y, sigmax, sigmay, r)
    return [term1*p, term2*p]
end
x = [range(-pi/2.0, stop=pi/2.0, length=100), range(pi/2-0.1, stop=pi/2+0.1, length=10)].|>Vector
sigmax, sigmay = 0.5, 1.0
mu_x, mu_y = 0.0, 0.0
r = 0.5
para_num = length(x)

p_tp = Matrix{Float64}(undef, length.(x)...)
dp_tp = Matrix{Vector{Float64}}(undef, length.(x)...)
rho = Matrix{Matrix{ComplexF64}}(undef, length.(x)...)
drho = Matrix{Vector{Matrix{ComplexF64}}}(undef, length.(x)...)
for i = 1:length(x[1])
    for j = 1:length(x[2])  
        x_tp = [x[1][i], x[2][j]]
        H0_tp = H0_func(x_tp)
        dH_tp = dH_func(x_tp)
        rho_tp, drho_tp  = QuanEstimation.expm(H0_tp, dH_tp, [zeros(ComplexF64,size(rho0)[1],size(rho0)[1])], [zeros(length(tspan)-1)], rho0, tspan, decay_opt, gamma)
        rho[i,j] = rho_tp[end]
        drho[i,j] = drho_tp[end]
        p_tp[i,j] = p_func(x[1][i], x[2][j], mu_x, mu_y, sigmax, sigmay, r)
        dp_tp[i,j] = dp_func(x[1][i], x[2][j], mu_x, mu_y, sigmax, sigmay, r)
    end
end
c = trapz(tuple(x...), p_tp)
p = p_tp/c
dp = dp_tp/c

f_BCRB1 = QuanEstimation.BCRB(x, p, rho, drho, M=nothing, btype=1)
f_BCRB2 = QuanEstimation.BCRB(x, p, rho, drho, M=nothing, btype=2)
f_VTB1 = QuanEstimation.VTB(x, p, dp, rho, drho, M=nothing, btype=1)
f_VTB2 = QuanEstimation.VTB(x, p, dp, rho, drho, M=nothing, btype=2)

f_BQCRB1 = QuanEstimation.BQCRB(x, p, rho, drho, btype=1)
f_BQCRB2 = QuanEstimation.BQCRB(x, p, rho, drho, btype=2)
f_QVTB1 = QuanEstimation.QVTB(x, p, dp, rho, drho, btype=1)
f_QVTB2 = QuanEstimation.QVTB(x, p, dp, rho, drho, btype=2)

for f in [f_BCRB1,f_BCRB2,f_VTB1,f_VTB2,f_BQCRB1,f_BQCRB2,f_QVTB1,f_QVTB2]
    println(f)
end