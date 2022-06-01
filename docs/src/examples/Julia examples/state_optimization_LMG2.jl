using QuanEstimation
using Random
using StableRNGs
using LinearAlgebra
using SparseArrays

# dimensions of the system
N = 8
# generation of the coherent spin state
j, theta, phi = N÷2, 0.5pi, 0.5pi
Jp = Matrix(spdiagm(1=>[sqrt(j*(j+1)-m*(m+1)) for m in j:-1:-j][2:end]))
Jm = Jp'
psi0 = exp(0.5*theta*exp(im*phi)*Jm - 0.5*theta*exp(-im*phi)*Jp)*
           QuanEstimation.basis(Int(2*j+1), 1)
dim = length(psi0)
# free Hamiltonian
lambda, g, h = 1.0, 0.5, 0.1
Jx = 0.5*(Jp + Jm)
Jy = -0.5im*(Jp - Jm)
Jz = spdiagm(j:-1:-j)
H0 = -lambda*(Jx*Jx + g*Jy*Jy) / N + g * Jy^2 / N - h*Jz
# derivative of the free Hamiltonian on g
dH = [-lambda*Jy*Jy/N, -Jz]
# dissipation
decay = [[Jz, 0.1]]
# time length for the evolution
tspan = range(0., 10., length=2500)
# weight matrix
W = [1/3 0.; 0. 2/3]
# set the optimization type
opt = QuanEstimation.StateOpt(psi=psi0, seed=1234)

##====================choose the state optimization algorithm====================##
# state optimization algorithm: AD
alg = QuanEstimation.AD(Adam=false, max_episode=300, epsilon=0.01, 
                        beta1=0.90, beta2=0.99)

# # state optimization algorithm: PSO
# alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
#                          c1=2.0, c2=2.0)

# # state optimization algorithm: DE
# alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)

# # state optimization algorithm: NM
# alg = QuanEstimation.NM(p_num=10, max_episode=1000, ar=1.0, 
#                         ae=2.0, ac=0.5, as0=0.5)

# # state optimization algorithm: DDPG
# alg = QuanEstimation.DDPG(max_episode=500, layer_num=3, layer_dim=200)

##====================choose the objective function====================##
##-------------objective function: QFI---------------------##
# objective function: tr(WF^{-1})
obj = QuanEstimation.QFIM_obj(W=W)
# input the dynamics data
dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, decay=decay) 
# run the state optimization problem
QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)

##-------------objective function: CFI---------------------##
# # objective function: tr(WI^{-1})
# obj = QuanEstimation.CFIM_obj(W=W)
# # input the dynamics data
# dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, decay=decay) 
# # run the state optimization problem
# QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)

##-------------objective function: HCRB---------------------##
# # objective function: HCRB
# obj = QuanEstimation.HCRB_obj(W=W)
# # input the dynamics data
# dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, decay=decay) 
# # run the state optimization problem
# QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
