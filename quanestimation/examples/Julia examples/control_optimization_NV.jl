using QuanEstimation
using Random
using LinearAlgebra

# initial state
rho0 = zeros(ComplexF64, 6, 6)
rho0[1:4:5, 1:4:5] .= 0.5
dim = size(rho0, 1)
# Hamiltonian
sx = [0. 1.; 1. 0.]
sy = [0. -im; im 0.]
sz = [1. 0.; 0. -1.]
s1 = [0. 1. 0.; 1. 0. 1.; 0. 1. 0.]/sqrt(2)
s2 = [0. -im 0.; im 0. -im; 0. im 0.]/sqrt(2)
s3 = [1. 0. 0.; 0. 0. 0.; 0. 0. -1.]
Is = I1, I2, I3 = [kron(I(3), sx), kron(I(3), sy), kron(I(3), sz)]
S = S1, S2, S3 = [kron(s1, I(2)), kron(s2, I(2)), kron(s3, I(2))]
B = B1, B2, B3 = [5.0e-4, 5.0e-4, 5.0e-4]
# All numbers are divided by 100 in this example 
# for better calculation accurancy
cons = 100
D = (2pi*2.87*1000)/cons
gS = (2pi*28.03*1000)/cons
gI = (2pi*4.32)/cons
A1 = (2pi*3.65)/cons
A2 = (2pi*3.03)/cons
H0 = sum([D*kron(s3^2, I(2)), sum(gS*B.*S), sum(gI*B.*Is),
          A1*(kron(s1, sx) + kron(s2, sy)), A2*kron(s3, sz)])
# derivatives of the free Hamiltonian on B1, B2 and B3
dH = gS*S+gI*Is
# control Hamiltonians 
Hc = [S1, S2, S3]
# dissipation
decay = [[S3, 2*pi/cons]]
# measurement
M = [QuanEstimation.basis(dim, i)*QuanEstimation.basis(dim, i)' \
     for i in 1:dim]
# time length for the evolution 
tspan = range(0., 2., length=4000)
# guessed control coefficients
cnum = 10
rng = MersenneTwister(1234)
cnum = length(Hc)
ini_1 = [zeros(cnum) for _ in 1:cnum]
ini_2 = 0.2.*[ones(cnum) for _ in 1:cnum]
ini_3 = -0.2.*[ones(cnum) for _ in 1:cnum]
ini_4 = [[range(-0.2, 0.2, length=cnum)...] for _ in 1:cnum]
ini_5 = [[range(-0.2, 0., length=cnum)...] for _ in 1:cnum]
ini_6 = [[range(0., 0.2, length=cnum)...] for _ in 1:cnum]
ini_7 = [-0.2*ones(cnum)+0.01*rand(rng,cnum) for _ in 1:cnum]
ini_8 = [-0.2*ones(cnum)+0.01*rand(rng,cnum) for _ in 1:cnum]
ini_9 = [-0.2*ones(cnum)+0.05*rand(rng,cnum) for _ in 1:cnum]
ini_10 = [-0.2*ones(cnum)+0.05*rand(rng,cnum) for _ in 1:cnum]
ctrl0 = [Symbol("ini_", i)|>eval for i in 1:10]
# set the optimization type
opt = QuanEstimation.Copt(ctrl=ini_1, ctrl_bound=[-0.2, 0.2])

##----------------------choose the control algorithm------------------------##
# control algorithm: auto-GRAPE
alg = QuanEstimation.autoGRAPE(Adam=true, max_episode=300, epsilon=0.01, 
                               beta1=0.90, beta2=0.99)

# # control algorithm: PSO
# alg = QuanEstimation.PSO(p_num=10, ini_particle=(ctrl0,), 
#                          max_episode=[1000, 100], c0=1.0, 
#                          c1=2.0, c2=2.0, seed=1234)

# # control algorithm: DE
# alg = QuanEstimation.DE(p_num=10, ini_population=(ctrl0,), 
#                         max_episode=1000, c=1.0, cr=0.5, seed=1234)

# # control algorithm: DDPG
# alg = QuanEstimation.DDPG(max_episode=500, layer_num=4, layer_dim=220, 
#                           seed=1234)
##---------------------------------------------------------------------------##

# input the dynamics data
dynamics = QuanEstimation.Lindblad(opt, tspan, rho0, H0, dH, Hc, decay)  

##----------------------choose the objective function------------------------##
# objective function: tr(WF^{-1})
obj = QuanEstimation.QFIM_obj()

# # objective function: tr(WI^{-1})
# obj = QuanEstimation.CFIM_obj(M=M)

# # objective function: HCRB
# obj = QuanEstimation.HCRB_obj()
##---------------------------------------------------------------------------##

# run the control optimization problem
QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
