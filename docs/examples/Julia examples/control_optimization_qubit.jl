using QuanEstimation
using Random

# initial state
rho0 = 0.5*ones(2, 2)
# free Hamiltonian
omega0 = 1.0
sx = [0. 1.; 1. 0.0im]
sy = [0. -im; im 0.]
sz = [1. 0.0im; 0. -1.]
H0 = 0.5*omega0*sz
# derivative of the free Hamiltonian on omega0
dH = [0.5*sz]
# control Hamiltonians 
Hc = [sx, sy, sz]
# dissipation
sp = [0. 1.; 0. 0.0im]
sm = [0. 0.; 1. 0.0im]
decay = [[sp, 0.], [sm, 0.1]]
# measurement
M1 = 0.5*[1.0+0.0im  1.; 1.  1.]
M2 = 0.5*[1.0+0.0im -1.; -1.  1.]
M = [M1, M2]
# time length for the evolution
tspan = range(0., 10., length=2500)
# guessed control coefficients
cnum = length(tspan)-1
ctrl = [zeros(cnum) for _ in 1:length(Hc)]
ctrl_bound = [-2., 2.]
# set the optimization type
opt = QuanEstimation.Copt(ctrl=ctrl, ctrl_bound=ctrl_bound)

##----------------------choose the control algorithm------------------------##
# control algorithm: auto-GRAPE
alg = QuanEstimation.autoGRAPE(Adam=true, max_episode=300, epsilon=0.01, 
                               beta1=0.90, beta2=0.99)

# # control algorithm: GRAPE
# alg = QuanEstimation.GRAPE(Adam=true, max_episode=300, epsilon=0.01, 
#                            beta1=0.90, beta2=0.99)

# # control algorithm: PSO
# alg = QuanEstimation.PSO(p_num=10, ini_particle=([ctrl],), 
#                          max_episode=[1000, 100], c0=1.0, 
#                          c1=2.0, c2=2.0, seed=1234)

# # control algorithm: DE
# alg = QuanEstimation.DE(p_num=10, ini_population=([ctrl],), 
#                         max_episode=1000, c=1.0, cr=0.5, seed=1234)

# # control algorithm: DDPG
# alg = QuanEstimation.DDPG(max_episode=500, layer_num=4, layer_dim=220, 
#                           seed=1234)
##---------------------------------------------------------------------------##
# input the dynamics data
dynamics = QuanEstimation.Lindblad(opt, tspan, rho0, H0, dH, Hc, decay)  

##----------------------choose the objective function------------------------##
# objective function: QFI
obj = QuanEstimation.QFIM_obj()

# objective function: CFI
# obj = QuanEstimation.CFIM_obj(M=M)
##---------------------------------------------------------------------------##

# run the control optimization problem
QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
