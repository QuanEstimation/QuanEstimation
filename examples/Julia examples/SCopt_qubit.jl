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
# control Hamiltonians 
Hc = [sx, sy, sz]
# dissipation
sp = [0. 1.; 0. 0.0im]
sm = [0. 0.; 1. 0.0im]
decay = [[sp, 0.0], [sm, 0.1]]
# measurement
M1 = 0.5*[1.0+0.0im  1.; 1.  1.]
M2 = 0.5*[1.0+0.0im -1.; -1.  1.]
M = [M1, M2]
# time length for the evolution
tspan = range(0., 10., length=2500)
# choose the optimization type
opt = QuanEstimation.SCopt(ctrl_bound=[-2.0,2.0], seed=1234)

##==========choose comprehensive optimization algorithm==========##
##-------------algorithm: DE---------------------##
alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
# input the dynamics data
dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, dyn_method=:Expm) 

##-------------algorithm: PSO---------------------##
# alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
#                          c1=2.0, c2=2.0)
# # input the dynamics data
# dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, dyn_method=:Expm) 

##-------------algorithm: AD---------------------##
# alg = QuanEstimation.AD(Adam=true, max_episode=300, epsilon=0.01, 
#                         beta1=0.90, beta2=0.99)
# # input the dynamics data
# dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, dyn_method=:Expm) 

##===================choose objective function===================##
##-------------objective function: QFI---------------------##
obj = QuanEstimation.QFIM_obj()
# run the comprehensive optimization problem
QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)

##-------------objective function: CFI---------------------##
# obj = QuanEstimation.CFIM_obj(M=M)
# # run the comprehensive optimization problem
# QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
