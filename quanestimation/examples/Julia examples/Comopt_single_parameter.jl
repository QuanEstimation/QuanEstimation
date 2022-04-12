using Random
using StableRNGs
include("../src/QuanEstimation.jl")

# initial state
ρ₀ = 0.5 * ones(2,2)
# Hamiltonian
ω₀ = 1.0
sx = [0.0 1.0; 1.0 0.0im]
sy = [0.0 -1.0im; 1.0im 0.0]
sz = [1.0 0.0im; 0.0 -1.0]
H0 = 0.5 * ω₀ * sz
dH = [0.5 * sz]
Hc = [sx, sy, sz]
# measurement
M1 = 0.5*[1.0+0.0im  1.0; 1.0  1.0]
M2 = 0.5*[1.0+0.0im -1.0; -1.0  1.0]
M = [M1, M2]
# dissipation
sp = [0 1; 0 0.0im]
sm = [0 0; 1 0.0im]
decay = [[sp, 0.0], [sm, 0.1]]
# dynamics
tspan = range(0.0, 10.0, length=2500)
# initial control coefficients
cnum = length(tspan) - 1
ctrl0 = [zeros(cnum) for _ in 1:length(Hc)]

# state control optimization 
opt = QuanEstimation.SCopt()
dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay)   

# control measurement optimization
opt = QuanEstimation.CMopt()
dynamics = QuanEstimation.Lindblad(opt, tspan, ρ₀, H0, dH, Hc, decay)

# state measurement optimization
opt = QuanEstimation.SMopt()
dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, ctrl0, decay)

# state control measurement optimization 
opt = QuanEstimation.SCMopt()
dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay) 
    
# control algorithm: AD
if opt isa QuanEstimation.SCopt
    alg = QuanEstimation.AD(Adam=true, max_episode=50, ϵ=0.01, beta1=0.90, beta2=0.99)
    obj = QuanEstimation.QFIM_Obj()
    obj = QuanEstimation.CFIM_Obj(M=M)
    # obj = QuanEstimation.HCRB_Obj()

    QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)
end

# control algorithm: PSO
alg = QuanEstimation.PSO(max_episode=[100, 10], p_num=10, c0=1.0, c1=2.0, c2=2.0, rng= MersenneTwister(1234))
obj = QuanEstimation.QFIM_Obj()
obj = QuanEstimation.CFIM_Obj(M=M)
# obj = QuanEstimation.HCRB_Obj()

QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

# control algorithm: DE
alg = QuanEstimation.DE(max_episode=100, p_num=10, c=1.0, cr=0.5, rng=MersenneTwister(1234))
obj = QuanEstimation.QFIM_Obj()
obj = QuanEstimation.CFIM_Obj(M=M)
# obj = QuanEstimation.HCRB_Obj()

QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)