using Plots: append!
using LinearAlgebra: similar
# using LinearAlgebra: similar
# using LinearAlgebra: Matrix
# using Core: Vector
using StableRNGs
using ReinforcementLearning
using LinearAlgebra

include("Control/DDPG.jl")
include("QuanEstimation.jl")

using .QuanEstimation

# sigmaz() = QuanEstimation.sigmaz()
# sigmam() = QuanEstimation.sigmam()
# sigmax() = QuanEstimation.sigmax()
# sigmay() = QuanEstimation.sigmay()

ρ_initial = [0.5 + 0.0im 0.5
    0.5 0.5]

M0 = 0.5 * [1.0 + 0.0im -1.
-1. 1.]
M1 = 0.5 * [1.0 1.
1. 1.]
M = [M0, M1]

tnum = 4000
cnum = 6
times = range(0, 20.0, length=tnum)

ϵ = 0.1
w = 1.
# θ = pi/4
H0 = 0.5 * w  *  sigmaz()
dH = [0.5 * sigmaz()]
Liouville_operator = [sigmam()]  #[(sigmax()+sigmaz())/sqrt(2)] #[zero(sigmam())]
γ = [0.1]

control_Hamiltonian= [sigmax(),sigmay(),sigmaz()]

Hc_coeff =[[0.0 for i=1:length(times)] for j=1:length(control_Hamiltonian)]
dim = size(ρ_initial, 1)
C =2.0

params = ControlEnvParams(H0, dH, ρ_initial, times|>Array, Liouville_operator, γ, control_Hamiltonian, Hc_coeff, tnum÷cnum ,1.0,  I(1), dim, C)   

rng = StableRNG(114514)
QFI_ori = QuanEstimation.QFI_ori
QFIM_ori = QuanEstimation.QFIM_ori
# propagate = QuanEstimation.propagate
# vec2mat = QuanEstimation.vec2mat
QFIM_saveall = QuanEstimation.QFIM_saveall
# bound! = QuanEstimation.bound!
# env = ControlEnv(params = params, rng = rng)

# env([1.0,1.,1.])

# RLBase.test_runnable!(env)

# hook = TotalRewardPerEpisode()

# run(RandomPolicy(action_space(env)), env, StopAfterEpisode(1_000), hook)

# using Plots

# plot(hook.rewards)

DDPG(params)


# @time QuanEstimation.QFIM(params)
# @time QFIM_ori(params)
# @time QuanEstimation.QFI_ori(params)| 