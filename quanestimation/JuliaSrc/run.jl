abstract type AbstractSystem end

mutable struct QuanEstSystem{
    T<:AbstractOpt,
    A<:AbstractAlgorithm,
    F<:AbstractObj,
    D<:AbstractDynamics,
    O<:AbstractOutput,
} <: AbstractSystem
    optim::T
    algorithm::A
    obj::F
    dynamics::D
    output::O
end

function run(system::QuanEstSystem)
    (; optim, algorithm, obj, dynamics, output) = system
    show(system) # io1
    update!(optim, algorithm, obj, dynamics, output)
    show(obj, output) #io4
end

function run(opt::AbstractOpt, alg::AbstractAlgorithm, obj::AbstractObj,dynamics::AbstractDynamics;savefile::Bool=false)
    output =  Output(opt;save=savefile)
    obj = Objective(dynamics, obj)
    system = QuanEstSystem(opt, alg, obj, dynamics, output)
    run(system)
end

# function run(opt::AbstractOpt, alg::AbstractAlgorithm, dynamics::AbstractDynamics; objective::Symbol, W=missing, M=missing, LD_type=:SLD, eps=GLOBAL_EPS, save::Bool=false)
#     output =  Output(opt;save=save)
#     obj = Objective(opt, objective, W, M, LD_type, eps)
#     system = QuanEstSystem(opt, alg, obj, dynamics, output)
#     run(system)
# end