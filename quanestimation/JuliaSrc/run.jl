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

@doc raw"""

    run(opt::AbstractOpt, alg::AbstractAlgorithm, obj::AbstractObj, dynamics::AbstractDynamics; savefile::Bool=false)

Run the optimization problem.
- `opt`: Types of optimization, options are `ControlOpt`, `StateOpt`, `MeasurementOpt`, `SMopt`, `SCopt`, `CMopt` and `SCMopt`.
- `alg`: Optimization algorithms, options are `auto-GRAPE`, `GRAPE`, `AD`, `PSO`, `DE`, 'NM' and `DDPG`.
- `obj`: Objective function, options are `QFIM_obj`, `CFIM_obj` and `HCRB_obj`.
- `dynamics`: Lindblad or Kraus parameterization process.
- `savefile`: Whether or not to save all the control coeffients. 

"""
function run(opt::AbstractOpt, alg::AbstractAlgorithm, obj::AbstractObj, dynamics::AbstractDynamics; savefile::Bool=false)
    output = Output(opt; save=savefile)
    obj = Objective(dynamics, obj)

    if obj isa HCRB_obj
        if alg isa AD || alg isa AD_Adam
            println("AD is not available when the objective function is HCRB.")
        elseif alg isa GRAPE || alg isa GRAPE_Adam || alg isa autoGRAPE || alg isa autoGRAPE_Adam
            println("GRAPE is not available when the objective function is HCRB.")
        elseif obj isa HCRB_obj{single_para}
            println("Program exit. In the single-parameter scenario, the HCRB is equivalent to the QFI. Please choose 'QFIM_obj()' as the objective function.")
        else
            system = QuanEstSystem(opt, alg, obj, dynamics, output)
            run(system)
        end
    else
        system = QuanEstSystem(opt, alg, obj, dynamics, output)
        run(system)
    end
end

@doc raw"""

    mintime(f::Number, opt::ControlOpt, alg::AbstractAlgorithm, obj::AbstractObj, dynamics::AbstractDynamics; savefile::Bool=false, method::String="binary")

Search of the minimum time to reach a given value of the objective function.
- `f`: The given value of the objective function.
- `opt`: Control Optimization.
- `alg`: Optimization algorithms, options are `auto-GRAPE`, `GRAPE`, `PSO`, `DE` and `DDPG`.
- `obj`: Objective function, options are `QFIM_obj`, `CFIM_obj` and `HCRB_obj`.
- `dynamics`: Lindblad dynamics.
- `savefile`: Whether or not to save all the control coeffients. 
- `method`: Methods for searching the minimum time to reach the given value of the objective function. Options are `binary` and `forward`.

- `system`: control system.
"""
function mintime(f::Number, opt::ControlOpt, alg::AbstractAlgorithm, obj::AbstractObj, dynamics::AbstractDynamics;savefile::Bool=false, method::String="binary")
    output = Output(opt; save=savefile)
    obj = Objective(dynamics, obj)
    system = QuanEstSystem(opt, alg, obj, dynamics, output)
    mintime(method, f, system)
end

# function run(opt::AbstractOpt, alg::AbstractAlgorithm, dynamics::AbstractDynamics; objective::Symbol, W=missing, M=missing, LD_type=:SLD, eps=GLOBAL_EPS, save::Bool=false)
#     output =  Output(opt;save=save)
#     obj = Objective(opt, objective, W, M, LD_type, eps)
#     system = QuanEstSystem(opt, alg, obj, dynamics, output)
#     run(system)
# end