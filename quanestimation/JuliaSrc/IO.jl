using Printf

const IO_obj = Dict(:QFIM => "quantum ", :CFIM => "classical ", :HCRB => "")

const IO_opt = Dict(
    :Copt => "control optimization",
    :Sopt => "state optimization",
    :Mopt => "measurement optimization",
    :CompOpt => "comprehensive optimization",
)

const IO_para = Dict(
    :single_para => "single parameter scenario",
    :multi_para => "multiparameter scenario",
)

const IO_alg = Dict(
    :GRAPE => "optimization algorithm: GRAPE",
    :AD => "optimization algorithm: auto-GRAPE",
    :PSO => "optimization algorithm: Particle Swarm Optimization (PSO)",
    :DE => "optimization algorithm: Differential Evolution (DE)",
    :DDPG => "optimization algorithm: deep deterministic policy gradient algorithm (DDPG)",
    :NM => "optimization algorithm: Nelder-Mead method (NM)",
)

const IO_ini = Dict(
    (
        :Copt,
        :QFIM,
        :single_para,
    ) => "non-controlled value of QFI is %f\n initial value of QFI is %f\n",
    (
        :Copt,
        :QFIM,
        :multi_para,
    ) => "non-controlled value of tr(WF^{-1}) is %f\n initial value of tr(WF^{-1}) is %f\n",
    (
        :Copt,
        :CFIM,
        :single_para,
    ) => "non-controlled value of CFI is %f\n initial value of CFI is %f\n",
    (
        :Copt,
        :CFIM,
        :multi_para,
    ) => "non-controlled value of tr(WI^{-1}) is %f\n initial value of tr(WI^{-1}) is %f\n",
    (
        :Copt,
        :HCRB,
        :multi_para,
    ) => "non-controlled value of HCRB is %f\n initial value of HCRB is %f\n",
)

const IO_current = Dict(
    (:QFIM, :single_para) => "current value of QFI is %f (%i episodes)\r",
    (:QFIM, :multi_para) => "current value of tr(WF^{-1}) is %f (%i episodes)\r",
    (:CFIM, :single_para) => "current value of CFI is %f (%i episodes)\r",
    (:CFIM, :multi_para) => "current value of of tr(WI^{-1}) is %f (%i episodes)\r",
    (:HCRB, :multi_para) => "current value of of HCRB is %f (%i episodes)\r",
)

const IO_final = Dict(
    (:QFIM, :single_para) => "\e[2KIteration over, data saved.\nFinal value of QFI is %f\n",
    (
        :QFIM,
        :multi_para,
    ) => "\e[2KIteration over, data saved.\nFinal value of tr(WF^{-1}) is %f\n",
    (:CFIM, :single_para) => "\e[2KIteration over, data saved.\nFinal value of CFI is %f\n",
    (
        :CFIM,
        :multi_para,
    ) => "\e[2KIteration over, data saved.\nFinal value of tr(WI^{-1}) is %f\n",
    (:HCRB, :multi_para) => "\e[2KIteration over, data saved.\nFinal value of HCRB is %f\n",
)

## io info
function show(system::QuanEstSystem)
    (; optim, algorithm, objective) = system
    println(
        (optim isa ControlOpt ? IO_obj[obj_type(objective)] : "") *
        IO_opt[opt_target(optim)],
    )
    println(IO_para[para_type(objective)])
    println(IO_alg[alg_type(algorithm)])
end

## io initialization
function show(opt::AbstractOpt, output::Output, obj::AbstractObj)
    (; io_buffer) = output
    @eval @printf $(IO_ini[opt_target(opt), obj_type(obj), para_type(obj)]) $(io_buffer...)
    SaveCurrent(output)
end

## io current
function show(output::Output, obj::AbstractObj)
    (; io_buffer) = output
    @eval @printf $(IO_current[obj_type(obj), para_type(obj)]) $(io_buffer...)
    SaveCurrent(output)
end

## io final
function show(obj::AbstractObj, output::AbstractOutput)
    (; io_buffer) = output
    @eval @printf $(IO_final[obj_type(obj), para_type(obj)]) $(io_buffer...)
    SaveFile(output)
end
