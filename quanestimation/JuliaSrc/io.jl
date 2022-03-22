using Printf

const IO_obj = Dict(:QFIM => "quantum ", :CFIM => "classical ", :HCRB => "")

const IO_opt = Dict(
    :Copt => "control optimization",
    :Sopt => "state optimization",
    :Mopt => "measurement optimization",
    :Mopt_input => "measurement optimization",
    :SCopt => "comprehensive optimization",
    :SMopt => "comprehensive optimization",
    :CMopt => "comprehensive optimization",
    :SCMopt => "comprehensive optimization",
)

const IO_para = Dict(
    :single_para => "single parameter scenario",
    :multi_para => "multiparameter scenario",
)

const IO_alg = Dict(
    :GRAPE => "optimization algorithm: GRAPE",
    :autoGRAPE => "optimization algorithm: auto-GRAPE",
    :AD => "optimization algorithm: Automatic Differentiation (AD)",
    :PSO => "optimization algorithm: Particle Swarm Optimization (PSO)",
    :DE => "optimization algorithm: Differential Evolution (DE)",
    :DDPG => "optimization algorithm: Deep Deterministic Policy Gradient (DDPG)",
    :NM => "optimization algorithm: Nelder-Mead (NM)",
)

const IO_ini = Dict(
    #### control optimization ####
    (
        :Copt,
        :QFIM,
        :single_para,
    ) => "non-controlled value of QFI is %f\ninitial value of QFI is %f\n",
    (
        :Copt,
        :QFIM,
        :multi_para,
    ) => "non-controlled value of tr(WF^{-1}) is %f\ninitial value of tr(WF^{-1}) is %f\n",
    (
        :Copt,
        :CFIM,
        :single_para,
    ) => "non-controlled value of CFI is %f\ninitial value of CFI is %f\n",
    (
        :Copt,
        :CFIM,
        :multi_para,
    ) => "non-controlled value of tr(WI^{-1}) is %f\ninitial value of tr(WI^{-1}) is %f\n",
    (
        :Copt,
        :HCRB,
        :multi_para,
    ) => "non-controlled value of HCRB is %f\ninitial value of HCRB is %f\n",
    #### state optimization ####
    (
        :Sopt,
        :QFIM,
        :single_para,
    ) => "initial value of QFI is %f\n",
    (
        :Sopt,
        :QFIM,
        :multi_para,
    ) => "initial value of tr(WF^{-1}) is %f\n",
    (
        :Sopt,
        :CFIM,
        :single_para,
    ) => "initial value of CFI is %f\n",
    (
        :Sopt,
        :CFIM,
        :multi_para,
    ) => "initial value of tr(WI^{-1}) is %f\n",
    (
        :Sopt,
        :HCRB,
        :multi_para,
    ) => "initial value of HCRB is %f\n",

    #### projective measurement optimization ####
    (
        :Mopt,
        :CFIM,
        :single_para,
    ) => "initial value of CFI is %f\nQFI is %f\n",
    (
        :Mopt,
        :CFIM,
        :multi_para,
    ) => "initial value of tr(WI^{-1}) is %f\ninitial value of tr(WF^{-1}) is %f\n",

    #### measurement optimization for LinearComb and Rotation ####
    (
        :Mopt_input,
        :CFIM,
        :single_para,
    ) => "initial value of CFI is %f\nCFI under the given POVM is %f\nQFI is %f\n",
    (
        :Mopt_input,
        :CFIM,
        :multi_para,
    ) => "initial value of tr(WI^{-1}) is %f\ntr(WI^{-1}) under the given POVM is %f\ntr(WF^{-1}) is %f\n",

    #### state and control optimization ####
    (
        :SCopt,
        :QFIM,
        :single_para,
    ) => "non-controlled value of QFI is %f\ninitial value of QFI is %f\n",
    (
        :SCopt,
        :QFIM,
        :multi_para,
    ) => "non-controlled value of tr(WF^{-1}) is %f\ninitial value of tr(WF^{-1}) is %f\n",
    (
        :SCopt,
        :CFIM,
        :single_para,
    ) => "non-controlled value of CFI is %f\ninitial value of CFI is %f\n",
    (
        :SCopt,
        :CFIM,
        :multi_para,
    ) => "non-controlled value of tr(WI^{-1}) is %f\ninitial value of tr(WI^{-1}) is %f\n",
    (
        :SCopt,
        :HCRB,
        :multi_para,
    ) => "non-controlled value of HCRB is %f\ninitial value of HCRB is %f\n",
    #### state and measurement optimization ####
    (
        :SMopt,
        :CFIM,
        :single_para,
    ) => "initial value of CFI is %f\n",
    (
        :SMopt,
        :CFIM,
        :multi_para,
    ) => "initial value of tr(WI^{-1}) is %f\n",
    #### control and measurement optimization ####
    (
        :CMopt,
        :CFIM,
        :single_para,
    ) => "initial value of CFI is %f\n",
    (
        :CMopt,
        :CFIM,
        :multi_para,
    ) => "initial value of tr(WI^{-1}) is %f\n",
    #### state, control and measurement optimization ####
    (
        :SCMopt,
        :CFIM,
        :single_para,
    ) => "initial value of CFI is %f\n",
    (
        :SCMopt,
        :CFIM,
        :multi_para,
    ) => "initial value of tr(WI^{-1}) is %f\n",

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
    (; optim, algorithm, obj) = system
    println(
        (optim isa ControlOpt ? IO_obj[obj_type(obj)] : "") *
        IO_opt[opt_target(optim)],
    )
    println(IO_para[para_type(obj)])
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
