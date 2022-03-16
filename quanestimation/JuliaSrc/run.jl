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
    objective::F
    dynamics::D
    output::O
end

function run(system::QuanEstSystem)
    (; optim, algorithm, objective, dynamics, output) = system
    show(system) # io1
    update!(optim, algorithm, objective, dynamics, output)
    show(objective, output) #io4
end
