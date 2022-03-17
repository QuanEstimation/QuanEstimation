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
