abstract type AbstractAlgorithm end

abstract type AbstractGRAPE <: AbstractAlgorithm end

struct GRAPE <: AbstractGRAPE
    max_episode::Int 
    epsilon::Number
end

struct GRAPE_Adam <: AbstractGRAPE
    max_episode::Int
    epsilon::Number
    beta1::Number
    beta2::Number
end


GRAPE(max_episode, epsilon, beta1, beta2) = GRAPE_Adam(max_episode, epsilon, beta1, beta2)

"""

    GRAPE(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true)

Control optimization algorithm: GRAPE.
- `max_episode`: The number of episodes.
- `epsilon`: Learning rate.
- `beta1`: The exponential decay rate for the first moment estimates.
- `beta2`: The exponential decay rate for the second moment estimates.
- `Adam`: Whether or not to use Adam for updating control coefficients.   
"""
GRAPE(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true) = Adam ? GRAPE_Adam(max_episode, epsilon, beta1, beta2) : GRAPE(max_episode, epsilon)

abstract type AbstractautoGRAPE <: AbstractAlgorithm end
struct autoGRAPE <: AbstractautoGRAPE
    max_episode::Int
    epsilon::Number
end

struct autoGRAPE_Adam <: AbstractautoGRAPE
    max_episode::Int
    epsilon::Number
    beta1::Number
    beta2::Number
end

autoGRAPE(max_episode, epsilon, beta1, beta2) = autoGRAPE_Adam(max_episode, epsilon, beta1, beta2)

"""

    autoGRAPE(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true)

Control optimization algorithm: auto-GRAPE.
- `max_episode`: The number of episodes.
- `epsilon`: Learning rate.
- `beta1`: The exponential decay rate for the first moment estimates.
- `beta2`: The exponential decay rate for the second moment estimates.
- `Adam`: Whether or not to use Adam for updating control coefficients.   
"""
autoGRAPE(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true) = Adam ? autoGRAPE_Adam(max_episode, epsilon, beta1, beta2) : autoGRAPE(max_episode, epsilon)

abstract type AbstractAD <:  AbstractAlgorithm end
struct AD <: AbstractAD
    max_episode::Number
    epsilon::Number
end

struct AD_Adam <: AbstractAD
    max_episode::Number
    epsilon::Number
    beta1::Number
    beta2::Number
end

AD(max_episode, epsilon, beta1, beta2) = AD_Adam(max_episode=max_episode, epsilon=epsilon, beta1=beta1, beta2=beta2)
"""

    AD(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true)

Optimization algorithm: AD.
- `max_episode`: The number of episodes.
- `epsilon`: Learning rate.
- `beta1`: The exponential decay rate for the first moment estimates.
- `beta2`: The exponential decay rate for the second moment estimates.
- `Adam`: Whether or not to use Adam for updating control coefficients.
"""
AD(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true) = Adam ? AD_Adam(max_episode, epsilon, beta1, beta2) : AD(max_episode, epsilon)

mutable struct PSO <: AbstractAlgorithm
    max_episode::Union{T,Vector{T}} where {T<:Int} 
    p_num::Int
    ini_particle::Union{Tuple, Missing}
    c0::Number 
    c1::Number
    c2::Number
end

"""

    PSO(;max_episode::Union{T,Vector{T}} where {T<:Int}=[1000, 100], p_num::Number=10, ini_particle=missing, c0::Number=1.0, c1::Number=2.0, c2::Number=2.0, seed::Number=1234)

Optimization algorithm: PSO.
- `max_episode`: The number of episodes, it accepts both integer and array with two elements.
- `p_num`: The number of particles. 
- `ini_particle`: Initial guesses of the optimization variables.
- `c0`: The damping factor that assists convergence, also known as inertia weight.
- `c1`: The exploitation weight that attracts the particle to its best previous position, also known as cognitive learning factor.
- `c2`: The exploitation weight that attracts the particle to the best position in the neighborhood, also known as social learning factor. 
"""
PSO(;max_episode::Union{T,Vector{T}} where {T<:Int}=[1000, 100], p_num::Number=10, ini_particle=missing, c0::Number=1.0, c1::Number=2.0, c2::Number=2.0) =
    PSO(max_episode, p_num, ini_particle, c0, c1, c2)

mutable struct DE <: AbstractAlgorithm
    max_episode::Number
    p_num::Number
    ini_population::Union{Tuple, Missing}
    c::Number
    cr::Number
end

"""

    DE(;max_episode::Number=1000, p_num::Number=10, ini_population=missing, c::Number=1.0, cr::Number=0.5, seed::Number=1234)

Optimization algorithm: DE.
- `max_episode`: The number of populations.
- `p_num`: The number of particles. 
- `ini_population`: Initial guesses of the optimization variables.
- `c`: Mutation constant.
- `cr`: Crossover constant.
"""
DE(;max_episode::Number=1000, p_num::Number=10, ini_population=missing, c::Number=1.0,cr::Number=0.5) = DE(max_episode, p_num, ini_population, c, cr)

struct DDPG <: AbstractAlgorithm
    max_episode::Int
    layer_num::Int
    layer_dim::Int
    rng::AbstractRNG
end

DDPG(max_episode, layer_num, layer_dim) =
    DDPG(max_episode, layer_num, layer_dim, StableRNG(1234))
DDPG(max_episode, layer_num, layer_dim, seed::Number) =
    DDPG(max_episode, layer_num, layer_dim, StableRNG(seed))
"""

    DDPG(;max_episode::Int=500, layer_num::Int=3, layer_dim::Int=200, seed::Number=1234)

Optimization algorithm: DE.
- `max_episode`: The number of populations.
- `layer_num`: The number of layers (include the input and output layer).
- `layer_dim`: The number of neurons in the hidden layer.
- `seed`: Random seed.
"""
DDPG(;max_episode::Int=500, layer_num::Int=3, layer_dim::Int=200, seed::Number=1234) =
    DDPG(max_episode, layer_num, layer_dim, StableRNG(seed))

struct NM <: AbstractAlgorithm
    max_episode::Int
    p_num::Int
    ini_state::Union{AbstractVector, Missing}
    ar::Number
    ae::Number
    ac::Number
    as0::Number
end

"""

    NM(;max_episode::Int=1000, p_num::Int=10, nelder_mead=missing, ar::Number=1.0, ae::Number=2.0, ac::Number=0.5, as0::Number=0.5, seed::Number=1234)

State optimization algorithm: NM.
- `max_episode`: The number of populations.
- `p_num`: The number of the input states.
- `nelder_mead`: Initial guesses of the optimization variables.
- `ar`: Reflection constant.
- `ae`: Expansion constant.
- `ac`: Constraction constant.
- `as0`: Shrink constant.
"""
NM(;max_episode::Int=1000, p_num::Int=10, nelder_mead=missing, ar::Number=1.0, ae::Number=2.0, ac::Number=0.5, as0::Number=0.5) = NM(max_episode, p_num, nelder_mead, ar, ae, ac, as0)

struct Iterative <: AbstractAlgorithm
    max_episode::Int
end

"""

    Iterative(;max_episode::Int=300, seed::Number=1234)

State optimization algorithm: Iterative.
- `max_episode`: The number of episodes.
"""
Iterative(;max_episode::Int=300) = Iterative(max_episode)

alg_type(::AD) = :AD
alg_type(::AD_Adam) = :AD
alg_type(::GRAPE) = :GRAPE
alg_type(::GRAPE_Adam) = :GRAPE
alg_type(::autoGRAPE) = :autoGRAPE
alg_type(::autoGRAPE_Adam) = :autoGRAPE
alg_type(::PSO) = :PSO
alg_type(::DDPG) = :DDPG
alg_type(::DE) = :DE
alg_type(::NM) = :NM
alg_type(::Iterative) = :Iterative

include("AD.jl")
include("DDPG.jl")
include("DE.jl")
include("GRAPE.jl")
include("NM.jl")
include("Iterative.jl")
include("PSO.jl")
