abstract type AbstractAlgorithm end

abstract type AbstractGRAPE <: AbstractAlgorithm end

Base.@kwdef struct GRAPE <: AbstractGRAPE
    max_episode::Int = 300 
    epsilon::Number = 0.01
end

Base.@kwdef struct GRAPE_Adam <: AbstractGRAPE
    max_episode::Int = 300
    epsilon::Number = 0.01
    beta1::Number = 0.90
    beta2::Number = 0.99
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
Base.@kwdef struct autoGRAPE <: AbstractautoGRAPE
    max_episode::Int = 300
    epsilon::Number = 0.01
end

Base.@kwdef struct autoGRAPE_Adam <: AbstractautoGRAPE
    max_episode::Int = 300
    epsilon::Number = 0.01
    beta1::Number = 0.90
    beta2::Number = 0.99
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
Base.@kwdef struct AD <: AbstractAD
    max_episode::Number = 300
    epsilon::Number = 0.01
end

Base.@kwdef struct AD_Adam <: AbstractAD
    max_episode::Number = 300
    epsilon::Number = 0.01
    beta1::Number = 0.90
    beta2::Number = 0.99
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

Base.@kwdef struct PSO <: AbstractAlgorithm
    max_episode::Union{T,Vector{T}} where {T<:Int} = [1000, 100]
    p_num::Int = 10
    ini_particle::Union{Tuple, Missing} = missing
    c0::Number = 1.0
    c1::Number = 2.0
    c2::Number = 2.0
    rng::AbstractRNG = GLOBAL_RNG
end

PSO(max_episode, p_num, ini_particle, c0, c1, c2) =
    PSO(max_episode, p_num, ini_particle, c0, c1, c2, GLOBAL_RNG)
PSO(max_episode, p_num, ini_particle, c0, c1, c2, seed::Number) =
    PSO(max_episode, p_num, ini_particle, c0, c1, c2, MersenneTwister(seed))
"""

    PSO(;max_episode::Union{T,Vector{T}} where {T<:Int}=[1000, 100], p_num::Number=10, ini_particle=missing, c0::Number=1.0, c1::Number=2.0, c2::Number=2.0, seed::Number=1234)

Optimization algorithm: PSO.
- `max_episode`: The number of episodes, it accepts both integer and array with two elements.
- `p_num`: The number of particles. 
- `ini_particle`: Initial guesses of the optimization variables.
- `c0`: The damping factor that assists convergence, also known as inertia weight.
- `c1`: The exploitation weight that attracts the particle to its best previous position, also known as cognitive learning factor.
- `c2`: The exploitation weight that attracts the particle to the best position in the neighborhood, also known as social learning factor. 
- `seed`: Random seed.
"""
PSO(;max_episode::Union{T,Vector{T}} where {T<:Int}=[1000, 100], p_num::Number=10, ini_particle=missing, c0::Number=1.0, c1::Number=2.0, c2::Number=2.0, seed::Number=1234) =
    PSO(max_episode, p_num, ini_particle, c0, c1, c2, MersenneTwister(seed))

Base.@kwdef struct DE <: AbstractAlgorithm
    max_episode::Number = 1000
    p_num::Number = 10
    ini_population::Union{Tuple, Missing} = missing
    c::Number = 1.0
    cr::Number = 0.5
    rng::AbstractRNG = GLOBAL_RNG
end

DE(max_episode, p_num, ini_population, c, cr) =
    DE(max_episode, p_num, ini_population, c, cr, GLOBAL_RNG)
DE(max_episode, p_num, ini_population, c, cr, seed::Number) =
    DE(max_episode, p_num, ini_population, c, cr, MersenneTwister(seed))
"""

    DE(;max_episode::Number=1000, p_num::Number=10, ini_population=missing, c::Number=1.0, cr::Number=0.5, seed::Number=1234)

Optimization algorithm: DE.
- `max_episode`: The number of populations.
- `p_num`: The number of particles. 
- `ini_population`: Initial guesses of the optimization variables.
- `c`: Mutation constant.
- `cr`: Crossover constant.
- `seed`: Random seed.
"""
DE(;max_episode::Number=1000, p_num::Number=10, ini_population=missing, c::Number=1.0, cr::Number=0.5, seed::Number=1234) =
    DE(max_episode, p_num, ini_population, c, cr, MersenneTwister(seed))

Base.@kwdef struct DDPG <: AbstractAlgorithm
    max_episode::Int = 500
    layer_num::Int = 3
    layer_dim::Int = 200
    rng::AbstractRNG = GLOBAL_RNG
end

DDPG(max_episode, layer_num, layer_dim) =
    DDPG(max_episode, layer_num, layer_dim, GLOBAL_RNG)
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

Base.@kwdef struct NM <: AbstractAlgorithm
    max_episode::Int = 1000 
    state_num::Int = 10
    ini_state::Union{AbstractVector, Missing} = missing
    ar::Number = 1.0
    ae::Number = 2.0
    ac::Number = 0.5
    as0::Number = 0.5
    rng::AbstractRNG = GLOBAL_RNG
end

NM(max_episode, state_num, nelder_mead, ar, ae, ac, as0) =
    NM(max_episode, state_num, nelder_mead, ar, ae, ac, as0, GLOBAL_RNG)
NM(max_episode, state_num, nelder_mead, ar, ae, ac, as0, seed::Number) =
    NM(max_episode, state_num, nelder_mead, ar, ae, ac, as0, MersenneTwister(seed))
"""

    NM(;max_episode::Int=1000, state_num::Int=10, nelder_mead=missing, ar::Number=1.0, ae::Number=2.0, ac::Number=0.5, as0::Number=0.5, seed::Number=1234)

State optimization algorithm: NM.
- `max_episode`: The number of populations.
- `state_num`: The number of the input states.
- `nelder_mead`: Initial guesses of the optimization variables.
- `ar`: Reflection constant.
- `ae`: Expansion constant.
- `ac`: Constraction constant.
- `as0`: Shrink constant.
- `seed`: Random seed.
"""
NM(;max_episode::Int=1000, state_num::Int=10, nelder_mead=missing, ar::Number=1.0, ae::Number=2.0, ac::Number=0.5, as0::Number=0.5, seed::Number=1234) =
    NM(max_episode, state_num, nelder_mead, ar, ae, ac, as0, MersenneTwister(seed))

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

include("AD.jl")
include("DDPG.jl")
include("DE.jl")
include("GRAPE.jl")
include("NM.jl")
include("PSO.jl")
