abstract type AbstractAlgorithm end

abstract type AbstractGRAPE <: AbstractAlgorithm end
struct GRAPE <: AbstractGRAPE
    max_episode::Number
    ϵ::Number
end

struct GRAPE_Adam <: AbstractGRAPE
    max_episode::Number
    ϵ::Number
    beta1::Number
    beta2::Number
end

GRAPE(max_episode, ϵ, beta1, beta2) = GRAPE_Adam(max_episode, ϵ, beta1, beta2)

abstract type AbstractautoGRAPE <: AbstractAlgorithm end
struct autoGRAPE <: AbstractautoGRAPE
    max_episode::Number
    ϵ::Number
end

struct autoGRAPE_Adam <: AbstractautoGRAPE
    max_episode::Number
    ϵ::Number
    beta1::Number
    beta2::Number
end

autoGRAPE(max_episode, ϵ, beta1, beta2) = autoGRAPE_Adam(max_episode, ϵ, beta1, beta2)

abstract type AbstractAD <:  AbstractAlgorithm end
struct AD <: AbstractAD
    max_episode::Number
    ϵ::Number
end

struct AD_Adam <: AbstractAD
    max_episode::Number
    ϵ::Number
    beta1::Number
    beta2::Number
end

AD(max_episode, ϵ, beta1, beta2) = AD_Adam(max_episode, ϵ, beta1, beta2)

struct PSO <: AbstractAlgorithm
    max_episode::Union{T,Vector{T}} where {T<:Number}
    p_num::Number
    ini_particle::Tuple
    c0::Number
    c1::Number
    c2::Number
    rng::AbstractRNG
end

PSO(max_episode, p_num, ini_particle, c0, c1, c2) =
    PSO(max_episode, p_num, ini_particle, c0, c1, c2, GLOBAL_RNG)
PSO(max_episode, p_num, ini_particle, c0, c1, c2, seed::Number) =
    PSO(max_episode, p_num, ini_particle, c0, c1, c2, StableRNG(seed))

struct DE <: AbstractAlgorithm
    max_episode::Number
    p_num::Number
    ini_population::Tuple
    c::Number
    cr::Number
    rng::AbstractRNG
end

DE(max_episode, p_num, c, cr) = DE(max_episode, p_num, c, cr, GLOBAL_RNG)
DE(max_episode, p_num, ini_population, c, cr, seed::Number) =
    DE(max_episode, p_num, ini_population, c, cr, MersenneTwister(seed))

struct DDPG <: AbstractAlgorithm
    max_episode::Number
    layer_num::Number
    layer_dim::Number
    rng::AbstractRNG
end

DDPG(max_episode, layer_num, layer_dim) =
    DDPG(max_episode, layer_num, layer_dim, GLOBAL_RNG)
DDPG(max_episode, layer_num, layer_dim, seed::Number) =
    DDPG(max_episode, layer_num, layer_dim, MersenneTwister(seed))

struct NM <: AbstractAlgorithm
    max_episode::Number
    state_num::Number
    ini_state::AbstractVector
    ar::Number
    ae::Number
    ac::Number
    as0::Number
    rng::AbstractRNG
end

NM(max_episode, state_num, nelder_mead, ar, ae, ac, as0) =
    NM(max_episode, state_num, nelder_mead, ar, ae, ac, as0, GLOBAL_RNG)
NM(max_episode, state_num, nelder_mead, ar, ae, ac, as0, seed::Number) =
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
