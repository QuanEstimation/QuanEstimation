using ReinforcementLearning

struct ControlEnvParams{T}

end

mutable struct ControlEnv{} <: AbstractEnv
    params::ControlEnvParams{T}
    action_space::A
    observation_space::Space{Vector{ClosedInterval}}
    state::Vector{Vector{T}}
    done::Bool
    tspan::Vector{T}
    rng::R
    reward::T
    n_actions::Int
end
