struct ControlEnvParams{T, M}
    freeHamiltonian::Matrix{T}
    Hamiltonian_derivative::Vector{Matrix{T}}
    ρ_initial::Matrix{T}
    times::Vector{M}
    Liouville_operator::Vector{Matrix{T}}
    γ::Vector{M}
    control_Hamiltonian::Vector{Matrix{T}}
    control_coefficients::Vector{Vector{M}}
    ctrl_bound::M
    W::AbstractArray
end

mutable struct ControlEnv{T, M, RNG} <: AbstractEnv
    params::ControlEnvParams{T, M}
    state::Vector{Vector{T}}
    done::Bool
    C::Float64
    rng::RNG
    reward::Float64
    tspan::Vector{T}
    tnum::Int
end

function ControlEnv(;
    T = ComplexF64,
    M = Float64,
    freeHamiltonian,
    Hamiltonian_derivative,
    ρ_initial,
    times,
    Liouville_operator,
    γ,
    control_Hamiltonian,
    control_coefficients,
    ctrl_bound,
    para_num = Hamiltonian_derivative|>length,
    W = I(para_num),
    )
    params = ControlEnvParams{T, M}(
        freeHamiltonian,
        Hamiltonian_derivative,
        ρ_initial,
        times,
        Liouville_operator,
        γ,
        control_Hamiltonian,
        control_coefficients,
        ctrl_bound,
        W
    )
end

function RLBase.reset!(env::ControlEnv{T, M}) where {T <: Complex}
    env.state[1] = env.params.ρ_initial
    env.state[2] = repeat(env.params.ρ_initial|>similar, env.params.control_Hamiltonian|>length)
    env.t = 0
    env.done = false
    env.reward = zero{M}
    nothing
end

RLBase.action_space(env::ControlEnv{T,M}) where {T, M} = Space(
    ClosedInterval{M}[]
) 
RLBase.state_space(env::ControlEnv) = env.observation_space
RLBase.reward(env::ControlEnv) = env.reward
RLBase.is_terminated(env::ControlEnv) = env.done 
RLBase.state(env::ControlEnv) = env.state

function (env::ControlEnv)(a)
    @assert a in env.action_space
    _step!(env, a)
end

function _step!(env::ControlEnv, a)
    env.t += 1
    ρₜ, ∂ₓρₜ = env.state

end

Random.seed!(env::ControlEnv, seed) = Random.seed!(env.rng, seed)