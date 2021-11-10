# using Base: _start
using Flux: glorot_normal, glorot_uniform
using StableRNGs
using Flux
using Flux.Losses
using IntervalSets
using Random
# using Plots
using PyPlot
struct ControlEnvParams{T, M}
    freeHamiltonian::Matrix{T}
    Hamiltonian_derivative::Vector{Matrix{T}}
    ρ_initial::Matrix{T}
    times::Vector{M}
    Liouville_operator::Vector{Matrix{T}}
    γ::Vector{M}
    control_Hamiltonian::Vector{Matrix{T}}
    control_coefficients::Vector{Vector{M}}
    ctrl_interval::Int
    ctrl_bound::Vector{M}
    W::Matrix{M}
    dim::Int
end

mutable struct ControlEnv{T, M, R<:AbstractRNG} <: AbstractEnv
    params::ControlEnvParams{T, M}
    action_space::Space
    state_space::Space
    state::Vector{M}
    dstate::Vector{Matrix{T}}
    done::Bool
    rng::R
    reward::Float64
    t::Int
    tspan::Vector{M}
    tnum::Int
    cnum::Int
    ctrl_num::Int
    para_num::Int
    f_init::Vector{M}
    f_list::Vector{M}
    a_list::Vector{Vector{M}}
end

function ControlEnv(;
    T = ComplexF64,
    M = Float64,
    params::ControlEnvParams,   
    para_num = params.Hamiltonian_derivative|>length,
    ctrl_num = params.control_coefficients|>length,
    rng = Random.GLOBAL_RNG
    )
    tnum = params.times|>length
    cnum = tnum÷params.ctrl_interval
    state = params.ρ_initial
    dstate = [state|>zero for _ in 1:para_num]
    state = state|>state_flatten
    action_space = Space([params.ctrl_bound[1]..params.ctrl_bound[end] for _ in 1:ctrl_num])
    state_space = Space(fill(M(-1e38)..M(1e38), length(state)))
    f_init = [0., (QFIM_saveall(params)[params.ctrl_interval:params.ctrl_interval:end].|>(x->x=1/(x|>inv|>tr)))...] 
    a_list = [ Vector{Float64}() for _ in 1:ctrl_num]
    f_list = Vector{Float64}()
    env = ControlEnv(params, action_space, state_space, state, dstate, true, rng, 0., 0, params.times, tnum, cnum, ctrl_num, para_num, f_init, f_list, a_list)
    reset!(env)
    env
end

function Base.rsplit( v, l::Int)
    u = reshape(v,l,length(v)÷l)
    [u[:,i] for i=1:size(u,2)]
end

state_flatten(s) = vcat((s|>reim.|>vec)...)
rsplit_half(v) = Base.rsplit(v, length(v)÷2)
density_matrix(s) = complex.(rsplit_half(s)...)|>vec2mat

function RLBase.reset!(env::ControlEnv)
    state = env.params.ρ_initial
    env.dstate =[state|>zero for _ in 1:(env.para_num)]
    env.state =  state|>state_flatten
    env.t = 1
    env.done = false
    env.reward = 0.
    nothing
end

RLBase.action_space(env::ControlEnv) = env.action_space
RLBase.state_space(env::ControlEnv) = env.state_space
RLBase.reward(env::ControlEnv) = env.reward
RLBase.is_terminated(env::ControlEnv) = env.done 
RLBase.state(env::ControlEnv) = env.state

function (env::ControlEnv)(a)
    @assert a in env.action_space
    _step!(env, a)
end

function _step!(env::ControlEnv, a)
    # append!.(env.a_list, a)
    # @show a
    env.t += 1
    ρₜ, ∂ₓρₜ = env.state|>density_matrix, env.dstate
    ρₜₙ, ∂ₓρₜₙ = propagate(ρₜ, ∂ₓρₜ, env.params, a, env.t)
    env.state = ρₜₙ|>state_flatten
    env.dstate = ∂ₓρₜₙ
    env.done = env.t >= env.cnum
    f_current = 1/((env.params.W*QFIM_ori(ρₜₙ, ∂ₓρₜₙ))|>inv|>tr)
    reward_current = log(f_current/env.f_init[env.t])
    env.reward = reward_current
    if env.done
        append!(env.f_list, f_current/env.params.times[end])
    end
    nothing 
end

Random.seed!(env::ControlEnv, seed) = Random.seed!(env.rng, seed)

function DDPG_QFIM(
    params::ControlEnvParams,
    seed = 123,
    layer_num = 3,
    layer_dim = 200
)
    rng = StableRNG(seed)
    env = ControlEnv(params = params, rng=rng)
    ns = 2*params.dim^2
    na = env.ctrl_num

    init = glorot_uniform(rng)

    create_actor() = Chain(
        Dense(ns, layer_dim, relu; init = init),
        [Dense(layer_dim, layer_dim, relu; init = init) for _ in 1:layer_num]...,
        Dense(layer_dim, na, tanh; init = init),
    )

    create_critic() = Chain(
        Dense(ns + na, layer_dim, relu; init = init),
        [Dense(layer_dim, layer_dim, relu; init = init) for _ in 1:layer_num]...,
        Dense(layer_dim, 1; init = init),
    )

    agent = Agent(
        policy = DDPGPolicy(
            behavior_actor = NeuralNetworkApproximator(
                model = create_actor(),
                optimizer = ADAM(),
            ),
            behavior_critic = NeuralNetworkApproximator(
                model = create_critic(),
                optimizer = ADAM(),
            ),
            target_actor = NeuralNetworkApproximator(
                model = create_actor(),
                optimizer = ADAM(),
            ),
            target_critic = NeuralNetworkApproximator(
                model = create_critic(),
                optimizer = ADAM(),
            ),
            γ = 0.99f0,
            ρ = 0.995f0,
            na = env.ctrl_num,
            batch_size = 64*env.cnum,
            start_steps = 100*env.cnum,
            start_policy = RandomPolicy(Space([-1.0..1.0 for _ in 1:env.ctrl_num]); rng = rng),
            update_after = 100*env.cnum,
            update_freq = 1*env.cnum,
            act_limit = env.params.ctrl_bound[end],
            act_noise = 0.01,
            rng = rng,
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 400*env.cnum,
            state = Vector{Float64} => (ns,),
            action = Vector{Float64} => (na, ),
        ),
    )

    stop_condition = StopAfterStep(10000*env.cnum, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    run(agent, env, stop_condition, hook)
    #plot([env.f_list, hook.rewards], layout = (1,2))
    plt.plot(env.f_list)
    plt.savefig("1.png")
    plt.clf()
    plt.plot(hook.rewards)
    plt.savefig("2.png")
    plt.clf()
end


