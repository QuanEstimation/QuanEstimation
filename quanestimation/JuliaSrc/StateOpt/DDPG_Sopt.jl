using Flux: glorot_normal, glorot_uniform
using StableRNGs
using Flux
using Flux.Losses
using IntervalSets

function Base.rsplit( v, l::Int)
    u = reshape(v,l,length(v)÷l)
    [u[:,i] for i=1:size(u,2)]
end

state_flatten(s) = vcat((s|>reim.|>vec)...)
rsplit_half(v) = Base.rsplit(v, length(v)÷2)
to_psi(s) = complex.(rsplit_half(s)...)

############# time-independent Hamiltonian (noiseless) ################
mutable struct ControlEnv_noiseless{T, M, R<:AbstractRNG} <: AbstractEnv
    Measurement::Vector{Matrix{T}}
    params::TimeIndepend_noiseless{T, M}
    action_space::Space
    state_space::Space
    state::Vector{M}
    done::Bool
    rng::R
    reward::Float64
    total_reward::Float64
    tspan::Vector{M}
    ctrl_num::Int
    para_num::Int
    f_ini::Float64
    f_list::Vector{M}
    reward_all::Vector{M}
    episode::Int
    quantum::Bool
    SinglePara::Bool
    save_file::Bool
end

function ControlEnv_noiseless(;T=ComplexF64, M=Float64, Measurement, params::TimeIndepend_noiseless, para_num=params.Hamiltonian_derivative|>length,
                    rng=Random.GLOBAL_RNG, episode, quantum, SinglePara, save_file)
    state = params.psi
    state = state |> state_flatten
    ctrl_num = length(state)
    action_space = Space(fill(-1.0e35..1.0e35, length(state)))
    state_space = Space(fill(-1.0e35..1.0e35, length(state))) 

    f_ini = 0.0
    if quantum 
        F_ini = QFIM_TimeIndepend(params.freeHamiltonian, params.Hamiltonian_derivative, params.psi, params.tspan, params.accuracy)
        f_ini = 1.0/real(tr(params.W*pinv(F_ini)))
    else
        F_ini = CFIM_TimeIndepend(Measurement, params.freeHamiltonian, params.Hamiltonian_derivative, params.psi, params.tspan, params.accuracy)
        f_ini = 1.0/real(tr(params.W*pinv(F_ini)))
    end

    f_list = Vector{Float64}()
    reward_all = Vector{Float64}()
    env = ControlEnv_noiseless(Measurement, params, action_space, state_space, state, true, rng, 0.0, 0.0, params.tspan, ctrl_num, 
                               para_num, f_ini, f_list, reward_all, episode, quantum, SinglePara, save_file)
    reset!(env)
    env
end

function RLBase.reset!(env::ControlEnv_noiseless)
    state = env.params.psi
    env.state = state |> state_flatten
    env.done = false
    nothing
end

RLBase.action_space(env::ControlEnv_noiseless) = env.action_space
RLBase.state_space(env::ControlEnv_noiseless) = env.state_space
RLBase.reward(env::ControlEnv_noiseless) = env.reward
RLBase.is_terminated(env::ControlEnv_noiseless) = env.done 
RLBase.state(env::ControlEnv_noiseless) = env.state

function (env::ControlEnv_noiseless)(a)
    # @assert a in env.action_space
    _step_noiseless!(env, a, Val(env.quantum), Val(env.SinglePara), Val(env.save_file))
end

####################### step functions #########################
#### quantum single parameter estimation and save_file=true ####
function _step_noiseless!(env::ControlEnv_noiseless, a, ::Val{true}, ::Val{true}, ::Val{true})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    F_tp = QFIM_TimeIndepend(env.params.freeHamiltonian, env.params.Hamiltonian_derivative, env.params.psi, env.params.tspan, env.params.accuracy)
    f_current = 1.0/real(tr(env.params.W*pinv(F_tp)))
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.rewards
    env.done = true

    append!(env.f_list, f_current)
    append!(env.reward_all, env.reward)
    SaveFile_state_ddpg(f_current, env.reward, env.params.psi)
    env.episode += 1
    print("current QFI is ", f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### quantum single parameter estimation and save_file=false ####
function _step_noiseless!(env::ControlEnv_noiseless, a, ::Val{true}, ::Val{true}, ::Val{false})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    F_tp = QFIM_TimeIndepend(env.params.freeHamiltonian, env.params.Hamiltonian_derivative, env.params.psi, env.params.tspan, env.params.accuracy)
    f_current = 1.0/real(tr(env.params.W*pinv(F_tp)))
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, f_current)
    append!(env.reward_all, env.reward)
    env.episode += 1
    print("current QFI is ", f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### quantum multiparameter estimation and save_file=true ####
function _step_noiseless!(env::ControlEnv_noiseless, a, ::Val{true}, ::Val{false}, ::Val{true})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    F_tp = QFIM_TimeIndepend(env.params.freeHamiltonian, env.params.Hamiltonian_derivative, env.params.psi, env.params.tspan, env.params.accuracy)
    f_current = 1.0/real(tr(env.params.W*pinv(F_tp)))
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, 1.0/f_current)
    append!(env.reward_all, env.reward)
    SaveFile_state_ddpg(1.0/f_current, env.reward, env.params.psi)
    env.episode += 1
    print("current value of Tr(WF^{-1}) is ", 1.0/f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### quantum multiparameter estimation and save_file=false ####
function _step_noiseless!(env::ControlEnv_noiseless, a, ::Val{true}, ::Val{false}, ::Val{false})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    F_tp = QFIM_TimeIndepend(env.params.freeHamiltonian, env.params.Hamiltonian_derivative, env.params.psi, env.params.tspan, env.params.accuracy)
    f_current = 1.0/real(tr(env.params.W*pinv(F_tp)))
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, 1.0/f_current)
    append!(env.reward_all, env.reward)
    env.episode += 1
    print("current value of Tr(WF^{-1}) is ", 1.0/f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### classical single parameter estimation and save_file=true ####
function _step_noiseless!(env::ControlEnv_noiseless, a, ::Val{false}, ::Val{true}, ::Val{true})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    F_tp = CFIM_TimeIndepend(env.Measurement, env.params.freeHamiltonian, env.params.Hamiltonian_derivative, env.params.psi, 
                             env.params.tspan, env.params.accuracy)
    f_current = 1.0/real(tr(env.params.W*pinv(F_tp)))
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, f_current)
    append!(env.reward_all, env.reward)
    SaveFile_state_ddpg(f_current, env.reward, env.params.psi)
    
    env.episode += 1
    print("current CFI is ", f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### classical single parameter estimation and save_file=false ####
function _step_noiseless!(env::ControlEnv_noiseless, a, ::Val{false}, ::Val{true}, ::Val{false})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    F_tp = CFIM_TimeIndepend(env.Measurement, env.params.freeHamiltonian, env.params.Hamiltonian_derivative, env.params.psi, 
                             env.params.tspan, env.params.accuracy)
    f_current = 1.0/real(tr(env.params.W*pinv(F_tp)))
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, f_current)
    append!(env.reward_all, env.reward)
    env.episode += 1
    print("current CFI is ", f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### classical multiparameter estimation and save_file=true ####
function _step_noiseless!(env::ControlEnv_noiseless, a, ::Val{false}, ::Val{false}, ::Val{true})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    F_tp = CFIM_TimeIndepend(env.Measurement, env.params.freeHamiltonian, env.params.Hamiltonian_derivative, env.params.psi, 
                             env.params.tspan, env.params.accuracy)
    f_current = 1.0/real(tr(env.params.W*pinv(F_tp)))
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, 1.0/f_current)
    append!(env.reward_all, env.reward)
    SaveFile_state_ddpg(1.0/f_current, env.reward, env.params.psi)
    env.episode += 1
    print("current value of Tr(WI^{-1}) is ", 1.0/f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### classical multiparameter estimation and save_file=false ####
function _step_noiseless!(env::ControlEnv_noiseless, a, ::Val{false}, ::Val{false}, ::Val{false})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    F_tp = CFIM_TimeIndepend(env.Measurement, env.params.freeHamiltonian, env.params.Hamiltonian_derivative, env.params.psi, 
                             env.params.tspan, env.params.accuracy)
    f_current = 1.0/real(tr(env.params.W*pinv(F_tp)))
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, 1.0/f_current)
    append!(env.reward_all, env.reward)
    env.episode += 1
    print("current value of Tr(WI^{-1}) is ", 1.0/f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

Random.seed!(env::ControlEnv_noiseless, seed) = Random.seed!(env.rng, seed)

function QFIM_DDPG_Sopt(params::TimeIndepend_noiseless, layer_num, layer_dim, seed, max_episode, save_file)
    rng = StableRNG(seed)
    env = ControlEnv_noiseless(Measurement=Vector{Matrix{ComplexF64}}(undef, 1), params=params, rng=rng, episode=1, quantum=true, 
                     SinglePara=(length(params.Hamiltonian_derivative)==1), save_file=save_file)
    ns = 2*length(params.psi)
    na = env.ctrl_num
    init = glorot_uniform(rng)

    create_actor() = Chain(Dense(ns, layer_dim, relu; init=init),
                           [Dense(layer_dim, layer_dim, relu; init=init) for _ in 1:layer_num]...,
                            Dense(layer_dim, na, tanh; init=init),)

    create_critic() = Chain(Dense(ns+na, layer_dim, relu; init=init),
                            [Dense(layer_dim, layer_dim, relu; init=init) for _ in 1:layer_num]...,
                            Dense(layer_dim, 1; init=init),)

    agent = Agent(policy=DDPGPolicy(behavior_actor=NeuralNetworkApproximator(model=create_actor(), optimizer=ADAM(),),
                                    behavior_critic=NeuralNetworkApproximator(model=create_critic(), optimizer=ADAM(),),
                                    target_actor=NeuralNetworkApproximator(model=create_actor(), optimizer=ADAM(),),
                                    target_critic=NeuralNetworkApproximator(model=create_critic(), optimizer=ADAM(),),
                                    γ=0.99f0, ρ=0.995f0, na=env.ctrl_num, batch_size=64, start_steps=100,
                                    start_policy=RandomPolicy(Space([-1.0..1.0 for _ in 1:env.ctrl_num]); rng=rng),
                                    update_after=100, update_freq=1, act_limit=1.0e35,
                                    act_noise=0.01, rng=rng,),
                  trajectory=CircularArraySARTTrajectory(capacity=400, state=Vector{Float64} => (ns,), action=Vector{Float64} => (na,),),)

    println("state optimization")
    if length(params.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: deep deterministic policy gradient algorithm (DDPG)")
        println("initial QFI is $(1.0/env.f_ini)")
        append!(env.f_list, 1.0/env.f_ini)
    else
        println("multiparameter scenario")
        println("search algorithm: deep deterministic policy gradient algorithm (DDPG)")
        println("initial value of Tr(WF^{-1}) is $(env.f_ini)")
        append!(env.f_list, env.f_ini)
    end

    if env.save_file
        SaveFile_state_ddpg(env.f_list, env.reward_all, env.params.psi)
    end

    stop_condition = StopAfterStep(max_episode, is_show_progress=false)
    hook = TotalRewardPerEpisode(is_display_on_exit=false)
    run(agent, env, stop_condition, hook)

    if !env.save_file
        SaveFile_state_ddpg(env.f_list, env.reward_all, env.params.psi)
    end
    print("\e[2K")
    println("Iteration over, data saved.")
    if length(params.Hamiltonian_derivative) == 1
        println("Final QFI is ", env.f_list[end])
    else
        println("Final value of Tr(WF^{-1}) is ", env.f_list[end])
    end
end

function CFIM_DDPG_Sopt(Measurement, params::TimeIndepend_noiseless, layer_num, layer_dim, seed, max_episode, save_file)
    rng = StableRNG(seed)
    env = ControlEnv_noiseless(Measurement=Measurement, params=params, rng=rng, episode=1, quantum=false,
                     SinglePara=(length(params.Hamiltonian_derivative)==1), save_file=save_file)
    ns = 2*length(params.psi)
    na = env.ctrl_num
    init = glorot_uniform(rng)

    create_actor() = Chain(Dense(ns, layer_dim, relu; init=init),
                           [Dense(layer_dim, layer_dim, relu; init=init) for _ in 1:layer_num]...,
                           Dense(layer_dim, na, tanh; init=init),)

    create_critic() = Chain(Dense(ns+na, layer_dim, relu; init=init),
                            [Dense(layer_dim, layer_dim, relu; init=init) for _ in 1:layer_num]...,
                            Dense(layer_dim, 1; init=init),)

    agent = Agent(policy=DDPGPolicy(behavior_actor=NeuralNetworkApproximator(model=create_actor(), optimizer=ADAM(),),
                                    behavior_critic=NeuralNetworkApproximator(model=create_critic(), optimizer=ADAM(),),
                                    target_actor=NeuralNetworkApproximator(model=create_actor(), optimizer=ADAM(),),
                                    target_critic=NeuralNetworkApproximator(model=create_critic(), optimizer=ADAM(),),
                                    γ=0.99f0, ρ=0.995f0, na=env.ctrl_num, batch_size=64, start_steps=100,
                                    start_policy=RandomPolicy(Space([-1.0..1.0 for _ in 1:env.ctrl_num]); rng=rng),
                                    update_after=100, update_freq=1, act_limit=1.0e35,
                                    act_noise=0.01, rng=rng,),
                  trajectory=CircularArraySARTTrajectory(capacity=400, state=Vector{Float64} => (ns,), action=Vector{Float64} => (na,),),)

    println("state optimization")
    if length(params.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: deep deterministic policy gradient algorithm (DDPG)")
        println("initial CFI is $(1.0/env.f_ini)")
        append!(env.f_list, 1.0/env.f_ini)
    else
        println("multiparameter scenario")
        println("search algorithm: deep deterministic policy gradient algorithm (DDPG)")
        println("initial value of Tr(WI^{-1}) is $(env.f_ini)")
        append!(env.f_list, env.f_ini)
    end

    if env.save_file
        SaveFile_state_ddpg(env.f_list, env.reward_all, env.params.psi)
    end

    stop_condition = StopAfterStep(max_episode, is_show_progress=false)
    hook = TotalRewardPerEpisode(is_display_on_exit=false)
    run(agent, env, stop_condition, hook)

    if !env.save_file
        SaveFile_state_ddpg(env.f_list, env.reward_all, env.params.psi)
    end
    print("\e[2K")
    println("Iteration over, data saved.")
    if length(params.Hamiltonian_derivative) == 1
        println("Final CFI is ", env.f_list[end])
    else
        println("Final value of Tr(WI^{-1}) is ", env.f_list[end])
    end
end


############# time-independent Hamiltonian (noise) ################
mutable struct ControlEnv_noise{T, M, R<:AbstractRNG} <: AbstractEnv
    Measurement::Vector{Matrix{T}}
    params::TimeIndepend_noise{T, M}
    action_space::Space
    state_space::Space
    state::Vector{M}
    done::Bool
    rng::R
    reward::Float64
    total_reward::Float64
    tspan::Vector{M}
    ctrl_num::Int
    para_num::Int
    f_ini::Float64
    f_list::Vector{M}
    reward_all::Vector{M}
    episode::Int
    quantum::Bool
    SinglePara::Bool
    save_file::Bool
end

function ControlEnv_noise(;T=ComplexF64, M=Float64, Measurement, params::TimeIndepend_noise, para_num=params.Hamiltonian_derivative|>length,
                    rng=Random.GLOBAL_RNG, episode, quantum, SinglePara, save_file)
    state = params.psi
    state = state |> state_flatten
    ctrl_num = length(state)
    action_space = Space(fill(-1.0e35..1.0e35, length(state)))
    state_space = Space(fill(-1.0e35..1.0e35, length(state))) 

    f_ini = 0.0
    if quantum 
        F_ini = QFIM_TimeIndepend(params.freeHamiltonian, params.Hamiltonian_derivative, params.psi*(params.psi)', 
                                  params.decay_opt, params.γ, params.tspan, params.accuracy)
        f_ini = 1.0/real(tr(params.W*pinv(F_ini)))
    else
        F_ini = CFIM_TimeIndepend(Measurement, params.freeHamiltonian, params.Hamiltonian_derivative, params.psi*(params.psi)', 
                                  params.decay_opt, params.γ, params.tspan, params.accuracy)
        f_ini = 1.0/real(tr(params.W*pinv(F_ini)))
    end

    f_list = Vector{Float64}()
    reward_all = Vector{Float64}()
    env = ControlEnv_noise(Measurement, params, action_space, state_space, state, true, rng, 0.0, 0.0, params.tspan, ctrl_num, 
                           para_num, f_ini, f_list, reward_all, episode, quantum, SinglePara, save_file)
    reset!(env)
    env
end

function RLBase.reset!(env::ControlEnv_noise)
    state = env.params.psi
    env.state = state |> state_flatten
    env.done = false
    nothing
end

RLBase.action_space(env::ControlEnv_noise) = env.action_space
RLBase.state_space(env::ControlEnv_noise) = env.state_space
RLBase.reward(env::ControlEnv_noise) = env.reward
RLBase.is_terminated(env::ControlEnv_noise) = env.done 
RLBase.state(env::ControlEnv_noise) = env.state

function (env::ControlEnv_noise)(a)
    # @assert a in env.action_space
    _step_noise!(env, a, Val(env.quantum), Val(env.SinglePara), Val(env.save_file))
end

####################### step functions #########################
#### quantum single parameter estimation and save_file=true ####
function _step_noise!(env::ControlEnv_noise, a, ::Val{true}, ::Val{true}, ::Val{true})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    F_tp = QFIM_TimeIndepend(env.params.freeHamiltonian, env.params.Hamiltonian_derivative, env.params.psi*(env.params.psi)', 
                             env.params.decay_opt, env.params.γ, env.params.tspan, env.params.accuracy)
    f_current = 1.0/real(tr(env.params.W*pinv(F_tp)))
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, f_current)
    append!(env.reward_all, env.reward)
    SaveFile_state_ddpg(f_current, env.reward, env.params.psi)
    env.episode += 1
    print("current QFI is ", f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### quantum single parameter estimation and save_file=false ####
function _step_noise!(env::ControlEnv_noise, a, ::Val{true}, ::Val{true}, ::Val{false})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    F_tp = QFIM_TimeIndepend(env.params.freeHamiltonian, env.params.Hamiltonian_derivative, env.params.psi*(env.params.psi)', 
                             env.params.decay_opt, env.params.γ, env.params.tspan, env.params.accuracy)
    f_current = 1.0/real(tr(env.params.W*pinv(F_tp)))
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, f_current)
    append!(env.reward_all, env.reward)
    env.episode += 1
    print("current QFI is ", f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### quantum multiparameter estimation and save_file=true ####
function _step_noise!(env::ControlEnv_noise, a, ::Val{true}, ::Val{false}, ::Val{true})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    F_tp = QFIM_TimeIndepend(env.params.freeHamiltonian, env.params.Hamiltonian_derivative, env.params.psi*(env.params.psi)', 
                             env.params.decay_opt, env.params.γ, env.params.tspan, env.params.accuracy)
    f_current = 1.0/real(tr(env.params.W*pinv(F_tp)))
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, 1.0/f_current)
    append!(env.reward_all, env.reward)
    SaveFile_state_ddpg(1.0/f_current, env.reward, env.params.psi)
    env.episode += 1
    print("current value of Tr(WF^{-1}) is ", 1.0/f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### quantum multiparameter estimation and save_file=false ####
function _step_noise!(env::ControlEnv_noise, a, ::Val{true}, ::Val{false}, ::Val{false})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    F_tp = QFIM_TimeIndepend(env.params.freeHamiltonian, env.params.Hamiltonian_derivative, env.params.psi*(env.params.psi)', 
                             env.params.decay_opt, env.params.γ, env.params.tspan, env.params.accuracy)
    f_current = 1.0/real(tr(env.params.W*pinv(F_tp)))
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, 1.0/f_current)
    append!(env.reward_all, env.reward)
    env.episode += 1
    print("current value of Tr(WF^{-1}) is ", 1.0/f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### classical single parameter estimation and save_file=true ####
function _step_noise!(env::ControlEnv_noise, a, ::Val{false}, ::Val{true}, ::Val{true})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    F_tp = CFIM_TimeIndepend(env.Measurement, env.params.freeHamiltonian, env.params.Hamiltonian_derivative, env.params.psi*(env.params.psi)', 
                             env.params.decay_opt, env.params.γ, env.params.tspan, env.params.accuracy)
    f_current = 1.0/real(tr(env.params.W*pinv(F_tp)))
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, f_current)
    append!(env.reward_all, env.reward)
    SaveFile_state_ddpg(f_current, env.reward, env.params.psi)
    
    env.episode += 1
    print("current CFI is ", f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### classical single parameter estimation and save_file=false ####
function _step_noise!(env::ControlEnv_noise, a, ::Val{false}, ::Val{true}, ::Val{false})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    F_tp = CFIM_TimeIndepend(env.Measurement, env.params.freeHamiltonian, env.params.Hamiltonian_derivative, env.params.psi*(env.params.psi)', 
                             env.params.decay_opt, env.params.γ, env.params.tspan, env.params.accuracy)
    f_current = 1.0/real(tr(env.params.W*pinv(F_tp)))
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, f_current)
    append!(env.reward_all, env.reward)
    env.episode += 1
    print("current CFI is ", f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### classical multiparameter estimation and save_file=true ####
function _step_noise!(env::ControlEnv_noise, a, ::Val{false}, ::Val{false}, ::Val{true})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    F_tp = CFIM_TimeIndepend(env.Measurement, env.params.freeHamiltonian, env.params.Hamiltonian_derivative, env.params.psi*(env.params.psi)', 
                             env.params.decay_opt, env.params.γ, env.params.tspan, env.params.accuracy)
    f_current = 1.0/real(tr(env.params.W*pinv(F_tp)))
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, 1.0/f_current)
    append!(env.reward_all, env.reward)
    SaveFile_state_ddpg(1.0/f_current, env.reward, env.params.psi)
    env.episode += 1
    print("current value of Tr(WI^{-1}) is ", 1.0/f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### classical multiparameter estimation and save_file=false ####
function _step_noise!(env::ControlEnv_noise, a, ::Val{false}, ::Val{false}, ::Val{false})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    F_tp = CFIM_TimeIndepend(env.Measurement, env.params.freeHamiltonian, env.params.Hamiltonian_derivative, env.params.psi*(env.params.psi)', 
                             env.params.decay_opt, env.params.γ, env.params.tspan, env.params.accuracy)
    f_current = 1.0/real(tr(env.params.W*pinv(F_tp)))
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, 1.0/f_current)
    append!(env.reward_all, env.reward)
    env.episode += 1
    print("current value of Tr(WI^{-1}) is ", 1.0/f_current, " ($(env.episode) episodes)    \r")

    nothing 
end

Random.seed!(env::ControlEnv_noise, seed) = Random.seed!(env.rng, seed)

function QFIM_DDPG_Sopt(params::TimeIndepend_noise, layer_num, layer_dim, seed, max_episode, save_file)
    rng = StableRNG(seed)
    env = ControlEnv_noise(Measurement=Vector{Matrix{ComplexF64}}(undef, 1), params=params, rng=rng, episode=1, quantum=true, 
                     SinglePara=(length(params.Hamiltonian_derivative)==1), save_file=save_file)
    ns = 2*length(params.psi)
    na = env.ctrl_num
    init = glorot_uniform(rng)

    create_actor() = Chain(Dense(ns, layer_dim, relu; init=init),
                           [Dense(layer_dim, layer_dim, relu; init=init) for _ in 1:layer_num]...,
                            Dense(layer_dim, na, tanh; init=init),)

    create_critic() = Chain(Dense(ns+na, layer_dim, relu; init=init),
                            [Dense(layer_dim, layer_dim, relu; init=init) for _ in 1:layer_num]...,
                            Dense(layer_dim, 1; init=init),)

    agent = Agent(policy=DDPGPolicy(behavior_actor=NeuralNetworkApproximator(model=create_actor(), optimizer=ADAM(),),
                                    behavior_critic=NeuralNetworkApproximator(model=create_critic(), optimizer=ADAM(),),
                                    target_actor=NeuralNetworkApproximator(model=create_actor(), optimizer=ADAM(),),
                                    target_critic=NeuralNetworkApproximator(model=create_critic(), optimizer=ADAM(),),
                                    γ=0.99f0, ρ=0.995f0, na=env.ctrl_num, batch_size=64, start_steps=100,
                                    start_policy=RandomPolicy(Space([-1.0..1.0 for _ in 1:env.ctrl_num]); rng=rng),
                                    update_after=100, update_freq=1, act_limit=1.0e35,
                                    act_noise=0.01, rng=rng,),
                  trajectory=CircularArraySARTTrajectory(capacity=400, state=Vector{Float64} => (ns,), action=Vector{Float64} => (na,),),)

    println("state optimization")
    if length(params.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: deep deterministic policy gradient algorithm (DDPG)")
        println("initial QFI is $(1.0/env.f_ini)")
        append!(env.f_list, 1.0/env.f_ini)
    else
        println("multiparameter scenario")
        println("search algorithm: deep deterministic policy gradient algorithm (DDPG)")
        println("initial value of Tr(WF^{-1}) is $(env.f_ini)")
        append!(env.f_list, env.f_ini)
    end

    if env.save_file
        SaveFile_state_ddpg(env.f_list, env.reward_all, env.params.psi)
    end

    stop_condition = StopAfterStep(max_episode, is_show_progress=false)
    hook = TotalRewardPerEpisode(is_display_on_exit=false)
    run(agent, env, stop_condition, hook)

    if !env.save_file
        SaveFile_state_ddpg(env.f_list, env.reward_all, env.params.psi)
    end
    print("\e[2K")
    println("Iteration over, data saved.")
    if length(params.Hamiltonian_derivative) == 1
        println("Final QFI is ", env.f_list[end])
    else
        println("Final value of Tr(WF^{-1}) is ", env.f_list[end])
    end
end

function CFIM_DDPG_Sopt(Measurement, params::TimeIndepend_noise, layer_num, layer_dim, seed, max_episode, save_file)
    rng = StableRNG(seed)
    env = ControlEnv_noise(Measurement=Measurement, params=params, rng=rng, episode=1, quantum=false,
                     SinglePara=(length(params.Hamiltonian_derivative)==1), save_file=save_file)
    ns = 2*length(params.psi)
    na = env.ctrl_num
    init = glorot_uniform(rng)

    create_actor() = Chain(Dense(ns, layer_dim, relu; init=init),
                           [Dense(layer_dim, layer_dim, relu; init=init) for _ in 1:layer_num]...,
                           Dense(layer_dim, na, tanh; init=init),)

    create_critic() = Chain(Dense(ns+na, layer_dim, relu; init=init),
                            [Dense(layer_dim, layer_dim, relu; init=init) for _ in 1:layer_num]...,
                            Dense(layer_dim, 1; init=init),)

    agent = Agent(policy=DDPGPolicy(behavior_actor=NeuralNetworkApproximator(model=create_actor(), optimizer=ADAM(),),
                                    behavior_critic=NeuralNetworkApproximator(model=create_critic(), optimizer=ADAM(),),
                                    target_actor=NeuralNetworkApproximator(model=create_actor(), optimizer=ADAM(),),
                                    target_critic=NeuralNetworkApproximator(model=create_critic(), optimizer=ADAM(),),
                                    γ=0.99f0, ρ=0.995f0, na=env.ctrl_num, batch_size=64, start_steps=100,
                                    start_policy=RandomPolicy(Space([-1.0..1.0 for _ in 1:env.ctrl_num]); rng=rng),
                                    update_after=100, update_freq=1, act_limit=1.0e35,
                                    act_noise=0.01, rng=rng,),
                  trajectory=CircularArraySARTTrajectory(capacity=400, state=Vector{Float64} => (ns,), action=Vector{Float64} => (na,),),)

    println("state optimization")
    if length(params.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: deep deterministic policy gradient algorithm (DDPG)")
        println("initial CFI is $(1.0/env.f_ini)")
        append!(env.f_list, 1.0/env.f_ini)
    else
        println("multiparameter scenario")
        println("search algorithm: deep deterministic policy gradient algorithm (DDPG)")
        println("initial value of Tr(WI^{-1}) is $(env.f_ini)")
        append!(env.f_list, env.f_ini)
    end

    if env.save_file
        SaveFile_state_ddpg(env.f_list, env.reward_all, env.params.psi)
    end

    stop_condition = StopAfterStep(max_episode, is_show_progress=false)
    hook = TotalRewardPerEpisode(is_display_on_exit=false)
    run(agent, env, stop_condition, hook)

    if !env.save_file
        SaveFile_state_ddpg(env.f_list, env.reward_all, env.params.psi)
    end
    print("\e[2K")
    println("Iteration over, data saved.")
    if length(params.Hamiltonian_derivative) == 1
        println("Final CFI is ", env.f_list[end])
    else
        println("Final value of Tr(WI^{-1}) is ", env.f_list[end])
    end
end
