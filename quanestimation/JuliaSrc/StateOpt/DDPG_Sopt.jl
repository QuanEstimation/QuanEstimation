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
mutable struct ControlEnv_noiseless{T<:Complex, M<:Real, R<:AbstractRNG} <: AbstractEnv
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
    SinglePara::Bool
    save_file::Bool
    sym
    str2
    str3
end

function DDPGEnv_noiseless(Measurement, params, episode, SinglePara, save_file, rng, sym, str2, str3)
    para_num=params.Hamiltonian_derivative|>length
    state = params.psi
    state = state |> state_flatten
    ctrl_num = length(state)
    action_space = Space(fill(-1.0e35..1.0e35, length(state)))
    state_space = Space(fill(-1.0e35..1.0e35, length(state))) 

    f_ini = obj_func(Val{sym}(), params, Measurement)

    f_list = Vector{Float64}()
    reward_all = Vector{Float64}()
    env = ControlEnv_noiseless(Measurement, params, action_space, state_space, state, true, rng, 0.0, 0.0, params.tspan, ctrl_num, 
                               para_num, f_ini, f_list, reward_all, episode, SinglePara, save_file, sym, str2, str3)
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
    _step_noiseless!(env, a, Val(env.SinglePara), Val(env.save_file))
end

####################### step functions #########################
#### single parameter estimation and save_file=true ####
function _step_noiseless!(env::ControlEnv_noiseless, a, ::Val{true}, ::Val{true})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    f_current = 1.0/obj_func(Val{env.sym}(), env.params, env.Measurement)
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, f_current)
    append!(env.reward_all, env.reward)
    SaveFile_state_ddpg(f_current, env.reward, env.params.psi)
    env.episode += 1
    print("current $(env.str2) is ", f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### single parameter estimation and save_file=false ####
function _step_noiseless!(env::ControlEnv_noiseless, a, ::Val{true}, ::Val{false})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    f_current = 1.0/obj_func(Val{env.sym}(), env.params, env.Measurement)
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, f_current)
    append!(env.reward_all, env.reward)
    env.episode += 1
    print("current $(env.str2) is ", f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### multiparameter estimation and save_file=true ####
function _step_noiseless!(env::ControlEnv_noiseless, a, ::Val{false}, ::Val{true})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    f_current = 1.0/obj_func(Val{env.sym}(), env.params, env.Measurement)
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, 1.0/f_current)
    append!(env.reward_all, env.reward)
    SaveFile_state_ddpg(1.0/f_current, env.reward, env.params.psi)
    env.episode += 1
    print("current value of $(env.str3) is ", 1.0/f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### multiparameter estimation and save_file=false ####
function _step_noiseless!(env::ControlEnv_noiseless, a, ::Val{false}, ::Val{false})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    f_current = 1.0/obj_func(Val{env.sym}(), env.params, env.Measurement)
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, 1.0/f_current)
    append!(env.reward_all, env.reward)
    env.episode += 1
    print("current value of $(env.str3) is ", 1.0/f_current, " ($(env.episode) episodes)    \r")
    nothing 
end


Random.seed!(env::ControlEnv_noiseless, seed) = Random.seed!(env.rng, seed)

function QFIM_DDPG_Sopt(params::TimeIndepend_noiseless, layer_num, layer_dim, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("QFIM_TimeIndepend_noiseless")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    Measurement = [zeros(ComplexF64, size(params.psi)[1], size(params.psi)[1])]
    return info_DDPG_noiseless(Measurement, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
end

function CFIM_DDPG_Sopt(Measurement, params::TimeIndepend_noiseless, layer_num, layer_dim, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM_TimeIndepend_noiseless")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_DDPG_noiseless(Measurement, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
end

function HCRB_DDPG_Sopt(params::TimeIndepend_noiseless, layer_num, layer_dim, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("HCRB_TimeIndepend_noiseless")
    str1 = ""
    str2 = "HCRB"
    str3 = "HCRB"
    Measurement = [zeros(ComplexF64, size(params.psi)[1], size(params.psi)[1])]
    if length(params.Hamiltonian_derivative) == 1
        println("In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the objection function for state optimization.")
        return nothing
    else
        return info_DDPG_noiseless(Measurement, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
    end
end

function info_DDPG_noiseless(Measurement, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
    rng = StableRNG(seed)
    episode = 1
    if length(params.Hamiltonian_derivative) == 1
        SinglePara = true
    else
        SinglePara = false
    end
    env = DDPGEnv_noiseless(Measurement, params, episode, SinglePara, save_file, rng, sym, str1, str2)
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

    println("$str1 state optimization")
    if length(params.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: deep deterministic policy gradient algorithm (DDPG)")
        println("initial $str2 is $(1.0/env.f_ini)")
        append!(env.f_list, 1.0/env.f_ini)
    else
        println("multiparameter scenario")
        println("search algorithm: deep deterministic policy gradient algorithm (DDPG)")
        println("initial value of $str3 is $(env.f_ini)")
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
        println("Final $str2 is ", env.f_list[end])
    else
        println("Final value of $str3 is ", env.f_list[end])
    end
end

############# time-independent Hamiltonian (noise) ################
mutable struct ControlEnv_noise{T<:Complex, M<:Real, R<:AbstractRNG} <: AbstractEnv
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
    SinglePara::Bool
    save_file::Bool
    sym
    str2
    str3
end

function DDPGEnv_noise(Measurement, params, episode, SinglePara, save_file, rng, sym, str2, str3)
    para_num=params.Hamiltonian_derivative|>length
    state = params.psi
    state = state |> state_flatten
    ctrl_num = length(state)
    action_space = Space(fill(-1.0e35..1.0e35, length(state)))
    state_space = Space(fill(-1.0e35..1.0e35, length(state))) 

    f_ini = obj_func(Val{sym}(), params, Measurement)

    f_list = Vector{Float64}()
    reward_all = Vector{Float64}()
    env = ControlEnv_noise(Measurement, params, action_space, state_space, state, true, rng, 0.0, 0.0, params.tspan, ctrl_num, 
                           para_num, f_ini, f_list, reward_all, episode, SinglePara, save_file, sym, str2, str3)
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
    _step_noise!(env, a, Val(env.SinglePara), Val(env.save_file))
end

####################### step functions #########################
#### single parameter estimation and save_file=true ####
function _step_noise!(env::ControlEnv_noise, a, ::Val{true}, ::Val{true})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    f_current = 1.0/obj_func(Val{env.sym}(), env.params, env.Measurement)
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, f_current)
    append!(env.reward_all, env.reward)
    SaveFile_state_ddpg(f_current, env.reward, env.params.psi)
    env.episode += 1
    print("current $(env.str2) is ", f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### single parameter estimation and save_file=false ####
function _step_noise!(env::ControlEnv_noise, a, ::Val{true}, ::Val{false})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    f_current = 1.0/obj_func(Val{env.sym}(), env.params, env.Measurement)
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, f_current)
    append!(env.reward_all, env.reward)
    env.episode += 1
    print("current $(env.str2) is ", f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### multiparameter estimation and save_file=true ####
function _step_noise!(env::ControlEnv_noise, a, ::Val{false}, ::Val{true})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    f_current = 1.0/obj_func(Val{env.sym}(), env.params, env.Measurement)
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, 1.0/f_current)
    append!(env.reward_all, env.reward)
    SaveFile_state_ddpg(1.0/f_current, env.reward, env.params.psi)
    env.episode += 1
    print("current value of $(env.str3) is ", 1.0/f_current, " ($(env.episode) episodes)    \r")
    nothing 
end

#### multiparameter estimation and save_file=false ####
function _step_noise!(env::ControlEnv_noise, a, ::Val{false}, ::Val{false})
    state_new = (env.state + a) |> to_psi
    env.params.psi = state_new/norm(state_new)

    f_current = 1.0/obj_func(Val{env.sym}(), env.params, env.Measurement)
    env.reward = -log(f_current/env.f_ini)
    env.total_reward = env.reward
    env.done = true

    append!(env.f_list, 1.0/f_current)
    append!(env.reward_all, env.reward)
    env.episode += 1
    print("current value of $(env.str3) is ", 1.0/f_current, " ($(env.episode) episodes)    \r")
    nothing 
end


Random.seed!(env::ControlEnv_noise, seed) = Random.seed!(env.rng, seed)

function QFIM_DDPG_Sopt(params::TimeIndepend_noise, layer_num, layer_dim, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("HCRB_TimeIndepend_noise")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    Measurement = [zeros(ComplexF64, size(params.psi)[1], size(params.psi)[1])]
    return info_DDPG_noise(Measurement, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
end

function CFIM_DDPG_Sopt(Measurement, params::TimeIndepend_noise, layer_num, layer_dim, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("HCRB_TimeIndepend_noise")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_DDPG_noise(Measurement, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
end

function HCRB_DDPG_Sopt(params::TimeIndepend_noise, layer_num, layer_dim, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("HCRB_TimeIndepend_noise")
    str1 = ""
    str2 = "HCRB"
    str3 = "HCRB"
    Measurement = [zeros(ComplexF64, size(params.psi)[1], size(params.psi)[1])]
    if length(params.Hamiltonian_derivative) == 1
        println("In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the objection function for state optimization.")
        return nothing
    else
        return info_DDPG_noise(Measurement, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
    end
end

function info_DDPG_noise(Measurement, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
    rng = StableRNG(seed)
    episode = 1
    if length(params.Hamiltonian_derivative) == 1
        SinglePara = true
    else
        SinglePara = false
    end
    env = DDPGEnv_noise(Measurement, params, episode, SinglePara, save_file, rng, sym, str2, str3)
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

    println("$str1 state optimization")
    if length(params.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("search algorithm: deep deterministic policy gradient algorithm (DDPG)")
        println("initial $str2 is $(1.0/env.f_ini)")
        append!(env.f_list, 1.0/env.f_ini)
    else
        println("multiparameter scenario")
        println("search algorithm: deep deterministic policy gradient algorithm (DDPG)")
        println("initial value of $str3 is $(env.f_ini)")
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
        println("Final $str2 is ", env.f_list[end])
    else
        println("Final value of $str3 is ", env.f_list[end])
    end
end
