using Flux: glorot_normal, glorot_uniform
using StableRNGs
using Flux
using Flux.Losses
using IntervalSets

mutable struct DDPG_Copt{T<:Complex, M<:Real}
    freeHamiltonian
    Hamiltonian_derivative::Vector{Matrix{T}}
    ρ0::Matrix{T}
    tspan::Vector{M}
    decay_opt::Vector{Matrix{T}}
    γ::Vector{M}
    control_Hamiltonian::Vector{Matrix{T}}
    control_coefficients::Vector{Vector{M}}
    ctrl_bound::Vector{M}
    W::Matrix{M}
    ctrl_interval::Int
    dim::Int
    eps::M
end

mutable struct ControlEnv{T<:Complex, M<:Real, R<:AbstractRNG} <:AbstractEnv
    M::Vector{Matrix{T}}
    params::DDPG_Copt{T, M}
    action_space::Space
    state_space::Space
    state::Vector{M}
    dstate::Vector{Matrix{T}}
    done::Bool
    rng::R
    reward::Float64
    total_reward::Float64
    t::Int
    tspan::Vector{M}
    tnum::Int
    cnum::Int
    ctrl_num::Int
    para_num::Int
    f_noctrl::Vector{M}
    f_final::Vector{M}
    ctrl_list::Vector{Vector{M}}
    ctrl_bound::Vector{M}
    total_reward_all::Vector{M}
    episode::Int
    SinglePara::Bool
    save_file::Bool
    sym
    str2
    str3
end

function DDPGEnv(M, params, episode, SinglePara, save_file, rng, sym, str2, str3)
    para_num=params.Hamiltonian_derivative|>length
    ctrl_num=params.control_coefficients|>length
    tnum = params.tspan|>length
    cnum = params.control_coefficients[1] |> length
    state = params.ρ0
    dstate = [state|>zero for _ in 1:para_num]
    state = state|>state_flatten
    action_space = Space([params.ctrl_bound[1]..params.ctrl_bound[2] for _ in 1:ctrl_num])
    state_space = Space(fill(-1.0e35..1.0e35, length(state))) 

    f_noctrl = F_noctrl(M, params, para_num, cnum, ctrl_num, sym)

    ctrl_list = [Vector{Float64}() for _ in 1:ctrl_num]
    f_final = Vector{Float64}()
    total_reward_all = Vector{Float64}()
    env = ControlEnv(M, params, action_space, state_space, state, dstate, true, rng, 0.0, 0.0, 0, params.tspan, tnum, 
                     cnum, ctrl_num, para_num, f_noctrl, f_final, ctrl_list, params.ctrl_bound, total_reward_all, episode, 
                     SinglePara, save_file, sym, str2, str3)
    reset!(env)
    env
end

function F_noctrl(M, params, para_num, cnum, ctrl_num, sym)
    rho = params.ρ0
    drho = [rho|>zero for _ in 1:para_num]
    f_noctrl = zeros(cnum+1)
    for i in 2:cnum+1
        rho, drho = propagate(rho, drho, params, [0.0 for i in 1:ctrl_num])
        f_noctrl[i] = 1.0/obj_func(Val{sym}(), rho, drho, params.W, M, params.eps)
    end
    f_noctrl
end

function Base.rsplit( v, l::Int)
    u = reshape(v,l,length(v)÷l)
    [u[:,i] for i=1:size(u,2)]
end

state_flatten(s) = vcat((s|>reim.|>vec)...)
rsplit_half(v) = Base.rsplit(v, length(v)÷2)
density_matrix(s) = complex.(rsplit_half(s)...)|>vec2mat

function RLBase.reset!(env::ControlEnv)
    state = env.params.ρ0
    env.dstate = [state|>zero for _ in 1:(env.para_num)]
    env.state = state|>state_flatten
    env.t = 1
    env.done = false
    env.reward = 0.0
    env.total_reward = 0.0
    env.ctrl_list = [Vector{Float64}() for _ in 1:env.ctrl_num]
    nothing
end

RLBase.action_space(env::ControlEnv) = env.action_space
RLBase.state_space(env::ControlEnv) = env.state_space
RLBase.reward(env::ControlEnv) = env.reward
RLBase.is_terminated(env::ControlEnv) = env.done 
RLBase.state(env::ControlEnv) = env.state

function (env::ControlEnv)(a)
    # @assert a in env.action_space
    bound!(a, env.ctrl_bound)
    _step!(env, a, Val(env.SinglePara), Val(env.save_file))
end

####################### step functions #########################
#### single parameter estimation and save_file=true ####
function _step!(env::ControlEnv, a, ::Val{true}, ::Val{true})
    env.t += 1
    ρₜ, ∂ₓρₜ = env.state|>density_matrix, env.dstate
    ρₜₙ, ∂ₓρₜₙ = propagate(ρₜ, ∂ₓρₜ, env.params, a, env.t)
    env.state = ρₜₙ |> state_flatten
    env.dstate = ∂ₓρₜₙ
    env.done = env.t > env.cnum
    f_current = 1.0/obj_func(Val{env.sym}(), ρₜₙ, ∂ₓρₜₙ, env.params.W, env.M, env.params.eps)
    reward_current = log(f_current/env.f_noctrl[env.t])
    env.reward = reward_current
    env.total_reward += reward_current
    [append!(env.ctrl_list[i], a[i]) for i in 1:length(a)]
    if env.done 
        append!(env.f_final, f_current)
        append!(env.total_reward_all, env.total_reward)
        SaveFile_ddpg(f_current, env.total_reward, env.ctrl_list)
        env.episode += 1
        print("current $(env.str2) is ", f_current, " ($(env.episode) episodes)    \r")
    end
    nothing 
end

#### single parameter estimation and save_file=false ####
function _step!(env::ControlEnv, a, ::Val{true}, ::Val{false})
    env.t += 1
    ρₜ, ∂ₓρₜ = env.state|>density_matrix, env.dstate
    ρₜₙ, ∂ₓρₜₙ = propagate(ρₜ, ∂ₓρₜ, env.params, a, env.t)
    env.state = ρₜₙ|>state_flatten
    env.dstate = ∂ₓρₜₙ
    env.done = env.t > env.cnum
    f_current = 1.0/obj_func(Val{env.sym}(), ρₜₙ, ∂ₓρₜₙ, env.params.W, env.M, env.params.eps)
    reward_current = log(f_current/env.f_noctrl[env.t])
    env.reward = reward_current
    env.total_reward += reward_current
    [append!(env.ctrl_list[i], a[i]) for i in 1:length(a)]
    if env.done
        append!(env.f_final, f_current)
        append!(env.total_reward_all, env.total_reward)
        env.episode += 1
        print("current $(env.str2) is ", f_current, " ($(env.episode) episodes)    \r")
    end
    nothing 
end

#### multiparameter estimation and save_file=true ####
function _step!(env::ControlEnv, a, ::Val{false}, ::Val{true})
    env.t += 1
    ρₜ, ∂ₓρₜ = env.state|>density_matrix, env.dstate
    ρₜₙ, ∂ₓρₜₙ = propagate(ρₜ, ∂ₓρₜ, env.params, a, env.t)
    env.state = ρₜₙ|>state_flatten
    env.dstate = ∂ₓρₜₙ
    env.done = env.t > env.cnum
    f_current = 1.0/obj_func(Val{env.sym}(), ρₜₙ, ∂ₓρₜₙ, env.params.W, env.M, env.params.eps)
    reward_current = log(f_current/env.f_noctrl[env.t])
    env.reward = reward_current
    env.total_reward += reward_current
    [append!(env.ctrl_list[i], a[i]) for i in 1:length(a)]
    if env.done
        append!(env.f_final, 1.0/f_current)
        append!(env.total_reward_all, env.total_reward)
        SaveFile_ddpg(1.0/f_current, env.total_reward, env.ctrl_list)
        env.episode += 1
        print("current value of $(env.str3) is ", 1.0/f_current, " ($(env.episode) episodes)    \r")
    end
    nothing 
end

#### multiparameter estimation and save_file=false ####
function _step!(env::ControlEnv, a, ::Val{false}, ::Val{false})
    env.t += 1
    ρₜ, ∂ₓρₜ = env.state|>density_matrix, env.dstate
    ρₜₙ, ∂ₓρₜₙ = propagate(ρₜ, ∂ₓρₜ, env.params, a, env.t)
    env.state = ρₜₙ|>state_flatten
    env.dstate = ∂ₓρₜₙ
    env.done = env.t > env.cnum
    f_current = 1.0/obj_func(Val{env.sym}(), ρₜₙ, ∂ₓρₜₙ, env.params.W, env.M, env.params.eps)
    reward_current = log(f_current/env.f_noctrl[env.t])
    env.reward = reward_current
    env.total_reward += reward_current
    [append!(env.ctrl_list[i], a[i]) for i in 1:length(a)]
    if env.done
        append!(env.f_final, 1.0/f_current)
        append!(env.total_reward_all, env.total_reward)
        env.episode += 1
        print("current value of $(env.str3) is ", 1.0/f_current, " ($(env.episode) episodes)    \r")
    end
    nothing 
end

Random.seed!(env::ControlEnv, seed) = Random.seed!(env.rng, seed)

function QFIM_DDPG_Copt(params::DDPG_Copt, layer_num, layer_dim, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("QFIM")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    M = [zeros(ComplexF64, size(params.ρ0)[1], size(params.ρ0)[1])]
    return info_DDPG_Copt(M, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
end

function CFIM_DDPG_Copt(M, params::DDPG_Copt, layer_num, layer_dim, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_DDPG_Copt(M, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
end

CFIM_DDPG_Copt(params::DDPG_Copt, M, layer_num, layer_dim, seed, max_episode, save_file)= CFIM_DDPG_Copt(M, params, layer_num, layer_dim, seed, max_episode, save_file)

function HCRB_DDPG_Copt(params::DDPG_Copt, layer_num, layer_dim, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("HCRB")
    str1 = ""
    str2 = "HCRB"
    str3 = "HCRB"
    M = [zeros(ComplexF64, size(params.ρ0)[1], size(params.ρ0)[1])]
    if length(params.Hamiltonian_derivative) == 1
        @warn "In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the objection function for control optimization."
        exit()
    else
        return info_DDPG_Copt(M, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
    end
end

function info_DDPG_Copt(M, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
    rng = StableRNG(seed)
    episode = 1
    if length(params.Hamiltonian_derivative) == 1
        SinglePara = true
    else
        SinglePara = false
    end
    env = DDPGEnv(M, params, episode, SinglePara, save_file, rng, sym, str2, str3)

    ns = 2*params.dim^2
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
                                    γ=0.99f0, ρ=0.995f0, na=env.ctrl_num, batch_size=128, start_steps=100*env.cnum,
                                    start_policy=RandomPolicy(Space([-10.0..10.0 for _ in 1:env.ctrl_num]); rng=rng),
                                    update_after=100*env.cnum, update_freq=1*env.cnum, act_limit=env.params.ctrl_bound[end],
                                    act_noise=0.01, rng=rng,),
                  trajectory=CircularArraySARTTrajectory(capacity=400*env.cnum, state=Vector{Float64} => (ns,), action=Vector{Float64} => (na,),),)

    println("$str1 parameter estimation")
    f_ini = obj_func(Val{sym}(), params, M)
    if length(params.Hamiltonian_derivative) == 1
        println("single parameter scenario")
        println("control algorithm: Deep Deterministic Policy Gradient algorithm (DDPG)")
        println("non-controlled $str2 is $(env.f_noctrl[end])")
        println("initial $str2 is $(1.0/f_ini)")
        append!(env.f_final, 1.0/f_ini)
    else
        println("multiparameter scenario")
        println("control algorithm: Deep Deterministic Policy Gradient algorithm (DDPG)")
        println("non-controlled value of $str3 is $(1.0/env.f_noctrl[end])")
        println("initial value of $str3 is $(f_ini)")
        append!(env.f_final, f_ini)
    end

    if env.save_file
        SaveFile_ddpg(env.f_final, env.total_reward_all, params.control_coefficients)
    end

    stop_condition = StopAfterStep(max_episode*env.cnum, is_show_progress=false)
    hook = TotalRewardPerEpisode(is_display_on_exit=false)
    run(agent, env, stop_condition, hook)

    if !env.save_file
        SaveFile_ddpg(env.f_final, env.total_reward_all, env.ctrl_list)
    end
    print("\e[2K")
    println("Iteration over, data saved.")
    if length(params.Hamiltonian_derivative) == 1
        println("Final $str2 is ", env.f_final[end])
        return env.f_final[end], env.ctrl_list
    else
        println("Final value of $str3 is ", env.f_final[end])
        return 1/env.f_final[end], env.ctrl_list
    end
    
end
