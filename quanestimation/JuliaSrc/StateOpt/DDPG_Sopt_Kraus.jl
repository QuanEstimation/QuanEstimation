
function QFIM_DDPG_Sopt(params::TimeIndepend_Kraus, layer_num, layer_dim, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("QFIM_TimeIndepend_Kraus")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    M = [zeros(ComplexF64, size(params.psi)[1], size(params.psi)[1])]
    return info_DDPG_Kraus(M, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
end

function CFIM_DDPG_Sopt(M, params::TimeIndepend_Kraus, layer_num, layer_dim, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM_TimeIndepend_Kraus")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_DDPG_Kraus(M, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
end

function HCRB_DDPG_Sopt(params::TimeIndepend_Kraus, layer_num, layer_dim, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("HCRB_TimeIndepend_Kraus")
    str1 = ""
    str2 = "HCRB"
    str3 = "HCRB"
    M = [zeros(ComplexF64, size(params.psi)[1], size(params.psi)[1])]
    if length(params.dK) == 1
        println("In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the objection function for state optimization.")
        return nothing
    else
        return info_DDPG_Kraus(M, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
    end
end

function info_DDPG_Kraus(M, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
    rng = StableRNG(seed)
    episode = 1
    if length(typeof(params)==TimeIndepend_noiseless ? params.Hamiltonian_derivative : params.dK) == 1
        SinglePara = true
    else
        SinglePara = false
    end
    env = DDPGEnv_noiseless(M, params, episode, SinglePara, save_file, rng, sym, str1, str2)
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
    if length(typeof(params)==TimeIndepend_noiseless ? params.Hamiltonian_derivative : params.dK) == 1
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
    if length(params.dK) == 1
        println("Final $str2 is ", env.f_list[end])
    else
        println("Final value of $str3 is ", env.f_list[end])
    end
end
