
function QFIM_DDPG_Sopt(params::TimeIndepend_Kraus, layer_num, layer_dim, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("QFIM_TimeIndepend_Kraus")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    M = [zeros(ComplexF64, size(params.psi)[1], size(params.psi)[1])]
    return info_DDPG_noiseless(M, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
end

function CFIM_DDPG_Sopt(M, params::TimeIndepend_Kraus, layer_num, layer_dim, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM_TimeIndepend_Kraus")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_DDPG_noiseless(M, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
end

function HCRB_DDPG_Sopt(params::TimeIndepend_Kraus, layer_num, layer_dim, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("HCRB_TimeIndepend_Kraus")
    str1 = ""
    str2 = "HCRB"
    str3 = "HCRB"
    M = [zeros(ComplexF64, size(params.psi)[1], size(params.psi)[1])]
    if length(params.Hamiltonian_derivative) == 1
        println("In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the objection function for state optimization.")
        return nothing
    else
        return info_DDPG_noiseless(M, params, layer_num, layer_dim, seed, max_episode, save_file, sym, str1, str2, str3)
    end
end
