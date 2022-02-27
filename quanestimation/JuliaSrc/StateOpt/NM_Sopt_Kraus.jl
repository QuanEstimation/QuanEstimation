
############# time-independent Hamiltonian (noiseless Kraus rep.) ################
function QFIM_NM_Sopt(NM::TimeIndepend_Kraus{T}, state_num, ini_state, ar, ae, ac, as0, max_episode, seed, save_file) where {T<:Complex}
    sym = Symbol("QFIM_TimeIndepend_Kraus")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    M = [zeros(ComplexF64, size(NM.psi)[1], size(NM.psi)[1])]
    return info_NM_noiseless(M, NM, state_num, ini_state, ar, ae, ac, as0, max_episode, seed, save_file, sym, str1, str2, str3)
end

function CFIM_NM_Sopt(M, NM::TimeIndepend_Kraus{T}, state_num, ini_state, ar, ae, ac, as0, max_episode, seed, save_file) where {T<:Complex}
    sym = Symbol("CFIM_TimeIndepend_Kraus")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_NM_noiseless(M, NM, state_num, ini_state, ar, ae, ac, as0, max_episode, seed, save_file, sym, str1, str2, str3)
end

function HCRB_NM_Sopt(NM::TimeIndepend_Kraus{T}, state_num, ini_state, ar, ae, ac, as0, max_episode, seed, save_file) where {T<:Complex}
    sym = Symbol("HCRB_TimeIndepend_Kraus")
    str1 = ""
    str2 = "HCRB"
    str3 = "HCRB"
    M = [zeros(ComplexF64, size(NM.psi)[1], size(NM.psi)[1])]
    if length(NM.Hamiltonian_derivative) == 1
        println("In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the objection function for state optimization.")
        return nothing
    else
        return info_NM_noiseless(M, NM, state_num, ini_state, ar, ae, ac, as0, max_episode, seed, save_file, sym, str1, str2, str3)
    end
end
