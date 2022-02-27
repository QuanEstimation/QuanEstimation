
############# time-independent Hamiltonian (noiseless Kraus rep.) ################
function QFIM_PSO_Sopt(pso::TimeIndepend_Kraus{T}, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file) where {T<:Complex}
    sym = Symbol("QFIM_TimeIndepend_Kraus")
    str1 = "quantum"
    str2 = "QFI"
    str3 = "tr(WF^{-1})"
    M = [zeros(ComplexF64, size(pso.psi)[1], size(pso.psi)[1])]
    return info_PSO_noiseless(M, pso, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file, sym, str1, str2, str3)
end

function CFIM_PSO_Sopt(M, pso::TimeIndepend_Kraus{T}, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file) where {T<:Complex}
    sym = Symbol("CFIM_TimeIndepend_Kraus")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    return info_PSO_noiseless(M, pso, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file, sym, str1, str2, str3)
end

function HCRB_PSO_Sopt(pso::TimeIndepend_Kraus{T}, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file) where {T<:Complex}
    sym = Symbol("HCRB_TimeIndepend_Kraus")
    str1 = ""
    str2 = "HCRB"
    str3 = "HCRB"
    M = [zeros(ComplexF64, size(pso.psi)[1], size(pso.psi)[1])]
    if length(pso.Hamiltonian_derivative) == 1
        println("In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the objection function for state optimization.")
        return nothing
    else
        return info_PSO_noiseless(M, pso, max_episode, particle_num, ini_particle, c0, c1, c2, v0, seed, save_file, sym, str1, str2, str3)
    end
end
