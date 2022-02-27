
function SM_PSO_Compopt(pso::SM_Compopt_Kraus{T}, max_episode, particle_num, psi0, measurement0, c0, c1, c2, seed, save_file) where {T<: Complex}
    sym = Symbol("CFIM_SMopt_Kraus")
    str1 = "classical"
    str2 = "CFI"
    str3 = "tr(WI^{-1})"
    M = [zeros(ComplexF64, size(pso.psi)[1], size(pso.psi)[1])]
    return info_PSO_SMopt(M, pso, max_episode, particle_num, psi0, measurement0, c0, c1, c2, seed, save_file, sym, str1, str2, str3)
end
