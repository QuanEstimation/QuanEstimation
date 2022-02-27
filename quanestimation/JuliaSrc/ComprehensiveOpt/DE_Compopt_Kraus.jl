
function SM_DE_Compopt(DE::SM_Compopt_Kraus{T}, popsize, psi0, measurement0, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM_SMopt_Kraus")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    M = [zeros(ComplexF64, size(DE.psi)[1], size(DE.psi)[1])]
    return info_DE_SMopt(M, DE, popsize, psi0, measurement0, c, cr, seed, max_episode, save_file, sym, str1, str2)
end

