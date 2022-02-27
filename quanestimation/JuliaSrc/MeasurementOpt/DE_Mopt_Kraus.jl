function CFIM_DE_Mopt(DE::projection_Mopt_Kraus{T}, popsize, ini_population, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM_noctrl_Kraus")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_DE_projection(DE, popsize, ini_population, c, cr, seed, max_episode, save_file, sym, str1, str2)
end


function CFIM_DE_Mopt(DE::LinearComb_Mopt_Kraus{T}, popsize, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM_noctrl_Kraus")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_DE_LinearComb(DE, popsize, c, cr, seed, max_episode, save_file, sym, str1, str2)
end

function CFIM_DE_Mopt(DE::RotateBasis_Mopt_Kraus{T}, popsize, c, cr, seed, max_episode, save_file) where {T<:Complex}
    sym = Symbol("CFIM_noctrl_Kraus")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_DE_RotateBasis(DE, popsize, c, cr, seed, max_episode, save_file, sym, str1, str2)
end
