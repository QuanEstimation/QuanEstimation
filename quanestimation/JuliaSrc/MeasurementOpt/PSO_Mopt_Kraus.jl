
function CFIM_PSO_Mopt(pso::projection_Mopt_Kraus{T}, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file) where {T<:Complex}
    sym = Symbol("CFIM_noctrl_Kraus")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_PSO_projection(pso, max_episode, particle_num, ini_particle, c0, c1, c2, seed, save_file, sym, str1, str2)
end

function CFIM_PSO_Mopt(pso::LinearComb_Mopt_Kraus{T}, max_episode, particle_num, c0, c1, c2, seed, save_file) where {T<:Complex}
    sym = Symbol("CFIM_noctrl_Kraus")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_PSO_LinearComb(pso, max_episode, particle_num, c0, c1, c2, seed, save_file, sym, str1, str2)
end


function CFIM_PSO_Mopt(pso::RotateBasis_Mopt_Kraus{T}, max_episode, particle_num, c0, c1, c2, seed, save_file) where {T<:Complex}
    sym = Symbol("CFIM_noctrl_Kraus")
    str1 = "CFI"
    str2 = "tr(WI^{-1})"
    return info_PSO_RotateBasis(pso, max_episode, particle_num, c0, c1, c2, seed, save_file, sym, str1, str2)
end
