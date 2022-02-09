mutable struct adaptive{T<:Complex, M <:Real}
    phase::M
    Jy::Matrix{T}
    ρ0::Matrix{T}
    Meas::Vector{Matrix{T}}
    W::Matrix{M}
    accuracy::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    adaptive(phase::M, beam_splitter::Matrix{T}, phaseShift_Mat::Matrix{T}, ρ0::Matrix{T}, 
    Meas::Vector{Matrix{T}}, W::Matrix{M}, accuracy::M, 
    ρ=Vector{Matrix{T}}(undef, 1), ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T<:Complex, M<:Real}=
    new{T,M}(phase, beam_splitter, phaseShift_Mat, ρ0, Meas, W, accuracy, ρ, ∂ρ_∂x) 
end

function update_distribution(rho, drho, p_in, dp_in, M)
    M_num = length(M)
    para_num = length(drho)
    p = zeros(M_num)
    dp = [zeros(M_num) for i in 1:para_num]
    for i in 1:M_num
        p[i] = real(tr(rho*M[i]))
        for pj in 1:para_num
            drho_i = drho[pj]
            dp[pj][i] = real(tr(drho_i*M[i]))
        end
    end

    py = sum([p[j]*p_in[j] for j in 1:M_num])
    p_now = [p[k]*p_in[k]/py for k in 1:M_num]
    dp_now = [zeros(M_num) for i in 1:para_num]
    for pk in 1:para_num
        term1 = [dp[pk][l]*p_in[l]+p[l]*dp_in[pk][l] for l in 1:M_num]
        term_tp = sum([dp[pk][m]*p_in[m]+p[m]*dp_in[pk][m] for m in 1:M_num])
        term2 = [p[n]*p_in[n]*term_tp for n in 1:M_num]
        dp_now[pk] = term1/py+term2/py^2 
    end

    return p_now, dp_now
end

function adaptive_Mopt(ada, p_in, dp_in, x, u, delta)
    phase = delta + (-1)^u*ada.phase
    phase_shift = exp(-1.0im*phase*ada.Jy)
    rho = phase_shift*ada.ρ0*phase_shift'
    drho = -1.0im*ada.Jy*rho+1.0im*rho*ada.Jy
    f = BQCRB(rho, drho, p_in, dp_in, x)
    p_in, dp_in = update_distribution(rho, drho, p_in, dp_in, ada.Meas) 
    return f, p_in, dp_in
end
