
function gradient_CFI!(AD::LinearComb_Mopt_Kraus{T}, epsilon, B, POVM_basis, M_num, basis_num) where {T<:Complex}
    δI = gradient(x->CFI(AD.K, AD.dK, AD.ρ0,[sum([x[i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num], eps=AD.eps), B)[1]
    B += epsilon*δI
    B = bound_LC_coeff(B)
    return B
end


function gradient_CFI_Adam!(AD::LinearComb_Mopt_Kraus{T}, epsilon, B, POVM_basis, M_num, basis_num) where {T<:Complex}
    K, dK, ρ0 = AD.K, AD.dK, AD.ρ0
    ρt, ∂ρt_∂x = K*ρ0*K', dK*ρ0*K' + K*ρ0*dK'
    δI = gradient(x->CFI(ρt, ∂ρt_∂x, [sum([x[i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num], eps=AD.eps), B)[1]
    B = MOpt_Adam!(B, δI, epsilon, mt, vt, beta1, beta2, AD.eps)
    B = bound_LC_coeff(B)
    return B
end

function gradient_CFIM!(AD::LinearComb_Mopt_Kraus{T}, epsilon, B, POVM_basis, M_num, basis_num) where {T<:Complex}
    K, dK, ρ0 = AD.K, AD.dK, AD.ρ0
    ρt, ∂ρt_∂x = K*ρ0*K', [dK*ρ0*K' + K*ρ0*dK' for dK in dK]
    δI = gradient(x->1/(AD.W*pinv(CFIM(ρt, ∂ρt_∂x, [sum([x[i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num], AD.eps), rtol=AD.eps) |> tr |>real), B) |>sum
    B += epsilon*δI
    B = bound_LC_coeff(B)
    return B
end

function gradient_CFIM_Adam!(AD::LinearComb_Mopt_Kraus{T}, epsilon, B, POVM_basis, M_num, basis_num) where {T<:Complex}
    K, dK, ρ0 = AD.K, AD.dK, AD.ρ0
    ρt, ∂ρt_∂x = K*ρ0*K', [dK*ρ0*K' + K*ρ0*dK' for dK in dK]
    δI = gradient(x->1/(AD.W*pinv(CFIM(ρt, ∂ρt_∂x, [sum([x[i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num], AD.eps), rtol=AD.eps) |> tr |>real), B) |>sum
    B = MOpt_Adam!(B, δI, epsilon, mt, vt, beta1, beta2, AD.eps)
    B = bound_LC_coeff(B)
    return B
end

function gradient_CFI!(AD::RotateBasis_Mopt_Kraus{T}, epsilon, Mbasis, s, Lambda, M_num, dim) where {T<:Complex}
    δI = gradient(x->CFI_AD(Mbasis, x, Lambda, AD.K, AD.dK[1], AD.ρ0, AD.eps), s)[1]
    s += epsilon*δI
    s = bound_rot_coeff(s)
    return s
end

function gradient_CFI!(AD::RotateBasis_Mopt_Kraus{T}, epsilon, Mbasis, s, Lambda, M_num, dim) where {T<:Complex}
    δI = gradient(x->CFI_AD(Mbasis, x, Lambda, AD.K, AD.dK[1], AD.ρ0, AD.eps), s)[1]
    s = MOpt_Adam!(s, δI, epsilon, mt, vt, beta1, beta2, AD.eps)
    s = bound_rot_coeff(s)
    return s
end

function gradient_CFIM!(AD::RotateBasis_Mopt_Kraus{T}, epsilon, Mbasis, s, Lambda, M_num, dim) where {T<:Complex}
    δI = gradient(x->1/(AD.W*pinv(CFIM_AD(Mbasis, x, Lambda, AD.K, AD.dK, AD.ρ0, AD.eps), rtol=AD.eps) |> tr |>real), s) |>sum
    s += epsilon*δI
    s = bound_rot_coeff(s)
    return s
end


function gradient_CFIM!(AD::RotateBasis_Mopt_Kraus{T}, epsilon, Mbasis, s, Lambda, M_num, dim) where {T<:Complex}
    δI = gradient(x->1/(AD.W*pinv(CFIM_AD(Mbasis, x, Lambda, AD.K, AD.dK, AD.ρ0, AD.eps), rtol=AD.eps) |> tr |>real), s) |>sum
    s = MOpt_Adam!(s, δI, epsilon, mt, vt, beta1, beta2, AD.eps)
    s = bound_rot_coeff(s)
    return s
end


