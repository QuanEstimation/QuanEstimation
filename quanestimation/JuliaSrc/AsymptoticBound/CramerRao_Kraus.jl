# QFI for pure state with kraus rep.
function QFIM_TimeIndepend(K::AbstractMatrix, dK::AbstractMatrix, psi0::AbstractVector, eps::Number)
    ρt, ∂ρt_∂x = K*psi0*psi0'*K', [dK*psi0*psi0'*K' + K*psi0*psi0'*dK' for dK in dK]
    QFIM_pure(ρt, ∂ρt_∂x, eps)
end

# QFI for rho0 with kraus rep.
function QFIM_TimeIndepend(K::AbstractMatrix, dK::AbstractMatrix, ρ0::AbstractMatrix, eps::Number)
    ρt, ∂ρt_∂x = K*ρ0*K', [dK*ρ0*K' + K*ρ0*dK' for dK in dK]
    QFIM_pure(ρt, ∂ρt_∂x, eps)
end

# QFIM for pure state with kraus rep.
function QFIM_TimeIndepend(K::AbstractMatrix, dK::AbstractVector, psi0::AbstractVector)
    ρt, ∂ρt_∂x = K*psi0*psi0'*K', [dK*psi0*psi0'*K' + K*psi0*psi0'*dK' for dK in dK]
    QFIM_pure(ρt, ∂ρt_∂x)
end

# QFIM for ρ0 with kraus rep.
function QFIM_TimeIndepend(K::AbstractMatrix, dK::AbstractVector, ρ0::AbstractMatrix, eps::Number)
    ρt, ∂ρt_∂x = K*ρ0*K', [dK*ρ0*K' + K*ρ0*dK' for dK in dK]
    QFIM(ρt, ∂ρt_∂x, eps)
end

function obj_func(x::Val{:QFIM_TimeIndepend_Kraus}, system, M)
    F = QFIM_TimeIndepend(system.K, system.dK, system.psi, system.eps)
    return (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:QFIM_TimeIndepend_Kraus}, system, M, psi)
    F = QFIM_TimeIndepend(system.K, system.dK, psi, system.eps)
    return (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function CFI_AD_Kraus(Mbasis::Vector{Vector{T}}, Mcoeff::Vector{R}, Lambda, K, dK, ρ0::Matrix{T}, eps::Number) where {T<:Complex,R<:Real}
    ρt, ∂ρt_∂x = K*ρ0*K', dK*ρ0*K' + K*ρ0*dK'
    U = rotation_matrix(Mcoeff, Lambda)
    M = [U*Mbasis[i]*Mbasis[i]'*U' for i in 1:length(Mbasis)]

    CFI(ρt, ∂ρt_∂x, M, eps)
end

# CFI for pure state with kraus rep.
function CFIM_TimeIndepend(M::AbstractVector, K::AbstractMatrix, dK::AbstractMatrix,psi0::AbstractVector,eps::Number)
    ρt, ∂ρt_∂x =  K*psi0*psi0'*K', [dK*psi0*psi0'*K' + K*psi0*psi0'*dK' for dK in dK]
    CFI(ρt, ∂ρt_∂x, M, eps)
end

function CFIM_TimeIndepend(M::AbstractVector, K::AbstractMatrix, dK::AbstractVector,psi0::AbstractVector,eps::Number)
    ρt, ∂ρt_∂x =  K*psi0*psi0'*K', [dK*psi0*psi0'*K' + K*psi0*psi0'*dK' for dK in dK]
    CFIM(ρt, ∂ρt_∂x, M, eps)
end

function obj_func(x::Val{:CFIM_TimeIndepend_Kraus}, system, M)
    F = CFIM_TimeIndepend(M, system.K, system.dK, system.psi, system.eps)
    return (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:CFIM_TimeIndepend_Kraus}, system, M, psi)
    F = CFIM_TimeIndepend(M, system.K, system.dK, psi, system.eps)
    return (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

function obj_func(x::Val{:CFIM_noctrl_Kraus}, system, M)
    ρ0 = system.psi*system.psi'
    K, dK, ρ0 = system.K, system.dK, ρ0
    ρt, ∂ρt_∂x = K*ρ0*K', [dK*ρ0*K' + K*ρ0*dK' for dK in dK]
    F = CFIM(ρt, ∂ρt_∂x, M, system.eps)
    return  (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end


function obj_func(x::Val{:CFIM_SMopt_Kraus}, system, psi, M)
    ρ0 = psi*psi'
    K, dK, ρ0 = system.K, system.dK, ρ0
    ρt, ∂ρt_∂x = K*ρ0*K', [dK*ρ0*K' + K*ρ0*dK' for dK in dK]
    F = CFIM(ρt, ∂ρt_∂x, M, system.eps)
    return  (abs(det(F)) < system.eps ? (1.0/system.eps) : real(tr(system.W*inv(F))))
end

