function obj_func(x::Val{:HCRB_TimeIndepend_Kraus}, system, M)
    K, dK, ρ0 = system.K, system.dK, system.psi*system.psi'
    ρt, ∂ρt_∂x = K*ρ0*K', [dK*ρ0*K' + K*ρ0*dK' for dK in dK]
    f, X, V = Holevo_bound_tgt(ρt, ∂ρt_∂x, system.W, system.eps)   
    return f        
end


function obj_func(x::Val{:HCRB_TimeIndepend_Kraus}, system, M, psi)
    K, dK, ρ0 = system.K, system.dK, psi*psi'
    ρt, ∂ρt_∂x = K*ρ0*K', [dK*ρ0*K' + K*ρ0*dK' for dK in dK]
    f, X, V = Holevo_bound_tgt(ρt, ∂ρt_∂x, system.W, system.eps)     
    return f      
end