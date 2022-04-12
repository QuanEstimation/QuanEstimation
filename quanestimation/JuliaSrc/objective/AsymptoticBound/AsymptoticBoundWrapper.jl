function Objective(dynamics::AbstractDynamics, obj::QFIM_Obj{P,D}) where {P,D}
    (;W, eps) = obj
    if ismissing(W)
        W = I(get_para(dynamics.data))|>Matrix
    end
    
    d = LD_type(obj) |> eval
    p = para_type(dynamics.data) |> eval

    return QFIM_Obj{p,d}(W,eps)
end  

function Objective(dynamics::AbstractDynamics, obj::CFIM_Obj{P}) where {P}
    (;W, M, eps) = obj
    if ismissing(W)
        W = I(get_para(dynamics.data))|>Matrix
    end
    
    if ismissing(M)
        M = SIC(get_dim(dynamics.data))
    end
    
    p = para_type(dynamics.data) |> eval

    return CFIM_Obj{p}(M,W,eps)
end  

function Objective(dynamics::AbstractDynamics, obj::HCRB_Obj{P}) where {P}
    (;W, eps) = obj
    if ismissing(W)
        W = I(get_para(dynamics.data))|>Matrix
    end

    p = para_type(dynamics.data) |> eval

    return HCRB_Obj{p}(W,eps)
end  


