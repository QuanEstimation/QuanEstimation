function Objective(dynamics::AbstractDynamics, obj::QFIM_obj{P,D}) where {P,D}
    (;W, eps) = obj
    if ismissing(W)
        W = I(get_para(dynamics.data))|>Matrix
    end
    
    d = LD_type(obj) |> eval
    p = para_type(dynamics.data) |> eval

    return QFIM_obj{p,d}(W,eps)
end  

function Objective(dynamics::AbstractDynamics, obj::CFIM_obj{P}) where {P}
    (;W, M, eps) = obj
    if ismissing(W)
        W = I(get_para(dynamics.data))|>Matrix
    end
    
    if ismissing(M)
        M = SIC(get_dim(dynamics.data))
    end
    
    p = para_type(dynamics.data) |> eval

    return CFIM_obj{p}(M,W,eps)
end  

function Objective(dynamics::AbstractDynamics, obj::HCRB_obj{P}) where {P}
    (;W, eps) = obj
    if ismissing(W)
        W = I(get_para(dynamics.data))|>Matrix
    end

    p = para_type(dynamics.data) |> eval

    return HCRB_obj{p}(W, eps)
end  


