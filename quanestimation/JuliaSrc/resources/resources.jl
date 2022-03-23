using SparseArrays

function J₊(j::Number)
    spdiagm(1=>[sqrt(j*(j+1)-m*(m+1)) for m in j:-1:-j][2:end])
end

function Jp_full(N)
    sp = [0.0 1.0; 0.0 0.0]
    Jp, jp_tp = zeros(2^N, 2^N), zeros(2^N, 2^N)
    for i in 0:N-1
        if i==0 
            jp_tp = kron(sp, Matrix(I, 2^(N-1), 2^(N-1)))
        elseif i==N-1
            jp_tp = kron(Matrix(I, 2^(N-1), 2^(N-1)), sp)
        else
            jp_tp = kron(Matrix(I, 2^i, 2^i), kron(sp, Matrix(I, 2^(N-1-i), 2^(N-1-i))))
        end
        Jp += jp_tp
    end
    Jp
end

function SpinSqueezing(ρ::AbstractMatrix; basis="Dicke", output = "KU")
    N = size(ρ)[1] - 1
    coef = 4.0/N
    j = N/2
    if basis == "Pauli"
        Jp = Jp_full(N)
    else
        Jp  = J₊(j)
    end
    Jx = 0.5*(Jp + Jp')
    Jy = -0.5im*(Jp - Jp')
    Jz = spdiagm(j:-1:-j)

    Jx_mean = tr(ρ*Jx) |> real
    Jy_mean = tr(ρ*Jy) |> real
    Jz_mean = tr(ρ*Jz) |> real

    cosθ = Jz_mean/sqrt(Jx_mean^2 + Jy_mean^2 + Jz_mean^2)
    sinθ = sin(acos(cosθ))
    cosϕ = Jx_mean/sqrt(Jx_mean^2 + Jy_mean^2)
    sinϕ = Jy_mean > 0 ? sin(acos(cosϕ)) : sin(2pi - acos(cosϕ))

    Jn1 = - Jx*sinϕ + Jy*cosϕ
    Jn2 = - Jx*cosθ*cosϕ - Jy*cosθ*sinϕ + Jz*sinθ
    A = tr(ρ*(Jn1*Jn1 - Jn2*Jn2))
    B = tr(ρ*(Jn1*Jn2 + Jn2*Jn1))
    C = tr(ρ*(Jn1*Jn1 + Jn2*Jn2))
    
    V₋ = 0.5*(C-sqrt(A^2+B^2))|>real
    ξ = coef*V₋
    ξ = ξ > 1 ? 1.0 : ξ

    if output == "KU"
        res = ξ
    elseif output == "WBIMH"
        res = (N/2)^2*ξ/(Jx_mean^2+Jy_mean^2+Jz_mean^2)
    else
        @warn "NameError: output should be choosen in {KU, WBIMH}"
    end

    return res
end

function TargetTime(f::Number, tspan, func::Function, args...;kwargs...)
    args = zip.(args...)|>a->[[x for x in x] for x in unzip(a)][1]
    tnum = length(tspan)

    f_last = func(args[1]...; kwargs...)
    idx = 2
    f_now = func(args[idx]...; kwargs...)

    while (f_now-f)*(f_last-f) > 0 && idx<tnum
        f_last = f_now
        idx += 1
        f_now = func(args[idx]...; kwargs...)
    end

    return tspan[idx]
end
