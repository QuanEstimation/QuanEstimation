trace_norm(X::AbstractMatrix{<:Number}) = norm(X|>svdvals, 1)

trace_norm(ρ::AbstractMatrix{<:Number}, σ::AbstractMatrix{<:Number}) = trace_norm(ρ-σ)

function fidelity(ρ::AbstractMatrix{<:Number}, σ::AbstractMatrix{<:Number})
    return (ρ|>sqrt)*σ*(ρ|>sqrt)|>sqrt|>tr|>real|>x->x^2
end # fidelity for density matrixes

function fidelity(ψ::AbstractVector{<:Number}, ϕ::AbstractVector{<:Number})
    overlap = ψ'ϕ
    return overlap'overlap
end  # fidelity for pure states

# Helstorm bound of error probability for the hypothesis testing problem 
function helstrom_bound(ρ::AbstractMatrix{<:Number},σ::AbstractMatrix{<:Number},ν=1,P0=0.5)
    return (1-trace_norm(P0*ρ-(1-P0)*σ))/2 |> real 
end

function helstrom_bound(ψ::AbstractVector{<:Number},ϕ::AbstractVector{<:Number},ν=1)
    return (1-sqrt(1-fidelity(ψ, ϕ))^ν)/2 |> real 
end

prior_uniform(W=1., μ=0.) = x -> abs(x-μ)>abs(W/2) ? 0 : 1/W

function QZZB(x::AbstractVector, p::AbstractVector, rho::AbstractVecOrMat; eps=1e-8, ν::Number=1)
    if typeof(x[1]) == Vector{Float64} || typeof(x[1]) == Vector{Int64}
        x = x[1]
    end

    tau = x .- x[1]
    p_num = length(p)
    f_tau = zeros(p_num)
    for i in 1:p_num
        arr = [real(2*minimum([p[j],p[j+i-1]])*helstrom_bound(rho[j],rho[j+i-1],ν)) for j in 1:p_num-i+1]
        f_tp = trapz(x[1:p_num-i+1], arr)
        f_tau[i] = f_tp
    end
    arr2 = [tau[m]*maximum(f_tau[m:end]) for m in 1:p_num]
    I = trapz(tau, arr2)

    return 0.5*I
end  # Quantum Ziv-Zakai bound for equally likely hypotheses with valley-filling
