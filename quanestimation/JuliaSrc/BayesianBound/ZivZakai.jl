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

function QZZB(
    ρₓ::AbstractVecOrMat,
    prior::AbstractVector,
    x::AbstractVector,
    accuracy=1e-8;
    ν::Number=1)

    x1, x2 = x[1], x[end]
    para = range(x1, stop=x2, length=length(p))

    τ = para .- para[1]
    N = length(para)
    I = trapz(τ, [τ[i]*trapz(para[1:N-i],
        [2*min(prior[j],prior[j+i])*helstrom_bound(ρₓ[j],ρₓ[j+i],ν) for j in 1:N-i])
        for i in 1:N])

    return 0.5*I|>real
end  # Quantum Ziv-Zakai bound for equally likely hypotheses without valley-filling

function QZZB(
    ρₓ::AbstractVecOrMat,
    prior::AbstractVector,
    x::AbstractVector,
    ::Type{Val{:opt}},
    accuracy=1e-8;
    ν::Number=1)

    x1, x2 = x[1], x[end]
    para = range(x1, stop=x2, length=length(p))
    τ = para .- para[1]
    N = length(para)
    I = trapz(τ, [τ[i]*trapz(para[1:N-i],
        [max([2*min(prior[j],prior[j+k])*helstrom_bound(ρₓ[j],ρₓ[j+k],ν) for k in 1:N-j]...)
        for j in 1:N-i]) for i in 1:N])
        
    return I
end  # Quantum Ziv-Zakai bound for equally likely hypotheses with valley-filling
