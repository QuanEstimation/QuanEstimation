using Zygote
using Zygote:@ignore
using Zygote:@adjoint
using LinearAlgebra
using SparseArrays

sigmax() = [.0im 1.;1. 0.]
sigmay() = [0. -1.0im;1.0im 0.]
sigmaz() = [1.0  .0im;0. -1.]
sigmam() = [.0im 0.;1. 0.]

function vec2mat(x::Vector{T}) where {T <: Number}
    reshape(x, x|>length|>sqrt|>Int, :)  
end


#function vec2mat(x::Vector{Vector{T}}) where {T <: Number}
#    map(x -> reshape(x, Int(sqrt(length(x))), :), x)
#end

#function vec2mat(x::Array{Array{Array{T,1},1},1}) where {T <: Number}
#    map(x -> map(x -> reshape(x, Int(sqrt(length(x))), :), x), x)
#end

function vec2mat(x)
    vec2mat.(x)
end

function liouville_commu(H::Matrix{T}) where {T <: Complex}
    kron(H, one(H)) - kron(one(H), transpose(H))
end

function liouville_commu(H) 
    kron(H, one(H)) - kron(one(H), transpose(H))
end



function destroy(M::T) where T <: Int
    spdiagm(M,M,1 => map(x -> x |> sqrt, 1:(M - 1)))
end


function liouville_dissip(Γ::Matrix{T}) where {T <: Complex}
    kron(Γ, conj(Γ)) - 0.5 * (kron(Γ' * Γ, one(Γ)) + kron(one(Γ), transpose(Γ) * conj(Γ)))
end

function dissipation(Γ::Vector{Matrix{T}}, γ::Vector{R}, t::Real) where {T <: Complex, R <:Real}
    [γ[i] * liouville_dissip(Γ[i]) for i in 1:length(Γ)] |> sum
end

function dissipation(Γ::Vector{Matrix{T}}, γ::Vector{Vector{R}}, t::Real) where {T <: Complex, R <:Real}
    [γ[i][t] * liouville_dissip(Γ[i]) for i in 1:length(Γ)] |> sum
end

function free_evolution(H0::Matrix{T}) where {T <: Complex}
    -1.0im * liouville_commu(H0)
end


function liouvillian(H0::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ, Htot::Matrix{T}, t::Real) where {T <: Complex}
    # dt = times[2] - times[1]
    dim = size(H0)[1]
    freepart = liouville_commu(Htot)
    dissp = dissipation(Liouville_operator, γ, t)
    -1.0im * freepart + dissp
end

# TODO: sparcilization ?



function _evolution( H0::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, times::Vector{M}, t::M) :: Matrix{T} where {T <: Complex, R <: Real, M <: Real}
    #x = x |> real |> complex
    Htot = H0
    tj = Int(round((t - times[1]) / (times[2] - times[1]))) + 1 
    dt = times[2] - times[1]
    
    for hn = 1:length(control_Hamiltonian)
        Hc_temp = control_coefficients[hn]
        Htot += control_Hamiltonian[hn] * Hc_temp[tj]
    end
    Ld = dt * liouvillian(H0, Liouville_operator, γ, Htot, tj)
    exp(Ld)
end

function _evolution( H0::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, times::StepRangeLen{M, Base.TwicePrecision{M}, Base.TwicePrecision{M}}, t::M) :: Matrix{T} where {T <: Complex, R <: Real, M <: Real}
    #x = x |> real |> complex
    Htot = H0
    tj = Int(round((t - times[1]) / (times[2] - times[1]))) + 1 
    dt = times[2] - times[1]
    
    for hn = 1:length(control_Hamiltonian)
        Hc_temp = control_coefficients[hn]
        Htot += control_Hamiltonian[hn] * Hc_temp[tj]
    end
    Ld = dt * liouvillian(H0, Liouville_operator, γ, Htot, tj)
    exp(Ld)
end

function _evolution( H0, Liouville_operator, γ, control_Hamiltonian, control_coefficients, times, t) 
    #x = x |> real |> complex
    Htot = H0
    tj = Int(round((t - times[1]) / (times[2] - times[1]))) + 1 
    dt = times[2] - times[1]
    
    for hn = 1:length(control_Hamiltonian)
        Hc_temp = control_coefficients[hn]
        Htot += control_Hamiltonian[hn] * Hc_temp[tj]
    end
    Ld = dt * liouvillian(H0, Liouville_operator, γ, Htot, tj)
    exp(Ld)
end
# TODO: γ ::Vector{Vector{Number}} -> γ(t)

function evolution(H0::Matrix{T},dH::Vector{Matrix{T}},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, times::Vector{M}) where {T <: Complex, R<: Real, M <: Real}
    dim = size(H0)[1]
    ρt = [zeros(ComplexF64, dim^2) for i in 1:length(times)]
    dρt = [[zeros(ComplexF64, dim^2) for i in 1:length(times)] for para in 1:length(dH)]
    ρt[1] += _evolution(H0, Liouville_operator, γ, control_Hamiltonian, control_coefficients, times, times[1]) * (ρ_initial|>vec)
    for para in 1:length(dH)
        dρt[para][1] = ((jacobian(x->(_evolution(x, Liouville_operator, γ, control_Hamiltonian, control_coefficients, times, times[1])) |>real, H0)[1]) * (dH[para]|>vec) |>vec2mat) *  (ρ_initial|>vec)

        for i in 2:length(times)
            ρt[i] = _evolution(H0, Liouville_operator, γ, control_Hamiltonian, control_coefficients, times, times[i]) * (ρt[i-1]|>vec) 
            dρt[para][i] = ((jacobian(x->(_evolution(x, Liouville_operator, γ, control_Hamiltonian, control_coefficients, times, times[i])) |>real, H0)[1]) *(dH[para]|>vec) |>vec2mat) *  (ρt[i-1]|>vec) + _evolution(H0,Liouville_operator, γ, control_Hamiltonian, control_coefficients, times, times[i]) * dρt[para][i-1]
        end
    end
    ρt,dρt
end
function evolution(H0,dH,  ρ_initial, Liouville_operator, γ, control_Hamiltonian, control_coefficients, times) 
    dim = size(H0)[1]
    ρt = [zeros(ComplexF64, dim^2) for i in 1:length(times)]
    dρt = [[zeros(ComplexF64, dim^2) for i in 1:length(times)] for para in 1:length(dH)]
    ρt[1] += _evolution(H0, Liouville_operator, γ, control_Hamiltonian, control_coefficients, times, times[1]) * (ρ_initial|>vec)
    for para in 1:length(dH)
        dρt[para][1] = ((jacobian(x->(_evolution(x, Liouville_operator, γ, control_Hamiltonian, control_coefficients, times, times[1])) |>real, H0)[1]) * (dH[para]|>vec) |>vec2mat) *  (ρ_initial|>vec)

        @inbounds for i in 2:length(times)
            ρt[i] = _evolution(H0, Liouville_operator, γ, control_Hamiltonian, control_coefficients, times, times[i]) * (ρt[i-1]|>vec) 
            dρt[para][i] = ((jacobian(x->(_evolution(x, Liouville_operator, γ, control_Hamiltonian, control_coefficients, times, times[i])) |>real, H0)[1]) *(dH[para]|>vec) |>vec2mat) *  (ρt[i-1]|>vec) + _evolution(H0,Liouville_operator, γ, control_Hamiltonian, control_coefficients, times, times[i]) * dρt[para][i-1]
        end
    end
    ρt,dρt
end
