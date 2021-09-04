mutable struct DiffEvo{T <: Complex,M <: Real} <: ControlSystem
    freeHamiltonian::Matrix{T}
    Hamiltonian_derivative::Vector{Matrix{T}}
    ρ_initial::Matrix{T}
    times::Vector{M}
    Liouville_operator::Vector{Matrix{T}}
    γ::Vector{M}
    control_Hamiltonian::Vector{Matrix{T}}
    control_coefficients::Vector{Vector{M}}
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    DiffEvo(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, ρ_initial::Matrix{T},
             times::Vector{M}, Liouville_operator::Vector{Matrix{T}},γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}},
             control_coefficients::Vector{Vector{M}}, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = new{T,M}(freeHamiltonian, 
                Hamiltonian_derivative, ρ_initial, times, Liouville_operator, γ, control_Hamiltonian, control_coefficients, ρ, ∂ρ_∂x) 
end

function DiffEvo_QFI(DE::DiffEvo{T}, population=10, c=0.5, c0=0.1, c1=0.6, seed=1234, episode=100)
    dim = size(DiffEvo.freeHamiltonian)[1]
    tnum = length(DiffEvo.times)
    cnum = length(DiffEvo.control_Hamiltonian)
    ctrl_length = length(DiffEvo.control_Hamiltonian)

    p_num = population
    populations = repeat(DE, p_num)
    p_fit = zeros(p_num)

    for ei in 1:episode

end


