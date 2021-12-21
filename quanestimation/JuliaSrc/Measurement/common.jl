mutable struct MeasurementOpt{T<:Complex, M <:Real} <:ControlSystem
    freeHamiltonian
    Hamiltonian_derivative::Vector{Matrix{T}}
    ρ0::Matrix{T}
    tspan::Vector{M}
    decay_opt::Vector{Matrix{T}}
    γ::Vector{M}
    Measurement::Vector{Vector{T}}
    W::Matrix{M}
    accuracy::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    MeasurementOpt(freeHamiltonian, Hamiltonian_derivative::Vector{Matrix{T}}, ρ0::Matrix{T}, tspan::Vector{M}, 
    decay_opt::Vector{Matrix{T}},γ::Vector{M}, Measurement::Vector{Vector{T}}, W::Matrix{M}, accuracy::M, 
    ρ=Vector{Matrix{T}}(undef, 1), ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T<:Complex, M<:Real}=
    new{T,M}(freeHamiltonian, Hamiltonian_derivative, ρ0, tspan, decay_opt, γ, Measurement, W, accuracy, ρ, ∂ρ_∂x) 
end

function SaveFile_meas(f_now::Float64, Measurement)
    open("f.csv","a") do f
        writedlm(f, [f_now])
    end
    open("measurements.csv","a") do g
        writedlm(g, Measurement)
    end
end

function SaveFile_meas(f_now::Vector{Float64}, Measurement)
    open("f.csv","w") do f
        writedlm(f, f_now)
    end
    open("measurements.csv","w") do g
        writedlm(g, Measurement)
    end
end

function GramSchmidt(A::Vector{Vector{ComplexF64}})
    n = length(A[1])
    m = length(A)
    Q = [zeros(ComplexF64,n) for i in 1:m]
    for j in 1:m
        q = A[j]
        for i in 1:j
            rij = dot(Q[i], q)
            q = q - rij*Q[i]
        end
        Q[j] = q/norm(q)
    end
    Q
end