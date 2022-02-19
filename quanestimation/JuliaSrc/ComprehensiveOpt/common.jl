mutable struct  SC_Compopt{T <: Complex,M <: Real} <: ControlSystem
    freeHamiltonian
    Hamiltonian_derivative::Vector{Matrix{T}}
    psi::Vector{T}
    tspan::Vector{M}
    decay_opt::Vector{Matrix{T}}
    γ::Vector{M}
    control_Hamiltonian::Vector{Matrix{T}}
    control_coefficients::Vector{Vector{M}}
    ctrl_bound::Vector{M}
    W::Matrix{M}
    eps::M 
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    SC_Compopt(freeHamiltonian, Hamiltonian_derivative::Vector{Matrix{T}}, psi::Vector{T},
             tspan::Vector{M}, decay_opt::Vector{Matrix{T}},γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}},
             control_coefficients::Vector{Vector{M}}, ctrl_bound::Vector{M}, W::Matrix{M}, eps::M, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = new{T,M}(freeHamiltonian, 
                Hamiltonian_derivative, psi, tspan, decay_opt, γ, control_Hamiltonian, control_coefficients, ctrl_bound, W, eps, ρ, ∂ρ_∂x) 
end

mutable struct  SM_Compopt{T <: Complex,M <: Real} <: ControlSystem
    freeHamiltonian
    Hamiltonian_derivative::Vector{Matrix{T}}
    psi::Vector{T}
    tspan::Vector{M}
    decay_opt::Vector{Matrix{T}}
    γ::Vector{M}
    C::Vector{Vector{T}}
    W::Matrix{M}
    eps::M 
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    SM_Compopt(freeHamiltonian, Hamiltonian_derivative::Vector{Matrix{T}}, psi::Vector{T},
             tspan::Vector{M}, decay_opt::Vector{Matrix{T}},γ::Vector{M}, 
             C::Vector{Vector{T}}, W::Matrix{M}, eps::M, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = new{T,M}(freeHamiltonian, 
            Hamiltonian_derivative, psi, tspan, decay_opt, γ, C, W, eps, ρ, ∂ρ_∂x) 
end

mutable struct  CM_Compopt{T <: Complex,M <: Real} <: ControlSystem
    freeHamiltonian
    Hamiltonian_derivative::Vector{Matrix{T}}
    tspan::Vector{M}
    decay_opt::Vector{Matrix{T}}
    γ::Vector{M}
    control_Hamiltonian::Vector{Matrix{T}}
    control_coefficients::Vector{Vector{M}}
    ctrl_bound::Vector{M}
    C::Vector{Vector{T}}
    W::Matrix{M}
    eps::M 
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    CM_Compopt(freeHamiltonian, Hamiltonian_derivative::Vector{Matrix{T}}, 
             tspan::Vector{M}, decay_opt::Vector{Matrix{T}},γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}},
             control_coefficients::Vector{Vector{M}}, ctrl_bound::Vector{M}, C::Vector{Vector{T}}, W::Matrix{M}, eps::M, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = new{T,M}(freeHamiltonian, 
                Hamiltonian_derivative, tspan, decay_opt, γ, control_Hamiltonian, control_coefficients, ctrl_bound, C, W, eps, ρ, ∂ρ_∂x) 
end

mutable struct  SCM_Compopt{T <: Complex,M <: Real} <: ControlSystem
    freeHamiltonian
    Hamiltonian_derivative::Vector{Matrix{T}}
    psi::Vector{T}
    tspan::Vector{M}
    decay_opt::Vector{Matrix{T}}
    γ::Vector{M}
    control_Hamiltonian::Vector{Matrix{T}}
    control_coefficients::Vector{Vector{M}}
    ctrl_bound::Vector{M}
    C::Vector{Vector{T}}
    W::Matrix{M}
    eps::M 
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    SCM_Compopt(freeHamiltonian, Hamiltonian_derivative::Vector{Matrix{T}}, psi::Vector{T},
             tspan::Vector{M}, decay_opt::Vector{Matrix{T}},γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}},
             control_coefficients::Vector{Vector{M}}, ctrl_bound::Vector{M}, C::Vector{Vector{T}}, W::Matrix{M}, eps::M, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = new{T,M}(freeHamiltonian, 
                Hamiltonian_derivative, psi, tspan, decay_opt, γ, control_Hamiltonian, control_coefficients, ctrl_bound, C, W, eps, ρ, ∂ρ_∂x) 
end

function SaveFile_SC(f_now::Float64, state, control)
    open("f.csv","a") do f
        writedlm(f, f_now)
    end
    open("states.csv","a") do g
        writedlm(g, state)
    end
    open("controls.csv","a") do m
        writedlm(m, control)
    end
end

function SaveFile_SC(f_now::Vector{Float64}, state, control)
    open("f.csv","w") do f
        writedlm(f, f_now)
    end
    open("states.csv","w") do g
        writedlm(g, state)
    end
    open("controls.csv","w") do m
        writedlm(m, control)
    end
end

function SaveFile_SM(f_now::Vector{Float64}, state, measurements)
    open("f.csv","w") do f
        writedlm(f, f_now)
    end
    open("states.csv","w") do g
        writedlm(g, state)
    end
    open("measurements.csv","w") do m
        writedlm(m, measurements)
    end
end

function SaveFile_CM(f_now::Vector{Float64}, control, measurements)
    open("f.csv","w") do f
        writedlm(f, f_now)
    end
    open("controls.csv","w") do m
        writedlm(m, control)
    end
    open("measurements.csv","w") do n
        writedlm(n, measurements)
    end
end

function SaveFile_SCM(f_now::Vector{Float64}, state, control, measurements)
    open("f.csv","w") do f
        writedlm(f, f_now)
    end
    open("states.csv","w") do g
        writedlm(g, state)
    end
    open("controls.csv","w") do m
        writedlm(m, control)
    end
    open("measurements.csv","w") do n
        writedlm(n, measurements)
    end
end
