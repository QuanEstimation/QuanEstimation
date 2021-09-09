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

function DiffEvo_QFI(DE::DiffEvo{T}, populations, c, c0, c1, seed, max_episodes)
    dim = size(DiffEvo.freeHamiltonian)[1]
    tnum = length(DiffEvo.times)
    ctrl_num = length(DiffEvo.control_Hamiltonian)
    ctrl_length = length(DiffEvo.control_Hamiltonian)

    p_num = population
    populations = repeat(DE, p_num)
    p_fit = [QFI(populations[i]) for i in 1:p_num]
    f_mean = p_fit |> mean

    Random.seed!(seed)
    for ei in 1:max_episodes
        for pi in 1:p_num
            #mutations
            mut_num = sample(1:p_num, 3, replace=false)
            ctrl_mut = zeros(ctrl_num, tnum)
            for ci in 1:ctrl_num
                for ti in 1:tnum
                    ctrl_mut[ci, ti] = populations[mut_num[1]].control_coefficients[ci][ti]+
                                  c*(populations[mut_num[2]].control_coefficients[ci][ti]-populations[mut_num[3]].control_coefficients[ci][ti])
                end
            end
            #crossover
            if p_fit[pi] > f_mean
                cr = c0 + (c1-c0)*(p_fit[pi]-min(p_fit))/(max(p_fit)-min(p_fit))
            else
                cr = c0
            end
            ctrl_cross = zeros(ctrl_num, tnum)
            cross_int = sample(1:p_num, 1, replace=false)
            for cj in 1:ctrl_num
                for tj in 1:tnum
                    rand_num = rand()
                    if rand_num < cr ||  cross_int == pi
                        ctrl_cross[cj, tj] = ctrl_mut[cj, tj]
                    else
                        ctrl_cross[cj, tj] = populations[pi].control_coefficients[cj][tj]
                    end
                end
            end
            #selection
            f_cross = QFI(populations[pi].freeHamiltonian, populations[pi].Hamiltonian_derivative[1], populations[pi].ρ_initial, 
                          populations[pi].Liouville_operator, populations[pi].γ, populations[pi].control_Hamiltonian, 
                          ctrl_cross, populations[pi].times)
            if f_cross > p_fit[pi]
                for ck in 1:ctrl_num
                    for tk in 1:tnum
                        populations[pi].control_coefficients[ck][tk] = ctrl_cross[ck, tk]
                    end
                end
            end
        end
    end
end


