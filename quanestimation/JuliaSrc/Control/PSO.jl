mutable struct PSO{T <: Complex,M <: Real} <: ControlSystem
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
    PSO(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, ρ_initial::Matrix{T},
             times::Vector{M}, Liouville_operator::Vector{Matrix{T}},γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}},
             control_coefficients::Vector{Vector{M}}, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = new{T,M}(freeHamiltonian, 
                Hamiltonian_derivative, ρ_initial, times, Liouville_operator, γ, control_Hamiltonian, control_coefficients, ρ, ∂ρ_∂x) 
end



function PSO_QFIM(pso::PSO{T};  particle_num= 10, c0=1.0, c1=2.0, c2=2.0,v0=0.1, sd=1234, episode=400) where {T<: Complex}
    println("PSO strategies")
    println("searching optimal controls with particle swarm optimization ")
    tnum = length(pso.times)
    ctrl_num = length(pso.control_Hamiltonian)
    particles = repeat(pso, particle_num)
    velocity = v0.*rand(ctrl_num, tnum, particle_num) |>SharedArray
    pbest = zeros(ctrl_num, tnum, particle_num) |>SharedArray
    gbest = zeros(ctrl_num, tnum) |> SharedArray
    velocity_best = zeros(ctrl_num,tnum)
    p_fit = zeros(particle_num)
    qfi_ini = QFI(pso)
    println("initial QFI is $(qfi_ini)")
    fit_pre = 0.0        
    fit = 0.0
    for ei in 1:episode
        @inbounds for pi in 1:particle_num
            # propagate!(particles[pi])
            f_now = QFI(particles[pi])
            if f_now > p_fit[pi]
                p_fit[pi] = f_now
                for di in 1:ctrl_num
                    for ni in 1:tnum
                        pbest[di,ni,pi] = particles[pi].control_coefficients[di][ni]
                    end
                end
            end
        end
        @inbounds for pj in 1:particle_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                for dj in 1:ctrl_num
                     @inbounds @threads for nj in 1:tnum
                        gbest[dj, nj] =  particles[pj].control_coefficients[dj][nj]
                        velocity_best[dj, nj] = velocity[dj, nj, pj]
                    end
                end
            end
        end
        Random.seed!(sd)
        @inbounds for pk in 1:particle_num
            for dk in 1:ctrl_num
                @inbounds  @threads for ck in 1:tnum
                    velocity[dk, ck, pk]  = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                          + c2*rand()*(gbest[dk, ck] - particles[pk].control_coefficients[dk][ck])  
                    particles[pk].control_coefficients[dk][ck] = particles[pk].control_coefficients[dk][ck] + velocity[dk, ck, pk] 
                end
            end
        end
        # println(particles[1].control_coefficients[1][1])
        if ei == episode#abs(fit-fit_pre) < 1e-5
            pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
            println("Final QFI is ", fit)
            return nothing
        end        
        fit_pre = fit
        print("current QFI is $fit ($ei epochs)    \r")
    end
end
function PSO_QFIM_Adam(pso::PSO{T};  particle_num= 10, c0=1.0, c1=2.0, c2=2.0, v0=0.1, sd=1234, episode=400) where {T<: Complex}
    println("PSO strategies")
    println("searching optimal controls with particle swarm optimization ")
    tnum = length(pso.times)
    ctrl_num = length(pso.control_Hamiltonian)
    particles, velocity = repeat(pso, particle_num), [[v0*ones(tnum) for i in 1:ctrl_num] for j in 1:particle_num]
    pbest = [[zeros(tnum) for i in 1:ctrl_num] for j in 1:particle_num]
    gbest = [zeros(tnum) for i in 1:ctrl_num]
    velocity_best = [zeros(Float64, tnum) for i in 1:ctrl_num]
    p_fit = zeros(particle_num)
    qfi_ini = QFI(pso)
    println("initial QFI is $(qfi_ini)")
    fit_pre = 0.0        
    fit = 0.0
    for ei in 1:episode
        for pi in 1:particle_num
            propagate!(particles[pi])
            f_now = QFI(particles[pi])
            
            if f_now > p_fit[pi]
                p_fit[pi] = f_now
                for di in 1:ctrl_num
                    for ni in 1:tnum
                        pbest[pi][di][ni] = particles[pi].control_coefficients[di][ni]
                    end
                end
            end
        end
        for pj in 1:particle_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                for dj in 1:ctrl_num
                    for nj in 1:tnum
                        gbest[dj][nj] =  particles[pj].control_coefficients[dj][nj]
                        velocity_best[dj][nj] = velocity[pj][dj][nj]
                    end
                end
            end
        end
        Random.seed!(sd)
        for pk in 1:particle_num
            for dk in 1:ctrl_num
                for ck in 1:tnum
                    velocity[pk][dk][ck]  = c0*velocity[pk][dk][ck] + c1*(rand()-0.5)*(pbest[pk][dk][ck] - particles[pk].control_coefficients[dk][ck]) 
                                          + c2*(rand()-0.5)*(gbest[dk][ck] - particles[pk].control_coefficients[dk][ck])  
                end
            end
            Adam!(particles[pk], velocity[pk])
        end
        if abs(fit-fit_pre) < 1e-5
            pso.control_coefficients = gbest
            println("Final QFI is ", fit)
            return nothing
        end        
        fit_pre = fit
        print("current QFI is $fit ($ei epochs)    \r")
    end
end