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

function PSO_QFIM(pso::PSO{T}; particle_num= 10, c0=1.0, c1=2.0, c2=2.0,v0=0.01, sd=100, ctrl_max=1e3, episode=400) where {T<: Complex}
    println("PSO strategies")
    println("searching optimal controls with particle swarm optimization ")
    Random.seed!(sd)
    tnum = length(pso.times)
    ctrl_num = length(pso.control_Hamiltonian)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ctrl_num, tnum, particle_num)
    pbest = zeros(ctrl_num, tnum, particle_num)
    gbest = zeros(ctrl_num, tnum)
    velocity_best = zeros(ctrl_num,tnum)
    p_fit = zeros(particle_num)
    qfi_ini = QFI_eig(pso)
    println("initial QFI is $(qfi_ini)")
    # fit_pre = 0.0        
    fit = 0.0
    f_list = [qfi_ini]
    for ei in 1:(10*episode)
        @inbounds for pi in 1:particle_num
            # propagate!(particles[pi])
            f_now = QFI_eig(particles[pi])
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
                    @inbounds for nj in 1:tnum
                        gbest[dj, nj] =  particles[pj].control_coefficients[dj][nj]
                        velocity_best[dj, nj] = velocity[dj, nj, pj]
                    end
                end
            end
        end  
        @inbounds for pk in 1:particle_num
            for dk in 1:ctrl_num
                @inbounds for ck in 1:tnum
                    velocity[dk, ck, pk]  = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                          + c2*rand()*(gbest[dk, ck] - particles[pk].control_coefficients[dk][ck])
                    # particles[pk].control_coefficients[dk][ck] += velocity[dk, ck, pk]
                    particles[pk].control_coefficients[dk][ck] = (x-> (x|>abs) < ctrl_max ? x+velocity[dk, ck, pk] : ctrl_max)(particles[pk].control_coefficients[dk][ck])
                end
            end
        end

        if ei == episode#abs(fit-fit_pre) < 1e-5
            pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
            Tend = pso.times[end]
            print("\e[2K")
            println("Final QFI is $fit ($ei epochs)")
            save("controls_T$Tend.jld", "controls", pso.control_coefficients, "time_span", pso.times)
            open("ctrl_pso_T$Tend.csv","w") do g
                writedlm(g, pso.control_coefficients)
            end
            open("qfilist_pso_T$Tend.csv","w") do h
                writedlm(h, f_list)
            end
            return nothing
        end
        # fit_pre = fit
        append!(f_list, fit)
        print("current QFI is $fit ($ei epochs) \r")
    end
end

function PSO_QFIM_Adam(pso::PSO{T}; particle_num= 10, c0=1.0, c1=2.0, c2=2.0,v0=0.1, sd=114514, ctrl_max=1e3, episode=400) where {T<: Complex}
    println("PSO strategies")
    println("searching optimal controls with particle swarm optimization ")
    Random.seed!(sd)
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
        @inbounds for pk in 1:particle_num
            for dk in 1:ctrl_num
                @inbounds  @threads for ck in 1:tnum
                    velocity[dk, ck, pk]  = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                          + c2*rand()*(gbest[dk, ck] - particles[pk].control_coefficients[dk][ck]) 
                    particles[pk].control_coefficients[dk][ck] += velocity[dk, ck, pk]
                    particles[pk].control_coefficients[dk][ck] = (x-> (x|>abs) < ctrl_max ? x : ctrl_max)(particles[pk].control_coefficients[dk][ck])
                end
            end
            Adam!(particles[pk], [[velocity[c,t,pk] for t in 1:tnum] for c in 1:ctrl_num])
        end
        if ei == episode#abs(fit-fit_pre) < 1e-5
            pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
            print("\e[2K")
            println("Final QFI is $fit ($ei epochs)")
            save("controls_T$(pso.times[end]).jld", "controls", pso.control_coefficients, "time_span", pso.times)
            return nothing
        end   
        fit_pre = fit
        print("current QFI is $fit ($ei epochs)    \r")
    end
end