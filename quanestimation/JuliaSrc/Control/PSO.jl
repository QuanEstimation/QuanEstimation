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

function PSO_QFI(pso::PSO{T}, episode; particle_num, c0, c1, c2, v0, sd, ctrl_max, save_file) where {T<: Complex}
    println("Particle Swarm Optimization:")
    println("quantum parameter estimation")
    println("single parameter scenario")
    Random.seed!(sd)
    ctrl_length = length(pso.control_coefficients[1])
    ctrl_num = length(pso.control_Hamiltonian)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ctrl_num, ctrl_length, particle_num)
    pbest = zeros(ctrl_num, ctrl_length, particle_num)
    gbest = zeros(ctrl_num, ctrl_length)
    velocity_best = zeros(ctrl_num,ctrl_length)
    p_fit = zeros(particle_num)
    qfi_ini = QFI_ori(pso)
    println("initial QFI is $(qfi_ini)")
    Tend = pso.times[end]
    # fit_pre = 0.0        
    fit = 0.0
    f_list = [qfi_ini]
    if save_file==true
        for ei in 1:episode
            @inbounds for pi in 1:particle_num
                # propagate!(particles[pi])
                f_now = QFI_ori(particles[pi])
                if f_now > p_fit[pi]
                    p_fit[pi] = f_now
                    for di in 1:ctrl_num
                        for ni in 1:ctrl_length
                            pbest[di,ni,pi] = particles[pi].control_coefficients[di][ni]
                        end
                    end
                end
            end
            @inbounds for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:ctrl_num
                        @inbounds for nj in 1:ctrl_length
                            gbest[dj, nj] =  particles[pj].control_coefficients[dj][nj]
                            velocity_best[dj, nj] = velocity[dj, nj, pj]
                        end
                    end
                end
            end  
            @inbounds for pk in 1:particle_num
                for dk in 1:ctrl_num
                    @inbounds for ck in 1:ctrl_length
                        velocity[dk, ck, pk]  = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                              + c2*rand()*(gbest[dk, ck] - particles[pk].control_coefficients[dk][ck])
                        # particles[pk].control_coefficients[dk][ck] += velocity[dk, ck, pk]
                        particles[pk].control_coefficients[dk][ck] = (x-> (x|>abs) < ctrl_max ? x+velocity[dk, ck, pk] : ctrl_max)(particles[pk].control_coefficients[dk][ck])
                    end
                end
            end
            append!(f_list, fit)
            print("current QFI is $fit ($ei epochs) \r")
            open("ctrl_pso_T$Tend.csv","w") do g
                writedlm(g, [gbest[k, :] for k in 1:ctrl_num])
            end
            open("f_pso_T$Tend.csv","w") do h
                writedlm(h, f_list)
            end
        end
        print("\e[2K")
        println("Final QFI is $fit ($(episode[1]) epochs)")

    else
        for ei in 1:episode
            @inbounds for pi in 1:particle_num
                # propagate!(particles[pi])
                f_now = QFI_ori(particles[pi])
                if f_now > p_fit[pi]
                    p_fit[pi] = f_now
                    for di in 1:ctrl_num
                        for ni in 1:ctrl_length
                            pbest[di,ni,pi] = particles[pi].control_coefficients[di][ni]
                        end
                    end
                end
            end
            @inbounds for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:ctrl_num
                        @inbounds for nj in 1:ctrl_length
                            gbest[dj, nj] =  particles[pj].control_coefficients[dj][nj]
                            velocity_best[dj, nj] = velocity[dj, nj, pj]
                        end
                    end
                end
            end  
            @inbounds for pk in 1:particle_num
                for dk in 1:ctrl_num
                    @inbounds for ck in 1:ctrl_length
                        velocity[dk, ck, pk]  = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                              + c2*rand()*(gbest[dk, ck] - particles[pk].control_coefficients[dk][ck])
                        # particles[pk].control_coefficients[dk][ck] += velocity[dk, ck, pk]
                        particles[pk].control_coefficients[dk][ck] = (x-> (x|>abs) < ctrl_max ? x+velocity[dk, ck, pk] : ctrl_max)(particles[pk].control_coefficients[dk][ck])
                    end
                end
            end
            append!(f_list, fit)
            print("current QFI is $fit ($ei epochs) \r")
        end
        pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
        print("\e[2K")
        # save("controls_T$Tend.jld", "controls", pso.control_coefficients, "time_span", pso.times)
        open("ctrl_pso_T$Tend.csv","w") do g
            writedlm(g, pso.control_coefficients)
        end
        open("f_pso_T$Tend.csv","w") do h
            writedlm(h, f_list)
        end
        println("Final QFI is $fit ($(episode[1]) epochs)")
    end
    return nothing
end

function PSO_QFI(pso::PSO{T}, episode::Vector{Int64}; particle_num, c0, c1, c2,v0, sd, ctrl_max, save_file) where {T<: Complex}
    println("Particle Swarm Optimization:")
    println("quantum parameter estimation")
    println("single parameter scenario")
    Random.seed!(sd)
    ctrl_length = length(pso.control_coefficients[1])
    ctrl_num = length(pso.control_Hamiltonian)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ctrl_num, ctrl_length, particle_num)
    pbest = zeros(ctrl_num, ctrl_length, particle_num)
    gbest = zeros(ctrl_num, ctrl_length)
    velocity_best = zeros(ctrl_num,ctrl_length)
    p_fit = zeros(particle_num)
    qfi_ini = QFI_ori(pso)
    println("initial QFI is $(qfi_ini)")
    Tend = pso.times[end]
    # fit_pre = 0.0
    fit = 0.0
    f_list = [qfi_ini]
    if save_file==true
        for ei in 1:episode[1]
            @inbounds for pi in 1:particle_num
                # propagate!(particles[pi])
                f_now = QFI_ori(particles[pi])
                if f_now > p_fit[pi]
                    p_fit[pi] = f_now
                    for di in 1:ctrl_num
                        for ni in 1:ctrl_length
                            pbest[di,ni,pi] = particles[pi].control_coefficients[di][ni]
                        end
                    end
                end
            end
            @inbounds for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:ctrl_num
                        @inbounds for nj in 1:ctrl_length
                            gbest[dj, nj] =  particles[pj].control_coefficients[dj][nj]
                            velocity_best[dj, nj] = velocity[dj, nj, pj]
                        end
                    end
                end
            end  
            @inbounds for pk in 1:particle_num
                for dk in 1:ctrl_num
                    @inbounds for ck in 1:ctrl_length
                        velocity[dk, ck, pk]  = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                              + c2*rand()*(gbest[dk, ck] - particles[pk].control_coefficients[dk][ck])
                        # particles[pk].control_coefficients[dk][ck] += velocity[dk, ck, pk]
                        particles[pk].control_coefficients[dk][ck] = (x-> (x|>abs) < ctrl_max ? x+velocity[dk, ck, pk] : ctrl_max)(particles[pk].control_coefficients[dk][ck])
                    end
                end
            end
    
            if ei%episode[2] == 0
                pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            print("current QFI is $fit ($ei epochs) \r")
            open("ctrl_pso_T$Tend.csv","w") do g
                writedlm(g, [gbest[k, :] for k in 1:ctrl_num])
            end
            open("f_pso_T$Tend.csv","w") do h
                writedlm(h, f_list)
            end
        end
        print("\e[2K")    
        println("Final QFI is $fit ($(episode[1]) epochs)")
    else
        for ei in 1:episode[1]
            @inbounds for pi in 1:particle_num
                # propagate!(particles[pi])
                f_now = QFI_ori(particles[pi])
                if f_now > p_fit[pi]
                    p_fit[pi] = f_now
                    for di in 1:ctrl_num
                        for ni in 1:ctrl_length
                            pbest[di,ni,pi] = particles[pi].control_coefficients[di][ni]
                        end
                    end
                end
            end
            @inbounds for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:ctrl_num
                        @inbounds for nj in 1:ctrl_length
                            gbest[dj, nj] =  particles[pj].control_coefficients[dj][nj]
                            velocity_best[dj, nj] = velocity[dj, nj, pj]
                        end
                    end
                end
            end  
            @inbounds for pk in 1:particle_num
                for dk in 1:ctrl_num
                    @inbounds for ck in 1:ctrl_length
                        velocity[dk, ck, pk]  = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                              + c2*rand()*(gbest[dk, ck] - particles[pk].control_coefficients[dk][ck])
                        # particles[pk].control_coefficients[dk][ck] += velocity[dk, ck, pk]
                        particles[pk].control_coefficients[dk][ck] = (x-> (x|>abs) < ctrl_max ? x+velocity[dk, ck, pk] : ctrl_max)(particles[pk].control_coefficients[dk][ck])
                    end
                end
            end
    
            if ei%episode[2] == 0
                pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            print("current QFI is $fit ($ei epochs) \r")
        end
        pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
        print("\e[2K")    
        save("controls_T$Tend.jld", "controls", pso.control_coefficients, "time_span", pso.times)
        open("ctrl_pso_T$Tend.csv","w") do g
            writedlm(g, pso.control_coefficients)
        end
        open("f_pso_T$Tend.csv","w") do h
            writedlm(h, f_list)
        end
        println("Final QFI is $fit ($(episode[1]) epochs)")
    end
    return nothing
end

function PSO_QFIM(pso::PSO{T}, episode; particle_num, c0, c1, c2, v0, sd, ctrl_max, save_file) where {T<: Complex}
    println("Particle Swarm Optimization:")
    println("quantum parameter estimation")
    println("multiparameter scenario")
    Random.seed!(sd)
    ctrl_length = length(pso.control_coefficients[1])
    ctrl_num = length(pso.control_Hamiltonian)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ctrl_num, ctrl_length, particle_num)
    pbest = zeros(ctrl_num, ctrl_length, particle_num)
    gbest = zeros(ctrl_num, ctrl_length)
    velocity_best = zeros(ctrl_num,ctrl_length)
    p_fit = zeros(particle_num)
    qfi_ini = 1.0/real(tr(pinv(QFIM_ori(pso))))
    println("initial value of the target function is $(1/qfi_ini)")
    Tend = pso.times[end]
    # fit_pre = 0.0        
    fit = 0.0
    f_list = [1.0/qfi_ini]
    if save_file==true
        for ei in 1:episode
            @inbounds for pi in 1:particle_num
                # propagate!(particles[pi])
                f_now = 1.0/real(tr(pinv(QFIM_ori(particles[pi]))))
                if f_now > p_fit[pi]
                    p_fit[pi] = f_now
                    for di in 1:ctrl_num
                        for ni in 1:ctrl_length
                            pbest[di,ni,pi] = particles[pi].control_coefficients[di][ni]
                        end
                    end
                end
            end
            @inbounds for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:ctrl_num
                        @inbounds for nj in 1:ctrl_length
                            gbest[dj, nj] =  particles[pj].control_coefficients[dj][nj]
                            velocity_best[dj, nj] = velocity[dj, nj, pj]
                        end
                    end
                end
            end  
            @inbounds for pk in 1:particle_num
                for dk in 1:ctrl_num
                    @inbounds for ck in 1:ctrl_length
                        velocity[dk, ck, pk]  = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                              + c2*rand()*(gbest[dk, ck] - particles[pk].control_coefficients[dk][ck])
                        # particles[pk].control_coefficients[dk][ck] += velocity[dk, ck, pk]
                        particles[pk].control_coefficients[dk][ck] = (x-> (x|>abs) < ctrl_max ? x+velocity[dk, ck, pk] : ctrl_max)(particles[pk].control_coefficients[dk][ck])
                    end
                end
            end
            append!(f_list, 1.0/fit)
            print("current value of the target function is $(1.0/fit) ($ei epochs) \r")
            open("ctrl_pso_T$Tend.csv","w") do g
                writedlm(g, [gbest[k, :] for k in 1:ctrl_num])
            end
            open("f_pso_T$Tend.csv","w") do h
                writedlm(h, f_list)
            end
        end
        print("\e[2K")
        println("Final value of the target function is $(1.0/fit) ($(episode[1]) epochs)")
    else
        for ei in 1:episode
            @inbounds for pi in 1:particle_num
                # propagate!(particles[pi])
                f_now = 1.0/real(tr(pinv(QFIM_ori(particles[pi]))))
                if f_now > p_fit[pi]
                    p_fit[pi] = f_now
                    for di in 1:ctrl_num
                        for ni in 1:ctrl_length
                            pbest[di,ni,pi] = particles[pi].control_coefficients[di][ni]
                        end
                    end
                end
            end
            @inbounds for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:ctrl_num
                        @inbounds for nj in 1:ctrl_length
                            gbest[dj, nj] =  particles[pj].control_coefficients[dj][nj]
                            velocity_best[dj, nj] = velocity[dj, nj, pj]
                        end
                    end
                end
            end  
            @inbounds for pk in 1:particle_num
                for dk in 1:ctrl_num
                    @inbounds for ck in 1:ctrl_length
                        velocity[dk, ck, pk]  = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                              + c2*rand()*(gbest[dk, ck] - particles[pk].control_coefficients[dk][ck])
                        # particles[pk].control_coefficients[dk][ck] += velocity[dk, ck, pk]
                        particles[pk].control_coefficients[dk][ck] = (x-> (x|>abs) < ctrl_max ? x+velocity[dk, ck, pk] : ctrl_max)(particles[pk].control_coefficients[dk][ck])
                    end
                end
            end
            append!(f_list, 1.0/fit)
            print("current value of the target function is $(1.0/fit) ($ei epochs) \r")
        end
        pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
        print("\e[2K")
        save("controls_T$Tend.jld", "controls", pso.control_coefficients, "time_span", pso.times)
        open("ctrl_pso_T$Tend.csv","w") do g
            writedlm(g, pso.control_coefficients)
        end
        open("f_pso_T$Tend.csv","w") do h
            writedlm(h, f_list)
        end
        println("Final value of the target function is $(1.0/fit) ($(episode[1]) epochs)")
    end
    return nothing
end

function PSO_QFIM(pso::PSO{T}, episode::Vector{Int64}, particle_num, c0, c1, c2, v0, sd, ctrl_max, save_file) where {T<: Complex}
    println("Particle Swarm Optimization:")
    println("quantum parameter estimation")
    println("multiparameter scenario")
    Random.seed!(sd)
    ctrl_length = length(pso.control_coefficients[1])
    ctrl_num = length(pso.control_Hamiltonian)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ctrl_num, ctrl_length, particle_num)
    pbest = zeros(ctrl_num, ctrl_length, particle_num)
    gbest = zeros(ctrl_num, ctrl_length)
    velocity_best = zeros(ctrl_num,ctrl_length)
    p_fit = zeros(particle_num)
    qfi_ini = 1.0/real(tr(pinv(QFIM_ori(pso))))
    println("initial value of the target function is $(1.0/qfi_ini)")
    Tend = pso.times[end]
    # fit_pre = 0.0
    fit = 0.0
    f_list = [1.0/qfi_ini]
    if save_file==true
        for ei in 1:episode[1]
            @inbounds for pi in 1:particle_num
                # propagate!(particles[pi])
                f_now = 1.0/real(tr(pinv(QFIM_ori(particles[pi]))))
                if f_now > p_fit[pi]
                    p_fit[pi] = f_now
                    for di in 1:ctrl_num
                        for ni in 1:ctrl_length
                            pbest[di,ni,pi] = particles[pi].control_coefficients[di][ni]
                        end
                    end
                end
            end
            @inbounds for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:ctrl_num
                        @inbounds for nj in 1:ctrl_length
                            gbest[dj, nj] =  particles[pj].control_coefficients[dj][nj]
                            velocity_best[dj, nj] = velocity[dj, nj, pj]
                        end
                    end
                end
            end  
            @inbounds for pk in 1:particle_num
                for dk in 1:ctrl_num
                    @inbounds for ck in 1:ctrl_length
                        velocity[dk, ck, pk]  = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                              + c2*rand()*(gbest[dk, ck] - particles[pk].control_coefficients[dk][ck])
                        # particles[pk].control_coefficients[dk][ck] += velocity[dk, ck, pk]
                        particles[pk].control_coefficients[dk][ck] = (x-> (x|>abs) < ctrl_max ? x+velocity[dk, ck, pk] : ctrl_max)(particles[pk].control_coefficients[dk][ck])
                    end
                end
            end
            if ei%episode[2] == 0
                pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, 1.0/fit)
            print("current value of the target function is $((1.0/fit)) ($ei epochs) \r")
            open("ctrl_pso_T$Tend.csv","w") do g
                writedlm(g, [gbest[k, :] for k in 1:ctrl_num])
            end
            open("f_pso_T$Tend.csv","w") do h
                writedlm(h, f_list)
            end
        end
        print("\e[2K")
        println("Final value of the target function is $(1.0/fit) ($(episode[1]) epochs)")
    else
        for ei in 1:episode[1]
            @inbounds for pi in 1:particle_num
                # propagate!(particles[pi])
                f_now = 1.0/real(tr(pinv(QFIM_ori(particles[pi]))))
                if f_now > p_fit[pi]
                    p_fit[pi] = f_now
                    for di in 1:ctrl_num
                        for ni in 1:ctrl_length
                            pbest[di,ni,pi] = particles[pi].control_coefficients[di][ni]
                        end
                    end
                end
            end
            @inbounds for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:ctrl_num
                        @inbounds for nj in 1:ctrl_length
                            gbest[dj, nj] =  particles[pj].control_coefficients[dj][nj]
                            velocity_best[dj, nj] = velocity[dj, nj, pj]
                        end
                    end
                end
            end  
            @inbounds for pk in 1:particle_num
                for dk in 1:ctrl_num
                    @inbounds for ck in 1:ctrl_length
                        velocity[dk, ck, pk]  = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                              + c2*rand()*(gbest[dk, ck] - particles[pk].control_coefficients[dk][ck])
                        # particles[pk].control_coefficients[dk][ck] += velocity[dk, ck, pk]
                        particles[pk].control_coefficients[dk][ck] = (x-> (x|>abs) < ctrl_max ? x+velocity[dk, ck, pk] : ctrl_max)(particles[pk].control_coefficients[dk][ck])
                    end
                end
            end
    
            if ei%episode[2] == 0
                pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, 1.0/fit)
            print("current value of the target function is $((1.0/fit)) ($ei epochs) \r")
        end
        pso.control_coefficients = [gbest[k, :] for k in 1:ctrl_num]
        print("\e[2K")    
        save("controls_T$Tend.jld", "controls", pso.control_coefficients, "time_span", pso.times)
        open("ctrl_pso_T$Tend.csv","w") do g
            writedlm(g, pso.control_coefficients)
        end
        open("f_pso_T$Tend.csv","w") do h
            writedlm(h, f_list)
        end
        println("Final value of the target function is $((1.0/fit)) ($(episode[1]) epochs)")
    end
    return nothing
end