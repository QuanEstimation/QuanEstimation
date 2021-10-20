mutable struct StateOpt_PSO{T <: Complex,M <: Real}
    freeHamiltonian::Matrix{T}
    Hamiltonian_derivative::Vector{Matrix{T}}
    psi::Vector{T}
    times::Vector{M}
    Liouville_operator::Vector{Matrix{T}}
    γ::Vector{M}
    control_Hamiltonian::Vector{Matrix{T}}
    control_coefficients::Vector{Vector{M}}
    ctrl_bound::M
    W::Matrix{M}
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    StateOpt_PSO(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, psi::Vector{T},
             times::Vector{M}, Liouville_operator::Vector{Matrix{T}},γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}},
             control_coefficients::Vector{Vector{M}}, ctrl_bound::M, W::Matrix{M}, ρ=Vector{Matrix{T}}(undef, 1), 
             ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = new{T,M}(freeHamiltonian, 
                Hamiltonian_derivative, psi, times, Liouville_operator, γ, control_Hamiltonian, control_coefficients, ctrl_bound, W, ρ, ∂ρ_∂x) 
end

function PSO_QFI(pso::StateOpt_PSO{T}, max_episodes::Int64, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("state optimization")
    println("single parameter scenario")
    println("control algorithm: PSO")
    Random.seed!(sd)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
    velocity_best = zeros(ComplexF64, dim)
    p_fit = zeros(particle_num)
    
    # initialize 
    for pj in 1:length(ini_particle)
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    qfi_ini = QFI_ori(pso.freeHamiltonian, pso.Hamiltonian_derivative[1], pso.psi*(pso.psi)', pso.Liouville_operator, pso.γ, 
                      pso.control_Hamiltonian, pso.control_coefficients, pso.times)
    f_list = [qfi_ini]
    println("initial QFI is $(qfi_ini)")
    Tend = pso.times[end]        
    fit = 0.0
    if save_file==true
        for ei in 1:max_episodes
            #### train ####
            for pj in 1:particle_num
                rho = particles[pj].psi*(particles[pj].psi)'
                f_now = QFI_ori(particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative[1], rho, particles[pj].Liouville_operator, particles[pj].γ, 
                                particles[pj].control_Hamiltonian, particles[pj].control_coefficients, particles[pj].times)
                if f_now > p_fit[pj]
                    p_fit[pj] = f_now
                    for di in 1:dim
                        pbest[di, pj] = particles[pj].psi[di]
                    end
                end
            end

            for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:dim
                        gbest[dj] = particles[pj].psi[dj]
                        velocity_best[dj] = velocity[dj, pj]
                    end
                end
            end  

            for pk in 1:particle_num
                psi_pre = zeros(ComplexF64, dim)
                for dk in 1:dim
                    psi_pre[dk] = particles[pk].psi[dk]
                    velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
                    particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
                end

                particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

                for dm in 1:dim
                    velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
                end
            end
            append!(f_list, fit)
            print("current QFI is $fit ($ei episodes) \r")
            open("state_pso_T$Tend.csv","w") do g
                writedlm(g, gbest)
            end
            open("f_pso_T$Tend.csv","w") do h
                writedlm(h, f_list)
            end
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final QFI is $fit")

    else
        for ei in 1:max_episodes
            #### train ####
            for pj in 1:particle_num
                rho = particles[pj].psi*(particles[pj].psi)'
                f_now = QFI_ori(particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative[1], rho, particles[pj].Liouville_operator, particles[pj].γ, 
                                particles[pj].control_Hamiltonian, particles[pj].control_coefficients, particles[pj].times)
                if f_now > p_fit[pj]
                    p_fit[pj] = f_now
                    for di in 1:dim
                        pbest[di,pj] = particles[pj].psi[di]
                    end
                end
            end

            for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:dim
                        gbest[dj] = particles[pj].psi[dj]
                        velocity_best[dj] = velocity[dj, pj]
                    end
                end
            end  

            for pk in 1:particle_num
                psi_pre = zeros(ComplexF64, dim)
                for dk in 1:dim
                    psi_pre[dk] = particles[pk].psi[dk]
                    velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
                    particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
                end

                particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

                for dm in 1:dim
                    velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
                end
            end
            append!(f_list, fit)
            print("current QFI is $fit ($ei episodes) \r")
        end
        open("state_pso_T$Tend.csv","w") do g
            writedlm(g, gbest)
        end
        open("f_pso_T$Tend.csv","w") do h
            writedlm(h, f_list)
        end
        print("\e[2K") 
        println("Iteration over, data saved.") 
        println("Final QFI is $fit")
    end
    return nothing
end

function PSO_QFIM(pso::StateOpt_PSO{T}, max_episodes::Int64, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("state optimization")
    println("multiparameter scenario")
    println("control algorithm: PSO")
    Random.seed!(sd)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
    velocity_best = zeros(ComplexF64, dim)
    p_fit = zeros(particle_num)
    
    # initialize 
    for pj in 1:length(ini_particle)
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    F = QFIM_ori(pso.freeHamiltonian, pso.Hamiltonian_derivative, pso.psi*(pso.psi)', pso.Liouville_operator, pso.γ, 
                      pso.control_Hamiltonian, pso.control_coefficients, pso.times)
    qfi_ini = real(tr(pso.W*pinv(F)))
    f_list = [qfi_ini]
    println("initial value of Tr(WF^{-1}) is $(qfi_ini)")
    Tend = pso.times[end]        
    fit = 0.0
    if save_file==true
        for ei in 1:max_episodes
            #### train ####
            for pj in 1:particle_num
                rho = particles[pj].psi*(particles[pj].psi)'
                F_tp = QFIM_ori(particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative, rho, particles[pj].Liouville_operator, particles[pj].γ, 
                                particles[pj].control_Hamiltonian, particles[pj].control_coefficients, particles[pj].times)
                f_now = 1.0/real(tr(particles[pj].W*pinv(F_tp)))
                if f_now > p_fit[pj]
                    p_fit[pj] = f_now
                    for di in 1:dim
                        pbest[di,pj] = particles[pj].psi[di]
                    end
                end
            end

            for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:dim
                        gbest[dj] = particles[pj].psi[dj]
                        velocity_best[dj] = velocity[dj, pj]
                    end
                end
            end  

            for pk in 1:particle_num
                psi_pre = zeros(ComplexF64, dim)
                for dk in 1:dim
                    psi_pre[dk] = particles[pk].psi[dk]
                    velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
                    particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
                end

                particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

                for dm in 1:dim
                    velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
                end
            end
            append!(f_list, (1.0/fit))
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
            open("state_pso_T$Tend.csv","w") do g
                writedlm(g, gbest)
            end
            open("f_pso_T$Tend.csv","w") do h
                writedlm(h, f_list)
            end
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")

    else
        for ei in 1:max_episodes
            #### train ####
            for pj in 1:particle_num
                rho = particles[pj].psi*(particles[pj].psi)'
                F_tp = QFIM_ori(particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative, rho, particles[pj].Liouville_operator, particles[pj].γ, 
                                particles[pj].control_Hamiltonian, particles[pj].control_coefficients, particles[pj].times)
                f_now = 1.0/real(tr(particles[pj].W*pinv(F_tp)))
                if f_now > p_fit[pj]
                    p_fit[pj] = f_now
                    for di in 1:dim
                        pbest[di,pj] = particles[pj].psi[di]
                    end
                end
            end

            for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:dim
                        gbest[dj] = particles[pj].psi[dj]
                        velocity_best[dj] = velocity[dj, pj]
                    end
                end
            end  

            for pk in 1:particle_num
                psi_pre = zeros(ComplexF64, dim)
                for dk in 1:dim
                    psi_pre[dk] = particles[pk].psi[dk]
                    velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
                    particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
                end

                particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

                for dm in 1:dim
                    velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
                end
            end
            append!(f_list, (1.0/fit))
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
        end
        open("state_pso_T$Tend.csv","w") do g
            writedlm(g, gbest)
        end
        open("f_pso_T$Tend.csv","w") do h
            writedlm(h, f_list)
        end
        print("\e[2K") 
        println("Iteration over, data saved.") 
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    end
    return nothing
end

function PSO_CFI(M, pso::StateOpt_PSO{T}, max_episodes::Int64, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("state optimization")
    println("single parameter scenario")
    println("control algorithm: PSO")
    Random.seed!(sd)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
    velocity_best = zeros(ComplexF64, dim)
    p_fit = zeros(particle_num)
    
    # initialize 
    for pj in 1:length(ini_particle)
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    f_ini = CFI(M, pso.freeHamiltonian, pso.Hamiltonian_derivative[1], pso.psi*(pso.psi)', pso.Liouville_operator, pso.γ, 
                      pso.control_Hamiltonian, pso.control_coefficients, pso.times)
    f_list = [f_ini]
    println("initial CFI is $(f_ini)")
    Tend = pso.times[end]        
    fit = 0.0
    if save_file==true
        for ei in 1:max_episodes
            #### train ####
            for pj in 1:particle_num
                rho = particles[pj].psi*(particles[pj].psi)'
                f_now = CFI(M, particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative[1], rho, particles[pj].Liouville_operator, particles[pj].γ, 
                                particles[pj].control_Hamiltonian, particles[pj].control_coefficients, particles[pj].times)
                if f_now > p_fit[pj]
                    p_fit[pj] = f_now
                    for di in 1:dim
                        pbest[di,pj] = particles[pj].psi[di]
                    end
                end
            end

            for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:dim
                        gbest[dj] = particles[pj].psi[dj]
                        velocity_best[dj] = velocity[dj, pj]
                    end
                end
            end  

            for pk in 1:particle_num
                psi_pre = zeros(ComplexF64, dim)
                for dk in 1:dim
                    psi_pre[dk] = particles[pk].psi[dk]
                    velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
                    particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
                end

                particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

                for dm in 1:dim
                    velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
                end
            end
            append!(f_list, fit)
            print("current CFI is $fit ($ei episodes) \r")
            open("state_pso_T$Tend.csv","w") do g
                writedlm(g, gbest)
            end
            open("f_pso_T$Tend.csv","w") do h
                writedlm(h, f_list)
            end
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final CFI is $fit")

    else
        for ei in 1:max_episodes
            #### train ####
            for pj in 1:particle_num
                rho = particles[pj].psi*(particles[pj].psi)'
                f_now = CFI(M, particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative[1], rho, particles[pj].Liouville_operator, particles[pj].γ, 
                                particles[pj].control_Hamiltonian, particles[pj].control_coefficients, particles[pj].times)
                if f_now > p_fit[pj]
                    p_fit[pj] = f_now
                    for di in 1:dim
                        pbest[di,pj] = particles[pj].psi[di]
                    end
                end
            end

            for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:dim
                        gbest[dj] = particles[pj].psi[dj]
                        velocity_best[dj] = velocity[dj, pj]
                    end
                end
            end  

            for pk in 1:particle_num
                psi_pre = zeros(ComplexF64, dim)
                for dk in 1:dim
                    psi_pre[dk] = particles[pk].psi[dk]
                    velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
                    particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
                end

                particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

                for dm in 1:dim
                    velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
                end
            end
            append!(f_list, fit)
            print("current CFI is $fit ($ei episodes) \r")
        end
        open("state_pso_T$Tend.csv","w") do g
            writedlm(g, gbest)
        end
        open("f_pso_T$Tend.csv","w") do h
            writedlm(h, f_list)
        end
        print("\e[2K") 
        println("Iteration over, data saved.") 
        println("Final CFI is $fit")
    end
    return nothing
end

function PSO_CFIM(M, pso::StateOpt_PSO{T}, max_episodes::Int64, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("state optimization")
    println("multiparameter scenario")
    println("control algorithm: PSO")
    Random.seed!(sd)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
    velocity_best = zeros(ComplexF64, dim)
    p_fit = zeros(particle_num)
    
    # initialize 
    for pj in 1:length(ini_particle)
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    F = CFIM(M, pso.freeHamiltonian, pso.Hamiltonian_derivative, pso.psi*(pso.psi)', pso.Liouville_operator, pso.γ, 
                      pso.control_Hamiltonian, pso.control_coefficients, pso.times)
    f_ini = real(tr(pso.W*pinv(F)))
    f_list = [f_ini]
    println("initial value of Tr(WF^{-1}) is $(f_ini)")
    Tend = pso.times[end]        
    fit = 0.0
    if save_file==true
        for ei in 1:max_episodes
            #### train ####
            for pj in 1:particle_num
                rho = particles[pj].psi*(particles[pj].psi)'
                F_tp = CFIM(M, particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative, rho, particles[pj].Liouville_operator, particles[pj].γ, 
                                particles[pj].control_Hamiltonian, particles[pj].control_coefficients, particles[pj].times)
                f_now = 1.0/real(tr(particles[pj].W*pinv(F_tp)))
                if f_now > p_fit[pj]
                    p_fit[pj] = f_now
                    for di in 1:dim
                        pbest[di,pj] = particles[pj].psi[di]
                    end
                end
            end

            for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:dim
                        gbest[dj] = particles[pj].psi[dj]
                        velocity_best[dj] = velocity[dj, pj]
                    end
                end
            end  

            for pk in 1:particle_num
                psi_pre = zeros(ComplexF64, dim)
                for dk in 1:dim
                    psi_pre[dk] = particles[pk].psi[dk]
                    velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
                    particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
                end

                particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

                for dm in 1:dim
                    velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
                end
            end
            append!(f_list, (1.0/fit))
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
            open("state_pso_T$Tend.csv","w") do g
                writedlm(g, gbest)
            end
            open("f_pso_T$Tend.csv","w") do h
                writedlm(h, f_list)
            end
        end
        print("\e[2K")
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")

    else
        for ei in 1:max_episodes
            #### train ####
            for pj in 1:particle_num
                rho = particles[pj].psi*(particles[pj].psi)'
                F_tp = CFIM(M, particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative, rho, particles[pj].Liouville_operator, particles[pj].γ, 
                                particles[pj].control_Hamiltonian, particles[pj].control_coefficients, particles[pj].times)
                f_now = 1.0/real(tr(particles[pj].W*pinv(F_tp)))
                if f_now > p_fit[pj]
                    p_fit[pj] = f_now
                    for di in 1:dim
                        pbest[di,pj] = particles[pj].psi[di]
                    end
                end
            end

            for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:dim
                        gbest[dj] = particles[pj].psi[dj]
                        velocity_best[dj] = velocity[dj, pj]
                    end
                end
            end  

            for pk in 1:particle_num
                psi_pre = zeros(ComplexF64, dim)
                for dk in 1:dim
                    psi_pre[dk] = particles[pk].psi[dk]
                    velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
                    particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
                end

                particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

                for dm in 1:dim
                    velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
                end
            end
            append!(f_list, (1.0/fit))
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
        end
        open("state_pso_T$Tend.csv","w") do g
            writedlm(g, gbest)
        end
        open("f_pso_T$Tend.csv","w") do h
            writedlm(h, f_list)
        end
        print("\e[2K") 
        println("Iteration over, data saved.") 
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    end
    return nothing
end

function PSO_QFI(pso::StateOpt_PSO{T}, max_episodes::Vector{Int64}, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("state optimization")
    println("single parameter scenario")
    println("control algorithm: PSO")
    Random.seed!(sd)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
    velocity_best = zeros(ComplexF64, dim)
    p_fit = zeros(particle_num)

    # initialize
    for pj in 1:length(ini_particle)
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    qfi_ini = QFI_ori(pso.freeHamiltonian, pso.Hamiltonian_derivative[1], pso.psi*(pso.psi)', pso.Liouville_operator, pso.γ, 
                      pso.control_Hamiltonian, pso.control_coefficients, pso.times)
    f_list = [qfi_ini]
    println("initial QFI is $(qfi_ini)")
    Tend = pso.times[end]        
    fit = 0.0
    if save_file==true
        for ei in 1:max_episodes[1]
            #### train ####
            for pj in 1:particle_num
                rho = particles[pj].psi*(particles[pj].psi)'
                f_now = QFI_ori(particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative[1], rho, particles[pj].Liouville_operator, particles[pj].γ, 
                                particles[pj].control_Hamiltonian, particles[pj].control_coefficients, particles[pj].times)
                if f_now > p_fit[pj]
                    p_fit[pj] = f_now
                    for di in 1:dim
                        pbest[di,pj] = particles[pj].psi[di]
                    end
                end
            end

            for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:dim
                        gbest[dj] = particles[pj].psi[dj]
                        velocity_best[dj] = velocity[dj, pj]
                    end
                end
            end  

            for pk in 1:particle_num
                psi_pre = zeros(ComplexF64, dim)
                for dk in 1:dim
                    psi_pre[dk] = particles[pk].psi[dk]
                    velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
                    particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
                end

                particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

                for dm in 1:dim
                    velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
                end
            end
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            print("current QFI is $fit ($ei episodes) \r")
            open("state_pso_T$Tend.csv","w") do g
                writedlm(g, gbest)
            end
            open("f_pso_T$Tend.csv","w") do h
                writedlm(h, f_list)
            end
        end
        print("\e[2K")    
        println("Iteration over, data saved.")
        println("Final QFI is $fit")
    else
        for ei in 1:max_episodes[1]
            #### train ####
            for pj in 1:particle_num
                rho = particles[pj].psi*(particles[pj].psi)'
                f_now = QFI_ori(particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative[1], rho, particles[pj].Liouville_operator, particles[pj].γ, 
                                particles[pj].control_Hamiltonian, particles[pj].control_coefficients, particles[pj].times)
                if f_now > p_fit[pj]
                    p_fit[pj] = f_now
                    for di in 1:dim
                        pbest[di,pj] = particles[pj].psi[di]
                    end
                end
            end

            for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:dim
                        gbest[dj] = particles[pj].psi[dj]
                        velocity_best[dj] = velocity[dj, pj]
                    end
                end
            end  

            for pk in 1:particle_num
                psi_pre = zeros(ComplexF64, dim)
                for dk in 1:dim
                    psi_pre[dk] = particles[pk].psi[dk]
                    velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
                    particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
                end

                particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

                for dm in 1:dim
                    velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
                end
            end
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            print("current QFI is $fit ($ei episodes) \r")
        end
        open("state_pso_T$Tend.csv","w") do g
            writedlm(g, gbest)
        end
        open("f_pso_T$Tend.csv","w") do h
            writedlm(h, f_list)
        end
        print("\e[2K") 
        println("Iteration over, data saved.") 
        println("Final QFI is $fit")
    end
    return nothing
end

function PSO_QFIM(pso::StateOpt_PSO{T}, max_episodes::Vector{Int64}, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("state optimization")
    println("multiparameter scenario")
    println("control algorithm: PSO")
    Random.seed!(sd)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
    velocity_best = zeros(ComplexF64, dim)
    p_fit = zeros(particle_num)

    # initialize
    for pj in 1:length(ini_particle)
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    F = QFIM_ori(pso.freeHamiltonian, pso.Hamiltonian_derivative, pso.psi*(pso.psi)', pso.Liouville_operator, pso.γ, 
                      pso.control_Hamiltonian, pso.control_coefficients, pso.times)
    qfi_ini = real(tr(pso.W*pinv(F)))
    f_list = [qfi_ini]
    println("initial value of Tr(WF^{-1}) is $(qfi_ini)")
    Tend = pso.times[end]        
    fit = 0.0
    if save_file==true
        for ei in 1:max_episodes[1]
            #### train ####
            for pj in 1:particle_num
                rho = particles[pj].psi*(particles[pj].psi)'
                F_tp = QFIM_ori(particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative, rho, particles[pj].Liouville_operator, particles[pj].γ, 
                                particles[pj].control_Hamiltonian, particles[pj].control_coefficients, particles[pj].times)
                f_now = 1.0/real(tr(particles[pj].W*pinv(F_tp)))
                if f_now > p_fit[pj]
                    p_fit[pj] = f_now
                    for di in 1:dim
                        pbest[di,pj] = particles[pj].psi[di]
                    end
                end
            end

            for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:dim
                        gbest[dj] = particles[pj].psi[dj]
                        velocity_best[dj] = velocity[dj, pj]
                    end
                end
            end  

            for pk in 1:particle_num
                psi_pre = zeros(ComplexF64, dim)
                for dk in 1:dim
                    psi_pre[dk] = particles[pk].psi[dk]
                    velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
                    particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
                end

                particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

                for dm in 1:dim
                    velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
                end
            end
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, (1.0/fit))
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
            open("state_pso_T$Tend.csv","w") do g
                writedlm(g, gbest)
            end
            open("f_pso_T$Tend.csv","w") do h
                writedlm(h, f_list)
            end
        end
        print("\e[2K")    
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    else
        for ei in 1:max_episodes[1]
            #### train ####
            for pj in 1:particle_num
                rho = particles[pj].psi*(particles[pj].psi)'
                F_tp = QFIM_ori(particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative, rho, particles[pj].Liouville_operator, particles[pj].γ, 
                                particles[pj].control_Hamiltonian, particles[pj].control_coefficients, particles[pj].times)
                f_now = 1.0/real(tr(particles[pj].W*pinv(F_tp)))
                if f_now > p_fit[pj]
                    p_fit[pj] = f_now
                    for di in 1:dim
                        pbest[di,pj] = particles[pj].psi[di]
                    end
                end
            end

            for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:dim
                        gbest[dj] = particles[pj].psi[dj]
                        velocity_best[dj] = velocity[dj, pj]
                    end
                end
            end  

            for pk in 1:particle_num
                psi_pre = zeros(ComplexF64, dim)
                for dk in 1:dim
                    psi_pre[dk] = particles[pk].psi[dk]
                    velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
                    particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
                end

                particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

                for dm in 1:dim
                    velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
                end
            end
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, (1.0/fit))
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
        end
        open("state_pso_T$Tend.csv","w") do g
            writedlm(g, gbest)
        end
        open("f_pso_T$Tend.csv","w") do h
            writedlm(h, f_list)
        end
        print("\e[2K") 
        println("Iteration over, data saved.") 
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    end
    return nothing
end

function PSO_CFI(M, pso::StateOpt_PSO{T}, max_episodes::Vector{Int64}, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("state optimization")
    println("single parameter scenario")
    println("control algorithm: PSO")
    Random.seed!(sd)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
    velocity_best = zeros(ComplexF64, dim)
    p_fit = zeros(particle_num)

    # initialize
    for pj in 1:length(ini_particle)
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    f_ini = CFI(M, pso.freeHamiltonian, pso.Hamiltonian_derivative[1], pso.psi*(pso.psi)', pso.Liouville_operator, pso.γ, 
                      pso.control_Hamiltonian, pso.control_coefficients, pso.times)
    f_list = [f_ini]
    println("initial CFI is $(f_ini)")
    Tend = pso.times[end]        
    fit = 0.0
    if save_file==true
        for ei in 1:max_episodes[1]
            #### train ####
            for pj in 1:particle_num
                rho = particles[pj].psi*(particles[pj].psi)'
                f_now = CFI(M, particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative[1], rho, particles[pj].Liouville_operator, particles[pj].γ, 
                                particles[pj].control_Hamiltonian, particles[pj].control_coefficients, particles[pj].times)
                if f_now > p_fit[pj]
                    p_fit[pj] = f_now
                    for di in 1:dim
                        pbest[di,pj] = particles[pj].psi[di]
                    end
                end
            end

            for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:dim
                        gbest[dj] = particles[pj].psi[dj]
                        velocity_best[dj] = velocity[dj, pj]
                    end
                end
            end  

            for pk in 1:particle_num
                psi_pre = zeros(ComplexF64, dim)
                for dk in 1:dim
                    psi_pre[dk] = particles[pk].psi[dk]
                    velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
                    particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
                end

                particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

                for dm in 1:dim
                    velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
                end
            end
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            print("current CFI is $fit ($ei episodes) \r")
            open("state_pso_T$Tend.csv","w") do g
                writedlm(g, gbest)
            end
            open("f_pso_T$Tend.csv","w") do h
                writedlm(h, f_list)
            end
        end
        print("\e[2K")    
        println("Iteration over, data saved.")
        println("Final CFI is $fit")
    else
        for ei in 1:max_episodes[1]
            #### train ####
            for pj in 1:particle_num
                rho = particles[pj].psi*(particles[pj].psi)'
                f_now = CFI(M, particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative[1], rho, particles[pj].Liouville_operator, particles[pj].γ, 
                                particles[pj].control_Hamiltonian, particles[pj].control_coefficients, particles[pj].times)
                if f_now > p_fit[pj]
                    p_fit[pj] = f_now
                    for di in 1:dim
                        pbest[di,pj] = particles[pj].psi[di]
                    end
                end
            end

            for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:dim
                        gbest[dj] = particles[pj].psi[dj]
                        velocity_best[dj] = velocity[dj, pj]
                    end
                end
            end  

            for pk in 1:particle_num
                psi_pre = zeros(ComplexF64, dim)
                for dk in 1:dim
                    psi_pre[dk] = particles[pk].psi[dk]
                    velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
                    particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
                end

                particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

                for dm in 1:dim
                    velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
                end
            end
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, fit)
            print("current CFI is $fit ($ei episodes) \r")
        end
        open("state_pso_T$Tend.csv","w") do g
            writedlm(g, gbest)
        end
        open("f_pso_T$Tend.csv","w") do h
            writedlm(h, f_list)
        end
        print("\e[2K") 
        println("Iteration over, data saved.") 
        println("Final CFI is $fit")
    end
    return nothing
end

function PSO_CFIM(M, pso::StateOpt_PSO{T}, max_episodes::Vector{Int64}, particle_num, ini_particle, c0, c1, c2, v0, sd, save_file) where {T<: Complex}
    println("state optimization")
    println("multiparameter scenario")
    println("control algorithm: PSO")
    Random.seed!(sd)
    dim = length(pso.psi)
    particles = repeat(pso, particle_num)
    velocity = v0.*randn(ComplexF64, dim, particle_num)
    pbest = zeros(ComplexF64, dim, particle_num)
    gbest = zeros(ComplexF64, dim)
    velocity_best = zeros(ComplexF64, dim)
    p_fit = zeros(particle_num)

    # initialize
    for pj in 1:length(ini_particle)
        particles[pj].psi = [ini_particle[pj][i] for i in 1:dim]
    end

    for pj in (length(ini_particle)+1):(particle_num-1)
        r_ini = 2*rand(dim)-rand(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(dim)
        particles[pj].psi = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end

    F = CFIM(M, pso.freeHamiltonian, pso.Hamiltonian_derivative, pso.psi*(pso.psi)', pso.Liouville_operator, pso.γ, 
                      pso.control_Hamiltonian, pso.control_coefficients, pso.times)
    f_ini = real(tr(pso.W*pinv(F)))
    f_list = [f_ini]
    println("initial value of Tr(WF^{-1}) is $(f_ini)")
    Tend = pso.times[end]        
    fit = 0.0
    if save_file==true
        for ei in 1:max_episodes[1]
            #### train ####
            for pj in 1:particle_num
                rho = particles[pj].psi*(particles[pj].psi)'
                F_tp = CFIM(M, particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative, rho, particles[pj].Liouville_operator, particles[pj].γ, 
                                particles[pj].control_Hamiltonian, particles[pj].control_coefficients, particles[pj].times)
                f_now = 1.0/real(tr(particles[pj].W*pinv(F_tp)))
                if f_now > p_fit[pj]
                    p_fit[pj] = f_now
                    for di in 1:dim
                        pbest[di,pj] = particles[pj].psi[di]
                    end
                end
            end

            for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:dim
                        gbest[dj] = particles[pj].psi[dj]
                        velocity_best[dj] = velocity[dj, pj]
                    end
                end
            end  

            for pk in 1:particle_num
                psi_pre = zeros(ComplexF64, dim)
                for dk in 1:dim
                    psi_pre[dk] = particles[pk].psi[dk]
                    velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
                    particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
                end

                particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

                for dm in 1:dim
                    velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
                end
            end
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, (1.0/fit))
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
            open("state_pso_T$Tend.csv","w") do g
                writedlm(g, gbest)
            end
            open("f_pso_T$Tend.csv","w") do h
                writedlm(h, f_list)
            end
        end
        print("\e[2K")    
        println("Iteration over, data saved.")
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    else
        for ei in 1:max_episodes[1]
            #### train ####
            for pj in 1:particle_num
                rho = particles[pj].psi*(particles[pj].psi)'
                F_tp = CFIM(M, particles[pj].freeHamiltonian, particles[pj].Hamiltonian_derivative, rho, particles[pj].Liouville_operator, particles[pj].γ, 
                                particles[pj].control_Hamiltonian, particles[pj].control_coefficients, particles[pj].times)
                f_now = 1.0/real(tr(particles[pj].W*pinv(F_tp)))
                if f_now > p_fit[pj]
                    p_fit[pj] = f_now
                    for di in 1:dim
                        pbest[di,pj] = particles[pj].psi[di]
                    end
                end
            end

            for pj in 1:particle_num
                if p_fit[pj] > fit
                    fit = p_fit[pj]
                    for dj in 1:dim
                        gbest[dj] = particles[pj].psi[dj]
                        velocity_best[dj] = velocity[dj, pj]
                    end
                end
            end  

            for pk in 1:particle_num
                psi_pre = zeros(ComplexF64, dim)
                for dk in 1:dim
                    psi_pre[dk] = particles[pk].psi[dk]
                    velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand()*(pbest[dk, pk] - particles[pk].psi[dk]) + c2*rand()*(gbest[dk] - particles[pk].psi[dk])
                    particles[pk].psi[dk] = particles[pk].psi[dk] + velocity[dk, pk]
                end

                particles[pk].psi = particles[pk].psi/norm(particles[pk].psi)

                for dm in 1:dim
                    velocity[dm, pk] = particles[pk].psi[dm] - psi_pre[dm]
                end
            end
            if ei%max_episodes[2] == 0
                pso.psi = [gbest[i] for i in 1:dim]
                particles = repeat(pso, particle_num)
            end
            append!(f_list, (1.0/fit))
            print("current value of Tr(WF^{-1}) is $(1.0/fit) ($ei episodes) \r")
        end
        open("state_pso_T$Tend.csv","w") do g
            writedlm(g, gbest)
        end
        open("f_pso_T$Tend.csv","w") do h
            writedlm(h, f_list)
        end
        print("\e[2K") 
        println("Iteration over, data saved.") 
        println("Final value of Tr(WF^{-1}) is $(1.0/fit)")
    end
    return nothing
end
