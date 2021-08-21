function RunPSO(grape::Gradient{T};  particle_num= 10, c0=0.5, c1=0.5, c2=0.5,v0=0.1, sd=1234, episode=400) where {T<: Complex}
    println("PSO strategies")
    println("searching optimal controls with particle swarm optimization ")
    tnum = length(grape.times)
    ctrl_num = length(grape.control_Hamiltonian)
    particles = repeat(grape, particle_num)
    velocity = v0.*rand(ctrl_num, tnum, particle_num)|>SharedArray
    pbest = zeros(ctrl_num, tnum, particle_num)
    gbest = zeros(ctrl_num, tnum) |> SharedArray
    velocity_best = zeros(ctrl_num,tnum)
    p_fit = zeros(particle_num)
    qfi_ini = QFI(grape)
    println("initial QFI is $(qfi_ini)")
    fit_pre = 0.0        
    fit = 0.0
    for ei in 1:episode
        @inbounds @threads for pi in 1:particle_num
            propagate!(particles[pi])
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
        @inbounds @threads for pj in 1:particle_num
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
        @inbounds @threads for pk in 1:particle_num
            for dk in 1:ctrl_num
                @inbounds @threads for ck in 1:tnum
                    velocity[dk, ck, pk]  = c0*velocity[dk, ck, pk] + c1*rand()*(pbest[dk, ck, pk] - particles[pk].control_coefficients[dk][ck]) 
                                          + c2*rand()*(gbest[dk, ck] - particles[pk].control_coefficients[dk][ck])  
                    particles[pk].control_coefficients[dk][ck] = particles[pk].control_coefficients[dk][ck] + velocity[dk, ck, pk] 
                end
            end
        end
        # println(particles[1].control_coefficients[1][1])
        if ei == episode#abs(fit-fit_pre) < 1e-5
            grape.control_coefficients = gbest
            println("Final QFI is ", fit)
            return nothing
        end        
        fit_pre = fit
        print("current QFI is $fit ($ei epochs)    \r")
    end
end
function RunPSOAdam(grape::Gradient{T};  particle_num= 10, c0=0.5, c1=0.5, c2=0.5,v0=0.01, sd=1234, episode=400) where {T<: Complex}
    println("PSO strategies")
    println("searching optimal controls with particle swarm optimization ")
    tnum = length(grape.times)
    ctrl_num = length(grape.control_Hamiltonian)
    particles, velocity = repeat(grape, particle_num), [[v0*ones(tnum) for i in 1:ctrl_num] for j in 1:particle_num]
    pbest = [[zeros(tnum) for i in 1:ctrl_num] for j in 1:particle_num]
    gbest = [zeros(tnum) for i in 1:ctrl_num]
    velocity_best = [zeros(Float64, tnum) for i in 1:ctrl_num]
    p_fit = zeros(particle_num)
    qfi_ini = QFI(grape)
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
            grape.control_coefficients = gbest
            println("Final QFI is ", fit)
            return nothing
        end        
        fit_pre = fit
        print("current QFI is $fit ($ei epochs)    \r")
    end
end