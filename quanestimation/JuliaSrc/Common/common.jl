function RunMixed( grape, particle_num= 10, c0=0.5, c1=0.5, c2=0.5, prec_rough = 0.1, sd=1234, episode=400)
    println("Combined strategies")
    println("searching initial controls with particle swarm optimization ")
    tnum = length(grape.times)
    ctrl_num = length(grape.control_Hamiltonian)
    particles, velocity = repeat(grape, particle_num), [[10*ones(tnum) for i in 1:ctrl_num] for j in 1:particle_num]
    pbest = [[zeros(tnum) for i in 1:ctrl_num] for j in 1:particle_num]
    gbest = [zeros(tnum) for i in 1:ctrl_num]
    velocity_best = [zeros(Float64, tnum) for i in 1:ctrl_num]
    p_fit = zeros(particle_num)
    qfi_ini = QFI(grape)
    println("initial QFI is $(qfi_ini)")
    for ei in 1:episode
        fit_pre = 0.0
        fit = 0.0
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
                    velocity[pk][dk][ck]  = c0*velocity[pk][dk][ck] + c1*rand()*(pbest[pk][dk][ck] - particles[pk].control_coefficients[dk][ck]) 
                                          + c2*rand()*(gbest[dk][ck] - particles[pk].control_coefficients[dk][ck])  
                    particles[pk].control_coefficients[dk][ck] = particles[pk].control_coefficients[dk][ck] + velocity[pk][dk][ck]
                end
            end
        end
        fit_pre = fit
        if abs(fit-fit_pre) < prec_rough
            grape.control_coefficients = gbest
            println("PSO strategy finished, switching to GRAPE")
            Run(grape)
            return nothing
        end
        print("current QFI is $fit ($ei epochs)    \r")
    end
end

function RunODE(grape)
    println("quantum parameter estimation")
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter estimation scenario")
        qfi_ini = QFI(grape)
        qfi_list = [qfi_ini]
        println("initial QFI is $(qfi_ini)")
        gradient_QFIM_ODE!(grape)
        while true
            qfi_now = QFI(grape)
            gradient_QFIM_ODE!(grape)
            if  0 < (qfi_now - qfi_ini) < 1e-4
                println("\n Iteration over, data saved.")
                println("Final QFI is ", qfi_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "qfi", qfi_list)
                break
            else
                qfi_ini = qfi_now
                append!(qfi_list,qfi_now)
                print("current QFI is ", qfi_now, " ($(qfi_list|>length) epochs)    \r")
            end
        end
    else
        println("multiple parameters estimation scenario")
        f_ini =1/(grape |> QFIM |> inv |> tr)
        f_list = [f_ini]
        println("initial 1/tr(F^-1) is $(f_ini)")
        gradient_QFIM!(grape)
        while true
            f_now = 1/(grape |> QFIM |> inv |> tr)
            gradient_QFIM!(grape)
            if  0< f_now - f_ini < 1e-4
                println("\n Iteration over, data saved.")
                println("Final 1/tr(F^-1) is ", f_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "f", f_list)
                break
            else
                f_ini = f_now
                append!(f_list,f_now)
                print("current 1/tr(F^-1) is ", f_now, " ($(f_list|>length) epochs)    \r")
            end
        end
    end
end 
