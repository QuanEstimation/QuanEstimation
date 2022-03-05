function adaptMZI_online(x, p, rho0, a, output)
    
    N = size(a)[1] -1
    exp_ix = [exp(1.0im*xi) for xi in x]
    phi_span = range(-pi, stop=pi, length=length(x)) |> collect

    phi = 0.0
    a_res = [Matrix{ComplexF64}(I, (N+1)^2, (N+1)^2) for i in 1:length(x)]
    xout, y = [], []
    if output == "phi"
        for ei in 1:N-1
            println("The tunable phase is $phi ($ei episodes)")
            print("Please enter the experimental result: ")
            enter = readline()
            u = parse(Int64, enter)
            
            pyx = zeros(length(x))
            for xi in 1:length(x)
                a_res_tp = a_res[xi]*a_u(a, x[xi], phi, u)
                pyx[xi] = real(tr(rho0*a_res_tp'*a_res_tp))*(factorial(N-ei)/factorial(N))
                a_res[xi] = a_res_tp
            end
            
            M_res = zeros(length(phi_span))
            for mj in 1:length(phi_span)
                M1_res = trapz(x, pyx.*p)
                pyx0, pyx1 = zeros(length(x)), zeros(length(x))
                M2_res = 0.0
                for xj in 1:length(x)
                    a_res0 = a_res[xj]*a_u(a, x[xj], phi_span[mj], 0)
                    a_res1 = a_res[xj]*a_u(a, x[xj], phi_span[mj], 1)
                    pyx0[xj] = real(tr(rho0*a_res0'*a_res0))*(factorial(N-(ei+1))/factorial(N))
                    pyx1[xj] = real(tr(rho0*a_res1'*a_res1))*(factorial(N-(ei+1))/factorial(N))
                    M2_res = abs(trapz(x, pyx0.*p.*exp_ix))+abs(trapz(x, pyx1.*p.*exp_ix))
                end
                M_res[mj] = M2_res/M1_res
            end
            indx_m = findmax(M_res)[2]
            phi_update = phi_span[indx_m]
            
            append!(xout, phi)
            append!(y, u)
            phi = phi_update
        end
        println("The estimator of the unknown phase is $phi ")
        append!(xout, phi)
        open("xout.csv","w") do m
            writedlm(m, xout)
        end
        open("y.csv","w") do n
            writedlm(n, y)
        end
    else
        println("The initial tunable phase is $phi")
        for ei in 1:N-1
            print("Please enter the experimental result: ")
            enter = readline()
            u = parse(Int64, enter)
            
            pyx = zeros(length(x))
            for xi in 1:length(x)
                a_res_tp = a_res[xi]*a_u(a, x[xi], phi, u)
                pyx[xi] = real(tr(rho0*a_res_tp'*a_res_tp))*(factorial(N-ei)/factorial(N))
                a_res[xi] = a_res_tp
            end
            
            M_res = zeros(length(phi_span))
            for mj in 1:length(phi_span)
                M1_res = trapz(x, pyx.*p)
                pyx0, pyx1 = zeros(length(x)), zeros(length(x))
                M2_res = 0.0
                for xj in 1:length(x)
                    a_res0 = a_res[xj]*a_u(a, x[xj], phi_span[mj], 0)
                    a_res1 = a_res[xj]*a_u(a, x[xj], phi_span[mj], 1)
                    pyx0[xj] = real(tr(rho0*a_res0'*a_res0))*(factorial(N-(ei+1))/factorial(N))
                    pyx1[xj] = real(tr(rho0*a_res1'*a_res1))*(factorial(N-(ei+1))/factorial(N))
                    M2_res = abs(trapz(x, pyx0.*p.*exp_ix))+abs(trapz(x, pyx1.*p.*exp_ix))
                end
                M_res[mj] = M2_res/M1_res
            end
            indx_m = findmax(M_res)[2]
            phi_update = phi_span[indx_m]
            println("The adjustments of the feedback phase is $(abs(phi_update-phi)) ($ei episodes)")
            append!(xout, abs(phi_update-phi))
            append!(y, u)
    
            phi = phi_update
        end
        open("xout.csv","w") do m
            writedlm(m, xout)
        end
        open("y.csv","w") do n
            writedlm(n, y)
        end
    end
end

function adaptMZI_offline(delta_phi, x, p, rho0, a, comb, eps)
    N = size(a)[1] - 1
    exp_ix = [exp(1.0im*xi) for xi in x]
    
    M_res = zeros(length(comb))
    Threads.@threads for ui in 1:length(comb)
        u = comb[ui]
        phi = 0.0

        a_res = [Matrix{ComplexF64}(I, (N+1)^2, (N+1)^2) for i in 1:length(x)]
        for ei in 1:N-1
            phi = phi - (-1)^u[ei]*delta_phi[ei]
            for xi in 1:length(x)
                a_res[xi] = a_res[xi]*a_u(a, x[xi], phi, u[ei])
            end
        end

        pyx = zeros(length(x))
        for xj in 1:length(x)
            pyx[xj] = real(tr(rho0*a_res[xj]'*a_res[xj]))*(1/factorial(N))
        end
        M_res[ui] = abs(trapz(x, pyx.*p.*exp_ix))
    end
    return sum(M_res)
end

function a_u(a, x, phi, u)
    N = size(a)[1] - 1
    a_in = kron(a, Matrix(I, N+1, N+1))
    b_in = kron(Matrix(I, N+1, N+1), a)

    value = 0.5*(x-phi)+0.5*pi*u
    return a_in*sin(value) + b_in*cos(value)
end

function logarithmic(number, N)
    res = zeros(N)
    res_tp = number
    for i in 1:N
        res_tp = res_tp/2
        res[i] = res_tp
    end
    return res
end

function DE_DeltaPhiOpt(x, p, rho0, a, comb, popsize, ini_population, c, cr, seed, max_episode, eps)
    Random.seed!(seed)
    N = size(a)[1] - 1
    DeltaPhi = [zeros(N) for i in 1:popsize]
    # initialize
    res = logarithmic(2.0*pi, N)
    if length(ini_population) > popsize
        ini_population = [ini_population[i] for i in 1:popsize]
    end
    for pj in 1:length(ini_population)
        DeltaPhi[pj] = [ini_population[pj][i] for i in 1:N]
    end
    for pk in (length(ini_population)+1):popsize
        DeltaPhi[pk] = [res[i]+rand() for i in 1:N]
    end

    p_fit = [0.0 for i in 1:popsize]
    for pl in 1:N
        p_fit[pl] = adaptMZI_offline(DeltaPhi[pl], x, p, rho0, a, comb, eps)
    end
    
    f_ini = maximum(p_fit)
    f_list = [f_ini]
    for ei in 1:(max_episode-1)
        for pm in 1:popsize
            #mutations
            mut_num = sample(1:popsize, 3, replace=false)
            DeltaPhi_mut = [0.0 for i in 1:N]
            for ci in 1:N
                DeltaPhi_mut[ci] = DeltaPhi[mut_num[1]][ci]+c*(DeltaPhi[mut_num[2]][ci]-DeltaPhi[mut_num[3]][ci])
            end
            #crossover
            DeltaPhi_cross = [0.0 for i in 1:N]
            cross_int = sample(1:N, 1, replace=false)[1]
            for cj in 1:N
                rand_num = rand()
                if rand_num <= cr
                    DeltaPhi_cross[cj] = DeltaPhi_mut[cj]
                else
                    DeltaPhi_cross[cj] = DeltaPhi[pm][cj]
                end
                DeltaPhi_cross[cross_int] = DeltaPhi_mut[cross_int]
            end
            #selection
            for cm in 1:N
                DeltaPhi_cross[cm] = (x-> x < 0.0 ? 0.0 : x > pi ? pi : x)(DeltaPhi_cross[cm])
            end
            f_cross = adaptMZI_offline(DeltaPhi_cross, x, p, rho0, a, comb, eps)
            if f_cross > p_fit[pm]
                p_fit[pm] = f_cross
                for ck in 1:N
                    DeltaPhi[pm][ck] = DeltaPhi_cross[ck]
                end
            end
        end
        println(maximum(p_fit))
        println(DeltaPhi[findmax(p_fit)[2]])
        append!(f_list, maximum(p_fit))
    end
    return DeltaPhi[findmax(p_fit)[2]]
end

function PSO_DeltaPhiOpt(x, p, rho0, a, comb, particle_num, ini_particle, c0, c1, c2, seed, max_episode, eps)   
    Random.seed!(seed)
    N = size(a)[1] - 1
    n = size(a)[1]

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    DeltaPhi = [zeros(N) for i in 1:particle_num]
    velocity = [zeros(N) for i in 1:particle_num]
    # initialize
    res = logarithmic(2.0*pi, N)
    if length(ini_particle) > particle_num
        ini_particle = [ini_particle[i] for i in 1:particle_num]
    end
    for pj in 1:length(ini_particle)
        DeltaPhi[pj] = [ini_particle[pj][i] for i in 1:N]
    end
    for pk in (length(ini_particle)+1):particle_num
        DeltaPhi[pk] = [res[i]+rand() for i in 1:N]
    end
    for pl in 1:particle_num
        velocity[pl] = [0.1*rand() for i in 1:N]
    end
    
    pbest = [zeros(N) for i in 1:particle_num]
    gbest = zeros(N)
    fit = 0.0
    p_fit = [0.0 for i in 1:particle_num]
    f_list = []
    for ei in 1:(max_episode[1]-1)
        for pm in 1:particle_num
            f_now = adaptMZI_offline(DeltaPhi[pm], x, p, rho0, a, comb, eps)
            if f_now > p_fit[pm]
                p_fit[pm] = f_now
                for ci in 1:N
                    pbest[pm][ci] = DeltaPhi[pm][ci]
                end
            end
        end

        for pn in 1:particle_num
            if p_fit[pn] > fit
                fit = p_fit[pn]
                for cj in 1:N
                    gbest[cj] = pbest[pn][cj]
                end
            end 
        end

        for pa in 1:particle_num
            DeltaPhi_pre = [0.0 for i in 1:N]
            for ck in 1:N
                DeltaPhi_pre[ck] = DeltaPhi[pa][ck]
                velocity[pa][ck] = c0*velocity[pa][ck] + c1*rand()*(pbest[pa][ck] - DeltaPhi[pa][ck]) 
                                    + c2*rand()*(gbest[ck] - DeltaPhi[pa][ck])
                DeltaPhi[pa][ck] += velocity[pa][ck]
            end

            for cn in 1:N
                DeltaPhi[pa][cn] = (x-> x < 0.0 ? 0.0 : x > pi ? pi : x)(DeltaPhi[pa][cn])
                velocity[pa][cn] = DeltaPhi[pa][cn] - DeltaPhi_pre[cn]
            end
        end
        println(fit)
        println(gbest)
        append!(f_list, fit)
        if ei%max_episode[2] == 0
            for pb in 1:particle_num
                DeltaPhi[pb] = [gbest[i] for i in 1:N]
            end
        end
    end
    return gbest
end