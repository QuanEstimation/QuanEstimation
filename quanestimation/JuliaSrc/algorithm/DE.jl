#### control optimization ####
function update!(opt::ControlOpt, alg::DE, obj, dynamics, output)
    (; max_episode, p_num, ini_population, c, cr, rng) = alg
    ini_population = ini_population[1]
    ctrl_length = length(dynamics.data.ctrl[1])
    ctrl_num = length(dynamics.data.Hc)
    populations = repeat(dynamics, p_num)

    # initialization
    initial_ctrl!(opt, ini_population, populations, p_num, rng)

    dynamics_copy = set_ctrl(dynamics, [zeros(ctrl_length) for i = 1:ctrl_num])
    f_noctrl, f_comp = objective(obj, dynamics_copy)
    p_fit, p_out = zeros(p_num), zeros(p_num)
    for i = 1:p_num
        p_out[i], p_fit[i] = objective(obj, populations[i])
    end

    set_f!(output, p_out[1])
    set_buffer!(output, dynamics.data.ctrl)
    set_io!(output, f_noctrl, p_out[1])
    show(opt, output, obj)
    
    for ei = 1:(max_episode-1)
        for pj = 1:p_num
            #mutations
            mut_num = sample(1:p_num, 3, replace = false)
            ctrl_mut = [Vector{Float64}(undef, ctrl_length) for i = 1:ctrl_num]
            for ci = 1:ctrl_num
                for ti = 1:ctrl_length
                    ctrl_mut[ci][ti] =
                        populations[mut_num[1]].data.ctrl[ci][ti] +
                        c * (
                            populations[mut_num[2]].data.ctrl[ci][ti] -
                            populations[mut_num[3]].data.ctrl[ci][ti]
                        )
                end
            end
            #crossover 
            ctrl_cross = [Vector{Float64}(undef, ctrl_length) for i = 1:ctrl_num]
            for cj = 1:ctrl_num
                cross_int = sample(1:ctrl_length, 1, replace = false)[1]
                for tj = 1:ctrl_length
                    rand_num = rand(rng)
                    if rand_num <= cr
                        ctrl_cross[cj][tj] = ctrl_mut[cj][tj]
                    else
                        ctrl_cross[cj][tj] = populations[pj].data.ctrl[cj][tj]
                    end
                end
                ctrl_cross[cj][cross_int] = ctrl_mut[cj][cross_int]
            end
            #selection
            bound!(ctrl_cross, opt.ctrl_bound)
            dynamics_cross = set_ctrl(populations[pj], ctrl_cross)
            f_out, f_cross = objective(obj, dynamics_cross)
            if f_cross > p_fit[pj]
                p_fit[pj] = f_cross
                p_out[pj] = f_out
                for ck = 1:ctrl_num
                    for tk = 1:ctrl_length
                        populations[pj].data.ctrl[ck][tk] = ctrl_cross[ck][tk]
                    end
                end
            end
        end
        idx = findmax(p_fit)[2]
        set_f!(output, p_out[idx])
        set_buffer!(output, populations[idx].data.ctrl)
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

#### state optimization ####
function update!(opt::StateOpt, alg::DE, obj, dynamics, output)
    (; max_episode, p_num, ini_population, c, cr, rng) = alg
    ini_population = ini_population[1]
    dim = length(dynamics.data.ψ0)
    populations = repeat(dynamics, p_num)
    # initialization  
    initial_state!(ini_population, populations, p_num, rng)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    for i in 1:p_num
        p_out[i], p_fit[i] = objective(obj, populations[i])
    end

    set_f!(output, p_out[1])
    set_buffer!(output, transpose(dynamics.data.ψ0))
    set_io!(output, p_out[1])
    show(opt, output, obj)

    for ei in 1:(max_episode-1)
        for pj in 1:p_num
            #mutations
            mut_num = sample(1:p_num, 3, replace=false)
            state_mut = zeros(ComplexF64, dim)
            for ci in 1:dim
                state_mut[ci] = populations[mut_num[1]].data.ψ0[ci]+c*(populations[mut_num[2]].data.ψ0[ci]-populations[mut_num[3]].data.ψ0[ci])
            end
            #crossover
            state_cross = zeros(ComplexF64, dim)
            cross_int = sample(1:dim, 1, replace=false)[1]
            for cj in 1:dim
                rand_num = rand(rng)
                if rand_num <= cr
                    state_cross[cj] = state_mut[cj]
                else
                    state_cross[cj] = populations[pj].data.ψ0[cj]
                end
                state_cross[cross_int] = state_mut[cross_int]
            end
            psi_cross = state_cross/norm(state_cross)
            dynamics_cross = set_state(populations[pj], psi_cross)
            f_out, f_cross = objective(obj, dynamics_cross)
            #selection
            if f_cross > p_fit[pj]
                p_fit[pj] = f_cross
                p_out[pj] = f_out
                for ck in 1:dim
                    populations[pj].data.ψ0[ck] = psi_cross[ck]
                end
            end
        end
        idx = findmax(p_fit)[2]
        set_f!(output, p_out[idx])
        set_buffer!(output, transpose(populations[idx].data.ψ0))
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end
    
#### projective measurement optimization ####
function update!(opt::Mopt_Projection, alg::DE, obj, dynamics, output)
    (; max_episode, p_num, ini_population, c, cr, rng) = alg
    ini_population = ini_population[1]

    dim = size(dynamics.data.ρ0)[1]
    M_num = length(opt.C)

    populations = [[zeros(ComplexF64, dim) for j in 1:M_num] for i in 1:p_num]
    # initialization  
    initial_M!(ini_population, populations, dim, p_num, M_num, rng)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    for pj in 1:p_num
        M = [populations[pj][i]*(populations[pj][i])' for i in 1:M_num]
        obj_copy = set_M(obj, M)
        p_out[pj], p_fit[pj] = objective(obj_copy, dynamics)
    end

    obj_QFIM = QFIM_Obj(obj)
    f_opt, f_comp = objective(obj_QFIM, dynamics)

    M = [populations[1][i]*(populations[1][i])' for i in 1:M_num]
    set_f!(output, p_out[1])
    set_buffer!(output, M)
    set_io!(output, p_out[1], f_opt)
    show(opt, output, obj)

    for ei in 1:(max_episode-1)
        for pj in 1:p_num
            #mutations
            mut_num = sample(1:p_num, 3, replace=false)
            M_mut = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
            for ci in 1:M_num
                for ti in 1:dim
                    M_mut[ci][ti] = populations[mut_num[1]][ci][ti] + c*(populations[mut_num[2]][ci][ti]-
                                    populations[mut_num[3]][ci][ti])
                end
            end
            #crossover
            M_cross = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
            for cj in 1:M_num
                cross_int = sample(1:dim, 1, replace=false)[1]
                for tj in 1:dim
                    rand_num = rand(rng)
                    if rand_num <= cr
                        M_cross[cj][tj] = M_mut[cj][tj]
                    else
                        M_cross[cj][tj] = populations[pj][cj][tj]
                    end
                end
                M_cross[cj][cross_int] = M_mut[cj][cross_int]
            end
            # orthogonality and normalization 
            M_cross = gramschmidt(M_cross)
            M = [M_cross[i]*(M_cross[i])' for i in 1:M_num]
            obj_cross = set_M(obj, M)
            f_out, f_cross = objective(obj_cross, dynamics)
            #selection
            if f_cross > p_fit[pj]
                p_fit[pj] = f_cross
                p_out[pj] = f_out
                for ck in 1:M_num
                    for tk in 1:dim
                        populations[pj][ck][tk] = M_cross[ck][tk]
                    end
                end
            end
        end
        idx = findmax(p_fit)[2]
        M = [populations[idx][i]*(populations[idx][i])' for i in 1:M_num]
        set_f!(output, p_out[idx])
        set_buffer!(output, M)
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end 

#### find the optimal linear combination of a given set of POVM ####
function update!(opt::Mopt_LinearComb, alg::DE, obj, dynamics, output)
    (; max_episode, p_num, ini_population, c, cr, rng) = alg
    ini_population = ini_population[1]
    (; B, POVM_basis, M_num) = opt
    basis_num = length(POVM_basis)
    populations = [[zeros(basis_num) for j in 1:M_num] for i in 1:p_num]

    # initialization
    initial_LinearComb!(ini_population, populations, basis_num, M_num, p_num, rng)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    for pj in 1:p_num
        M = [sum([populations[pj][i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
        obj_copy = set_M(obj, M)
        p_out[pj], p_fit[pj] = objective(obj_copy, dynamics)
    end

    obj_QFIM = QFIM_Obj(obj)
    f_opt, f_comp = objective(obj_QFIM, dynamics)
    obj_POVM = set_M(obj, POVM_basis)
    f_povm, f_comp = objective(obj_POVM, dynamics)

    M = [sum([populations[1][i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
    set_f!(output, p_out[1])
    set_buffer!(output, M)
    set_io!(output, p_out[1], f_povm, f_opt)
    show(opt, output, obj)

    for ei in 1:(max_episode-1)
        for pj in 1:p_num
            #mutations
            mut_num = sample(1:p_num, 3, replace=false)
            M_mut = [Vector{Float64}(undef, basis_num) for i in 1:M_num]
            for ci in 1:M_num
                for ti in 1:basis_num
                    M_mut[ci][ti] = populations[mut_num[1]][ci][ti] + c*(populations[mut_num[2]][ci][ti]-populations[mut_num[3]][ci][ti])
                end
            end
            #crossover
            M_cross = [Vector{Float64}(undef, basis_num) for i in 1:M_num]
            for cj in 1:M_num
                cross_int = sample(1:basis_num, 1, replace=false)[1]
                for tj in 1:basis_num
                    rand_num = rand(rng)
                    if rand_num <= cr
                        M_cross[cj][tj] = M_mut[cj][tj]
                    else
                        M_cross[cj][tj] = populations[pj][cj][tj]
                    end
                end
                M_cross[cj][cross_int] = M_mut[cj][cross_int]
            end
    
            # normalize the coefficients 
            bound_LC_coeff!(M_cross)
            M = [sum([M_cross[i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
            obj_cross = set_M(obj, M)
            f_out, f_cross = objective(obj_cross, dynamics)
            #selection
            if f_cross > p_fit[pj]
                p_fit[pj] = f_cross
                p_out[pj] = f_out
                for ck in 1:M_num
                    for tk in 1:basis_num
                        populations[pj][ck][tk] = M_cross[ck][tk]
                    end
                end
            end
        end
        idx = findmax(p_fit)[2]
        M = [sum([populations[idx][i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
        set_f!(output, p_out[idx])
        set_buffer!(output, M)
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

#### find the optimal rotated measurement of a given set of POVM ####
function update!(opt::Mopt_Rotation, alg::DE, obj, dynamics, output)
    (; max_episode, p_num, ini_population, c, cr, rng) = alg
    ini_population = ini_population[1]
    (; s, POVM_basis, Lambda) = opt
    dim = size(dynamics.data.ρ0)[1]
    suN = suN_generator(dim)
    append!(Lambda, [Matrix{ComplexF64}(I,dim,dim)])
    append!(Lambda, [suN[i] for i in 1:length(suN)])
    M_num = length(POVM_basis)
    populations = [zeros(dim^2) for i in 1:p_num]
    # initialization  
    initial_Rotation!(ini_population, populations, dim, p_num, rng)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    for pj in 1:p_num
        U = rotation_matrix(populations[pj], Lambda)
        M = [U*POVM_basis[i]*U' for i in 1:M_num]
        obj_copy = set_M(obj, M)
        p_out[pj], p_fit[pj] = objective(obj_copy, dynamics)
    end

    obj_QFIM = QFIM_Obj(obj)
    f_opt, f_comp = objective(obj_QFIM, dynamics)
    obj_POVM = set_M(obj, POVM_basis)
    f_povm, f_comp = objective(obj_POVM, dynamics)

    U = rotation_matrix(populations[1], Lambda)
    M = [U*POVM_basis[i]*U' for i in 1:M_num]
    set_f!(output, p_out[1])
    set_buffer!(output, M)
    set_io!(output, p_out[1], f_povm, f_opt)
    show(opt, output, obj)

    for ei in 1:(max_episode-1)
        for pj in 1:p_num
            #mutations
            mut_num = sample(1:p_num, 3, replace=false)
            M_mut = Vector{Float64}(undef, dim^2)
            for ti in 1:dim^2
                M_mut[ti] = populations[mut_num[1]][ti] + c*(populations[mut_num[2]][ti]-populations[mut_num[3]][ti])
            end
    
            #crossover
            M_cross = Vector{Float64}(undef, dim^2)
            cross_int = sample(1:dim^2, 1, replace=false)[1]
            for tj in 1:dim^2
                rand_num = rand(rng)
                if rand_num <= cr
                    M_cross[tj] = M_mut[tj]
                else
                    M_cross[tj] = populations[pj][tj]
                end
            end
            M_cross[cross_int] = M_mut[cross_int]
    
            # normalize the coefficients 
            bound_rot_coeff!(M_cross)
            U = rotation_matrix(M_cross, Lambda)
            M = [U*POVM_basis[i]*U' for i in 1:M_num]
            obj_cross = set_M(obj, M)
            f_out, f_cross = objective(obj_cross, dynamics)
            #selection
            if f_cross > p_fit[pj]
                p_fit[pj] = f_cross
                p_out[pj] = f_out
                for tk in 1:dim^2
                    populations[pj][tk] = M_cross[tk]
                end
            end
        end
        idx = findmax(p_fit)[2]
        U = rotation_matrix(populations[idx], Lambda)
        M = [U*POVM_basis[i]*U' for i in 1:M_num]
        set_f!(output, p_out[idx])
        set_buffer!(output, M)
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

#### state and control optimization ####
function update!(opt::StateControlOpt, alg::DE, obj, dynamics, output)
    (; max_episode, p_num, ini_population, c, cr, rng) = alg
    psi0, ctrl0 = ini_population
    ctrl_length = length(dynamics.data.ctrl[1])
    ctrl_num = length(dynamics.data.Hc)
    dim = length(dynamics.data.ψ0)
    populations = repeat(dynamics, p_num)

    # initialization 
    initial_state!(psi0, populations, p_num, rng)
    initial_ctrl!(opt, ctrl0, populations, p_num, rng)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    for i in 1:p_num
        p_out[i], p_fit[i] = objective(obj, populations[i])
    end

    dynamics_copy = set_ctrl(dynamics, [zeros(ctrl_length) for i in 1:ctrl_num])
    f_noctrl, f_comp = objective(obj, dynamics_copy)

    set_f!(output, p_out[1])
    set_buffer!(output, transpose(populations[1].data.ψ0), populations[1].data.ctrl)
    set_io!(output, f_noctrl, p_out[1])
    show(opt, output, obj)

    for ei in 1:(max_episode-1)
        for pj in 1:p_num
            #mutations
            mut_num = sample(1:p_num, 3, replace = false)
            state_mut = zeros(ComplexF64, dim)
            for ci in 1:dim
                state_mut[ci] = populations[mut_num[1]].data.ψ0[ci] + c * (populations[mut_num[2]].data.ψ0[ci] - populations[mut_num[3]].data.ψ0[ci])
            end
            ctrl_mut = [Vector{Float64}(undef, ctrl_length) for i in 1:ctrl_num]
            for ci in 1:ctrl_num
                for ti in 1:ctrl_length
                    ctrl_mut[ci][ti] = populations[mut_num[1]].data.ctrl[ci][ti] +
                                       c * (populations[mut_num[2]].data.ctrl[ci][ti] -
                                            populations[mut_num[3]].data.ctrl[ci][ti])
                end
            end
            #crossover
            state_cross = zeros(ComplexF64, dim)
            cross_int1 = sample(1:dim, 1, replace = false)[1]
            for cj in 1:dim
                rand_num = rand(rng)
                if rand_num <= cr
                    state_cross[cj] = state_mut[cj]
                else
                    state_cross[cj] = populations[pj].data.ψ0[cj]
                end
                state_cross[cross_int1] = state_mut[cross_int1]
            end
            psi_cross = state_cross / norm(state_cross)
            ctrl_cross = [Vector{Float64}(undef, ctrl_length) for i in 1:ctrl_num]
            for cj in 1:ctrl_num
                cross_int2 = sample(1:ctrl_length, 1, replace = false)[1]
                for tj in 1:ctrl_length
                    rand_num = rand(rng)
                    if rand_num <= cr
                        ctrl_cross[cj][tj] = ctrl_mut[cj][tj]
                    else
                        ctrl_cross[cj][tj] = populations[pj].data.ctrl[cj][tj]
                    end
                end
                ctrl_cross[cj][cross_int2] = ctrl_mut[cj][cross_int2]
            end
            bound!(ctrl_cross, opt.ctrl_bound)
    
            dynamics_copy = set_state(populations[pj], psi_cross)
            dynamics_copy = set_ctrl(dynamics_copy, ctrl_cross)
            f_out, f_cross = objective(obj, dynamics_copy)
            #selection
            if f_cross > p_fit[pj]
                p_fit[pj] = f_cross
                p_out[pj] = f_out
                for ck in 1:dim
                    populations[pj].data.ψ0[ck] = psi_cross[ck]
                end
                for ck in 1:ctrl_num
                    for tk in 1:ctrl_length
                        populations[pj].data.ctrl[ck][tk] = ctrl_cross[ck][tk]
                    end
                end
            end
        end
        idx = findmax(p_fit)[2]
        set_f!(output, p_out[idx])
        set_buffer!(output, transpose(populations[idx].data.ψ0), populations[idx].data.ctrl)
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

#### state and measurement optimization ####
function update!(opt::StateMeasurementOpt, alg::DE, obj, dynamics, output)
    (; max_episode, p_num, ini_population, c, cr, rng) = alg
    psi0, measurement0 = ini_population
    dim = length(dynamics.data.ψ0)
    M_num = length(opt.C)
    populations = repeat(dynamics, p_num)

    # initialization 
    initial_state!(psi0, populations, p_num, rng)
    C_all = [[zeros(ComplexF64, dim) for j in 1:M_num] for i in 1:p_num]
    initial_M!(measurement0, C_all, dim, p_num, M_num, rng)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    for pj in 1:p_num
        M = [C_all[pj][i]*(C_all[pj][i])' for i in 1:M_num]
        obj_copy = set_M(obj, M)
        p_out[pj], p_fit[pj] = objective(obj_copy, populations[pj])
    end

    M = [C_all[1][i]*(C_all[1][i])' for i in 1:M_num]
    set_f!(output, p_out[1])
    set_buffer!(output, transpose(populations[1].data.ψ0), M)
    set_io!(output, p_out[1])
    show(opt, output, obj)

    for ei in 1:(max_episode-1)
        for pj in 1:p_num
            #mutations
            mut_num = sample(1:p_num, 3, replace = false)
            state_mut = zeros(ComplexF64, dim)
            for ci in 1:dim
                state_mut[ci] = populations[mut_num[1]].data.ψ0[ci] + c * (populations[mut_num[2]].data.ψ0[ci] - populations[mut_num[3]].data.ψ0[ci])
            end
    
            M_mut = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
            for ci in 1:M_num
                for ti in 1:dim
                    M_mut[ci][ti] = C_all[mut_num[1]][ci][ti] + c * (C_all[mut_num[2]][ci][ti] -
                                                                    C_all[mut_num[3]][ci][ti])
                end
            end
            #crossover
            state_cross = zeros(ComplexF64, dim)
            cross_int1 = sample(1:dim, 1, replace = false)[1]
            for cj in 1:dim
                rand_num = rand(rng)
                if rand_num <= cr
                    state_cross[cj] = state_mut[cj]
                else
                    state_cross[cj] = populations[pj].data.ψ0[cj]
                end
                state_cross[cross_int1] = state_mut[cross_int1]
            end
            psi_cross = state_cross / norm(state_cross)
    
            M_cross = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
            for cj in 1:M_num
                cross_int = sample(1:dim, 1, replace = false)[1]
                for tj in 1:dim
                    rand_num = rand(rng)
                    if rand_num <= cr
                        M_cross[cj][tj] = M_mut[cj][tj]
                    else
                        M_cross[cj][tj] = C_all[pj][cj][tj]
                    end
                end
                M_cross[cj][cross_int] = M_mut[cj][cross_int]
            end
            # orthogonality and normalization 
            M_cross = gramschmidt(M_cross)
            M = [M_cross[i] * (M_cross[i])' for i in 1:M_num]
            dynamics_cross = set_state(populations[pj], psi_cross)
            obj_cross = set_M(obj, M)
            f_out, f_cross = objective(obj_cross, dynamics_cross)
            #selection
            if f_cross > p_fit[pj]
                p_fit[pj] = f_cross
                p_out[pj] = f_out
                for ck in 1:dim
                    populations[pj].data.ψ0[ck] = psi_cross[ck]
                end
    
                for ck in 1:M_num
                    for tk in 1:dim
                       C_all[pj][ck][tk] = M_cross[ck][tk]
                    end
                end
            end
        end
        idx = findmax(p_fit)[2]
        M = [C_all[idx][i]*(C_all[idx][i])' for i in 1:M_num]
        set_f!(output, p_out[idx])
        set_buffer!(output, transpose(populations[idx].data.ψ0), M)
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

#### control and measurement optimization ####
function update!(opt::ControlMeasurementOpt, alg::DE, obj, dynamics, output)
    (; max_episode, p_num, ini_population, c, cr, rng) = alg
    ctrl0, measurement0 = ini_population
    dim = size(dynamics.data.ρ0)[1]
    ctrl_length = length(dynamics.data.ctrl[1])
    ctrl_num = length(dynamics.data.Hc)
    M_num = length(opt.C)
    populations = repeat(dynamics, p_num)

    # initialization 
    initial_ctrl!(opt, ctrl0, populations, p_num, rng)
    C_all = [[zeros(ComplexF64, dim) for j in 1:M_num] for i in 1:p_num]
    initial_M!(measurement0, C_all, dim, p_num, M_num, rng)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    for pj in 1:p_num
        M = [C_all[pj][i]*(C_all[pj][i])' for i in 1:M_num]
        obj_copy = set_M(obj, M)
        p_out[pj], p_fit[pj] = objective(obj_copy, populations[pj])
    end

    M = [C_all[1][i]*(C_all[1][i])' for i in 1:M_num]
    set_f!(output, p_out[1])
    set_buffer!(output, populations[1].data.ctrl, M)
    set_io!(output, p_out[1])
    show(opt, output, obj)

    for ei in 1:(max_episode-1)
        for pj in 1:p_num
            #mutations
            mut_num = sample(1:p_num, 3, replace = false)
            ctrl_mut = [Vector{Float64}(undef, ctrl_length) for i in 1:ctrl_num]
            for ci in 1:ctrl_num
                for ti in 1:ctrl_length
                    ctrl_mut[ci][ti] = populations[mut_num[1]].data.ctrl[ci][ti] +
                                       c * (populations[mut_num[2]].data.ctrl[ci][ti] -
                                            populations[mut_num[3]].data.ctrl[ci][ti])
                end
            end
    
            M_mut = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
            for ci in 1:M_num
                for ti in 1:dim
                    M_mut[ci][ti] = C_all[mut_num[1]][ci][ti] + c * (C_all[mut_num[2]][ci][ti] -
                                                                C_all[mut_num[3]][ci][ti])
                end
            end
    
            #crossover   
            ctrl_cross = [Vector{Float64}(undef, ctrl_length) for i in 1:ctrl_num]
            for cj in 1:ctrl_num
                cross_int2 = sample(1:ctrl_length, 1, replace = false)[1]
                for tj in 1:ctrl_length
                    rand_num = rand(rng)
                    if rand_num <= cr
                        ctrl_cross[cj][tj] = ctrl_mut[cj][tj]
                    else
                        ctrl_cross[cj][tj] = populations[pj].data.ctrl[cj][tj]
                    end
                end
                ctrl_cross[cj][cross_int2] = ctrl_mut[cj][cross_int2]
            end
            bound!(ctrl_cross, opt.ctrl_bound)
    
            M_cross = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
            for cj in 1:M_num
                cross_int = sample(1:dim, 1, replace = false)[1]
                for tj in 1:dim
                    rand_num = rand(rng)
                    if rand_num <= cr
                        M_cross[cj][tj] = M_mut[cj][tj]
                    else
                        M_cross[cj][tj] = C_all[pj][cj][tj]
                    end
                end
                M_cross[cj][cross_int] = M_mut[cj][cross_int]
            end
            # orthogonality and normalization 
            M_cross = gramschmidt(M_cross)
            M = [M_cross[i] * (M_cross[i])' for i in 1:M_num]
            dynamics_cross = set_ctrl(populations[pj], ctrl_cross)
            obj_cross = set_M(obj, M)
            f_out, f_cross = objective(obj_cross, dynamics_cross)
            #selection
            if f_cross > p_fit[pj]
                p_fit[pj] = f_cross
                p_out[pj] = f_out
                for ck in 1:ctrl_num
                    for tk in 1:ctrl_length
                        populations[pj].data.ctrl[ck][tk] = ctrl_cross[ck][tk]
                    end
                end
    
                for ck in 1:M_num
                    for tk in 1:dim
                        C_all[pj][ck][tk] = M_cross[ck][tk]
                    end
                end
            end
        end
        idx = findmax(p_fit)[2]
        M = [C_all[idx][i]*(C_all[idx][i])' for i in 1:M_num]
        set_f!(output, p_out[idx])
        set_buffer!(output, populations[idx].data.ctrl, M)
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

#### state, control and measurement optimization ####
function update!(opt::StateControlMeasurementOpt, alg::DE, obj, dynamics, output)
    (; max_episode, p_num, ini_population, c, cr, rng) = alg
    psi0, ctrl0, measurement0 = ini_population
    dim = length(dynamics.data.ψ0)
    ctrl_length = length(dynamics.data.ctrl[1])
    ctrl_num = length(dynamics.data.Hc)
    M_num = length(opt.C)
    populations = repeat(dynamics, p_num)

    # initialization 
    initial_state!(psi0, populations, p_num, rng)
    initial_ctrl!(opt, ctrl0, populations, p_num, rng)
    C_all = [[zeros(ComplexF64, dim) for j in 1:M_num] for i in 1:p_num]
    initial_M!(measurement0, C_all, dim, p_num, M_num, rng)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    for pj in 1:p_num
        M = [C_all[pj][i]*(C_all[pj][i])' for i in 1:M_num]
        obj_copy = set_M(obj, M)
        p_out[pj], p_fit[pj] = objective(obj_copy, populations[pj])
    end

    M = [C_all[1][i]*(C_all[1][i])' for i in 1:M_num]
    set_f!(output, p_out[1])
    set_buffer!(output, transpose(populations[1].data.ψ0), populations[1].data.ctrl, M)
    set_io!(output, p_out[1])
    show(opt, output, obj)

    for ei in 1:(max_episode-1)
        for pj in 1:p_num
            #mutations
            mut_num = sample(1:p_num, 3, replace = false)
            state_mut = zeros(ComplexF64, dim)
            for ci in 1:dim
                state_mut[ci] = populations[mut_num[1]].data.ψ0[ci] + c * (populations[mut_num[2]].data.ψ0[ci] - populations[mut_num[3]].data.ψ0[ci])
            end
            ctrl_mut = [Vector{Float64}(undef, ctrl_length) for i in 1:ctrl_num]
            for ci in 1:ctrl_num
                for ti in 1:ctrl_length
                    ctrl_mut[ci][ti] = populations[mut_num[1]].data.ctrl[ci][ti] +
                                       c * (populations[mut_num[2]].data.ctrl[ci][ti] -
                                            populations[mut_num[3]].data.ctrl[ci][ti])
                end
            end
            M_mut = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
            for ci in 1:M_num
                for ti in 1:dim
                    M_mut[ci][ti] = C_all[mut_num[1]][ci][ti] + c * (C_all[mut_num[2]][ci][ti] -
                                                                C_all[mut_num[3]][ci][ti])
                end
            end
            #crossover
            state_cross = zeros(ComplexF64, dim)
            cross_int1 = sample(1:dim, 1, replace = false)[1]
            for cj in 1:dim
                rand_num = rand(rng)
                if rand_num <= cr
                    state_cross[cj] = state_mut[cj]
                else
                    state_cross[cj] = populations[pj].data.ψ0[cj]
                end
                state_cross[cross_int1] = state_mut[cross_int1]
            end
            psi_cross = state_cross / norm(state_cross)
            ctrl_cross = [Vector{Float64}(undef, ctrl_length) for i in 1:ctrl_num]
            for cj in 1:ctrl_num
                cross_int2 = sample(1:ctrl_length, 1, replace = false)[1]
                for tj in 1:ctrl_length
                    rand_num = rand(rng)
                    if rand_num <= cr
                        ctrl_cross[cj][tj] = ctrl_mut[cj][tj]
                    else
                        ctrl_cross[cj][tj] = populations[pj].data.ctrl[cj][tj]
                    end
                end
                ctrl_cross[cj][cross_int2] = ctrl_mut[cj][cross_int2]
            end
            bound!(ctrl_cross, opt.ctrl_bound)
    
            M_cross = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
            for cj in 1:M_num
                cross_int = sample(1:dim, 1, replace = false)[1]
                for tj in 1:dim
                    rand_num = rand(rng)
                    if rand_num <= cr
                        M_cross[cj][tj] = M_mut[cj][tj]
                    else
                        M_cross[cj][tj] = C_all[pj][cj][tj]
                    end
                end
                M_cross[cj][cross_int] = M_mut[cj][cross_int]
            end
            # orthogonality and normalization 
            M_cross = gramschmidt(M_cross)
            M = [M_cross[i] * (M_cross[i])' for i in 1:M_num]
            dynamics_cross = set_state(populations[pj], psi_cross)
            dynamics_cross = set_ctrl(dynamics_cross, ctrl_cross)
            obj_cross = set_M(obj, M)
            f_out, f_cross = objective(obj_cross, dynamics_cross)
            #selection
            if f_cross > p_fit[pj]
                p_fit[pj] = f_cross
                p_out[pj] = f_out
                for ck in 1:dim
                    populations[pj].data.ψ0[ck] = psi_cross[ck]
                end
                for ck in 1:ctrl_num
                    for tk in 1:ctrl_length
                        populations[pj].data.ctrl[ck][tk] = ctrl_cross[ck][tk]
                    end
                end
                for ck in 1:M_num
                    for tk in 1:dim
                        C_all[pj][ck][tk] = M_cross[ck][tk]
                    end
                end
            end
        end
        idx = findmax(p_fit)[2]
        M = [C_all[idx][i]*(C_all[idx][i])' for i in 1:M_num]
        set_f!(output, p_out[idx])
        set_buffer!(output, transpose(populations[idx].data.ψ0), populations[idx].data.ctrl, M)
        set_io!(output, p_out[idx], ei)
        show(output, obj)
        end
    set_io!(output, output.f_list[end])
end
