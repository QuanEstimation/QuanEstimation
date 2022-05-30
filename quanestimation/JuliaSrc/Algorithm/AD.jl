#### control optimization ####
function update!(opt::ControlOpt, alg::AbstractautoGRAPE, obj, dynamics, output)
    (; max_episode) = alg
    ctrl_length = length(dynamics.data.ctrl[1])
    ctrl_num = length(dynamics.data.Hc)

    dynamics_copy = set_ctrl(dynamics, [zeros(ctrl_length) for i = 1:ctrl_num])
    f_noctrl, f_comp = objective(obj, dynamics_copy)
    f_ini, f_comp = objective(obj, dynamics)

    set_f!(output, f_ini)
    set_buffer!(output, dynamics.data.ctrl)
    set_io!(output, f_noctrl, f_ini)
    show(opt, output, obj)

    for ei = 1:(max_episode-1)
        δ = Flux.gradient(() -> objective(obj, dynamics)[2], Flux.Params([dynamics.data.ctrl]))
        update_ctrl!(alg, obj, dynamics, δ[dynamics.data.ctrl])
        bound!(dynamics.data.ctrl, opt.ctrl_bound)
        f_out, f_now = objective(obj, dynamics)

        set_f!(output, f_out)
        set_buffer!(output, dynamics.data.ctrl)
        set_io!(output, f_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

function update_ctrl!(alg::autoGRAPE_Adam, obj, dynamics, δ)
    (; epsilon, beta1, beta2) = alg
    for ci in 1:length(δ)
        mt, vt = 0.0, 0.0
        for ti in 1:length(δ[1])
            dynamics.data.ctrl[ci][ti], mt, vt = Adam(δ[ci][ti], ti, 
            dynamics.data.ctrl[ci][ti], mt, vt, epsilon, beta1, beta2, obj.eps)
        end
    end
end

function update_ctrl!(alg::autoGRAPE, obj, dynamics, δ)
    dynamics.data.ctrl += alg.epsilon*δ
end

#### state optimization ####
function update!(opt::StateOpt, alg::AbstractAD, obj, dynamics, output)
    (; max_episode) = alg
    f_ini, f_comp = objective(obj, dynamics)
    set_f!(output, f_ini)
    set_buffer!(output, transpose(dynamics.data.ψ0))
    set_io!(output, f_ini)
    show(opt, output, obj)
    for ei in 1:(max_episode-1)
        δ = Flux.gradient(() -> objective(obj, dynamics)[2], Flux.Params([dynamics.data.ψ0]))
        update_state!(alg, obj, dynamics, δ[dynamics.data.ψ0])
        dynamics.data.ψ0 = dynamics.data.ψ0/norm(dynamics.data.ψ0)
        f_out, f_now = objective(obj, dynamics)
        set_f!(output, f_out)
        set_buffer!(output, transpose(dynamics.data.ψ0))
        set_io!(output, f_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

function update_state!(alg::AD_Adam, obj, dynamics, δ)
    (; epsilon, beta1, beta2) = alg
    mt, vt = 0.0, 0.0
    for ti in 1:length(δ)
        dynamics.data.ψ0[ti], mt, vt = Adam(δ[ti], ti, dynamics.data.ψ0[ti], mt, vt, epsilon, beta1, beta2, obj.eps)
    end
end

function update_state!(alg::AD, obj, dynamics, δ)
    dynamics.data.ψ0 += alg.epsilon*δ
end

#### find the optimal linear combination of a given set of POVM ####
function update!(opt::Mopt_LinearComb, alg::AbstractAD, obj, dynamics, output)
    (; max_episode) = alg
    (; POVM_basis, M_num) = opt
    rng = MersenneTwister(1234)
    basis_num = length(POVM_basis)

    bound_LC_coeff!(opt.B, rng)
    M = [sum([opt.B[i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
    obj_QFIM = QFIM_obj(obj)
    f_opt, f_comp = objective(obj_QFIM, dynamics)
    obj_POVM = set_M(obj, POVM_basis)
    f_povm, f_comp = objective(obj_POVM, dynamics)
    obj_copy = set_M(obj, M)
    f_ini, f_comp = objective(obj_copy, dynamics)
    set_f!(output, f_ini)
    set_buffer!(output, M)
    set_io!(output, f_ini, f_povm, f_opt)
    show(opt, output, obj)
    for ei in 1:(max_episode-1)
        δ = Flux.gradient(() -> objective(opt, obj, dynamics)[2], Flux.Params([opt.B]))
        update_M!(opt, alg, obj, δ[opt.B])
        bound_LC_coeff!(opt.B, rng)
        M = [sum([opt.B[i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
        obj_copy = set_M(obj, M)
        f_out, f_now = objective(obj_copy, dynamics)
        set_f!(output, f_out)
        set_buffer!(output, M)
        set_io!(output, f_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

function update_M!(opt::Mopt_LinearComb, alg::AD_Adam, obj, δ)
    (; epsilon, beta1, beta2) = alg
    for ci in 1:length(δ)
        mt, vt = 0.0, 0.0
        for ti in 1:length(δ[1])
            opt.B[ci][ti], mt, vt = Adam(δ[ci][ti], ti, opt.B[ci][ti], mt, vt, epsilon, beta1, beta2, obj.eps)
        end
    end
end

function update_M!(opt::Mopt_LinearComb, alg::AD, obj, δ)
    opt.B += alg.epsilon*δ
end

#### find the optimal rotated measurement of a given set of POVM ####
function update!(opt::Mopt_Rotation, alg::AbstractAD, obj, dynamics, output)
    (; max_episode) = alg
    (; POVM_basis) = opt
    dim = size(dynamics.data.ρ0)[1]
    M_num = length(POVM_basis)
    suN = suN_generator(dim)
    opt.Lambda = Matrix{ComplexF64}[]
    append!(opt.Lambda, [Matrix{ComplexF64}(I,dim,dim)])
    append!(opt.Lambda, [suN[i] for i in 1:length(suN)])

    # if ismissing(Lambda)
    #     opt.Lambda = Matrix{ComplexF64}[]
    #     append!(opt.Lambda, [Matrix{ComplexF64}(I,dim,dim)])
    #     append!(opt.Lambda, [suN[i] for i in 1:length(suN)])
    # end
    
    U = rotation_matrix(opt.s, opt.Lambda)
    M = [U*POVM_basis[i]*U' for i in 1:M_num]
    obj_QFIM = QFIM_obj(obj)
    f_opt, f_comp = objective(obj_QFIM, dynamics)
    obj_POVM = set_M(obj, POVM_basis)
    f_povm, f_comp = objective(obj_POVM, dynamics)
    obj_copy = set_M(obj, M)
    f_ini, f_comp = objective(obj_copy, dynamics)
    set_f!(output, f_ini)
    set_buffer!(output, M)
    set_io!(output, f_ini, f_povm, f_opt)
    show(opt, output, obj)
    for ei in 1:(max_episode-1)
        δ = Flux.gradient(() -> objective(opt, obj, dynamics)[2], Flux.Params([opt.s]))
        update_M!(opt, alg, obj, δ[opt.s])
        bound_rot_coeff!(opt.s)
        U = rotation_matrix(opt.s, opt.Lambda)
        M = [U*POVM_basis[i]*U' for i in 1:M_num]
        obj_copy = set_M(obj, M)
        f_out, f_now = objective(obj_copy, dynamics)
        set_f!(output, f_out)
        set_buffer!(output, M)
        set_io!(output, f_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

function update_M!(opt::Mopt_Rotation, alg::AD_Adam, obj, δ)
    (; epsilon, beta1, beta2) = alg
    mt, vt = 0.0, 0.0
    for ti in 1:length(δ)
        opt.s[ti], mt, vt = Adam(δ[ti], ti, opt.s[ti], mt, vt, epsilon, beta1, beta2, obj.eps)
    end
end

function update_M!(opt::Mopt_Rotation, alg::AD, obj, δ)
    opt.s += alg.epsilon*δ
end

#### state abd control optimization ####
function update!(opt::StateControlOpt, alg::AbstractAD, obj, dynamics, output)
    (; max_episode) = alg
    ctrl_length = length(dynamics.data.ctrl[1])
    ctrl_num = length(dynamics.data.Hc)

    dynamics_copy = set_ctrl(dynamics, [zeros(ctrl_length) for i = 1:ctrl_num])
    f_noctrl, f_comp = objective(obj, dynamics_copy)
    f_ini, f_comp = objective(obj, dynamics)

    set_f!(output, f_ini)
    set_buffer!(output, transpose(dynamics.data.ψ0), dynamics.data.ctrl)
    set_io!(output, f_noctrl, f_ini)
    show(opt, output, obj)

    for ei = 1:(max_episode-1)
        δ = Flux.gradient(() -> objective(obj, dynamics)[2], Flux.Params([dynamics.data.ψ0, dynamics.data.ctrl]))
        update_state!(alg, obj, dynamics, δ[dynamics.data.ψ0])
        update_ctrl!(alg, obj, dynamics, δ[dynamics.data.ctrl])
        bound!(dynamics.data.ctrl, opt.ctrl_bound)
        dynamics.data.ψ0 = dynamics.data.ψ0/norm(dynamics.data.ψ0)
        f_out, f_now = objective(obj, dynamics)

        set_f!(output, f_out)
        set_buffer!(output, transpose(dynamics.data.ψ0), dynamics.data.ctrl)
        set_io!(output, f_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

function update_ctrl!(alg::AD_Adam, obj, dynamics, δ)
    (; epsilon, beta1, beta2) = alg
    for ci in 1:length(δ)
        mt, vt = 0.0, 0.0
        for ti in 1:length(δ[1])
            dynamics.data.ctrl[ci][ti], mt, vt = Adam(δ[ci][ti], ti, 
            dynamics.data.ctrl[ci][ti], mt, vt, epsilon, beta1, beta2, obj.eps)
        end
    end
end

function update_ctrl!(alg::AD, obj, dynamics, δ)
    dynamics.data.ctrl += alg.epsilon*δ
end
