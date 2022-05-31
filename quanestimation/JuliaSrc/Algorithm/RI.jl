function update!(opt::StateOpt, alg::RI, obj, dynamics, output)
    (; max_episode) = alg

    rho, drho = evolve(dynamics)
    f = QFIM(rho, drho)

    set_f!(output, f[1,1])
    set_buffer!(output, transpose(dynamics.data.ψ0))
    set_io!(output, f[1,1])
    show(opt, output, obj)

    f_list = [f[1,1]]
    idx = 0
    ## single-parameter scenario
    for ei in 1:(max_episode-1)
        rho, drho = evolve(dynamics)
        f, LD = QFIM(rho, drho, exportLD=true)
        M1 = d_DualMap(LD[1], dynamics.data.K, dynamics.data.dK)
        M2 = DualMap(LD[1]*LD[1], dynamics.data.K)
        M = 2*M1[1] - M2
        value, vec = eigen(M)
        val, idx = findmax(real(value))
        psi0 = vec[:, idx]
        dynamics.data.ψ0 = psi0

        set_f!(output, f[1,1])
        set_buffer!(output, transpose(dynamics.data.ψ0))
        set_io!(output, f[1,1], ei)
        show(output, obj)
    end
    set_io!(output, f[1,1])
end

function DualMap(L, K)
    return [Ki'*L*Ki for Ki in K] |> sum
end

function d_DualMap(L, K, dK)
    K_num = length(K)
    para_num = length(dK[1])
    Lambda = [[dK[i][j]' * L * K[i] + K[i]' * L * dK[i][j] for i in 1:K_num] |> sum for j in 1:para_num]
    return Lambda
end
