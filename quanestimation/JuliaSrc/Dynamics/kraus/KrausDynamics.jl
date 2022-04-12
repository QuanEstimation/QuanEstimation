#### evolution of pure states under time-independent Hamiltonian without noise and controls ####
function evolve(dynamics::Kraus{ket})
    (; K, dK, ψ0) = dynamics.data
    ρ0 = ψ0 * ψ0'
    K_num = length(K)
    para_num = length(dK[1])
    ρ = [K[i] * ρ0 * K[i]' for i in 1:K_num] |> sum
    dρ = [[dK[i][j] * ρ0 * K[i]' + K[i] * ρ0 * dK[i][j]' for i in 1:K_num] |> sum for j in 1:para_num]

    ρ, dρ
end

#### evolution of density matrix under time-independent Hamiltonian without noise and controls ####
function evolve(dynamics::Kraus{dm})
    (; K, dK, ρ0) = dynamics.data
    K_num = length(K)
    para_num = length(dK[1])
    ρ = [K[i] * ρ0 * K[i]' for i in 1:K_num] |> sum
    dρ = [[dK[i][j] * ρ0 * K[i]' + K[i] * ρ0 * dK[i][j]' for i in 1:K_num] |> sum for j in 1:para_num]
    
    ρ, dρ
end
