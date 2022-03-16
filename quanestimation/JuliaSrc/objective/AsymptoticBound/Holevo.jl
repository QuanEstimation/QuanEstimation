function decomposition(A)
    C = bunchkaufman(A; check = false)
    R = sqrt(Array(C.D)) * C.U'C.P
    return R
end

function HCRB(
    ρ::Matrix{T},
    ∂ρ_∂x::Vector{Matrix{T}},
    C::Matrix{Float64},
    eps = 1e-6,
) where {T<:Complex}
    if length(∂ρ_∂x) == 1
        println(
            "In single parameter scenario, HCRB is equivalent to QFI. This function will return the value of QFI",
        )
        f = QFI(ρ, ∂ρ_∂x[1], eps)
        return f
    else
        Holevo_bound(ρ, ∂ρ_∂x, C; eps = eps)
    end
end

function Holevo_bound(
    ρ::Matrix{T},
    ∂ρ_∂x::Vector{Matrix{T}},
    C::Matrix{Float64};
    eps = eps_default,
) where {T<:Complex}

    dim = size(ρ)[1]
    num = dim * dim
    para_num = length(∂ρ_∂x)
    suN = suN_generator(dim) / sqrt(2)
    Lambda = [Matrix{ComplexF64}(I, dim, dim) / sqrt(2)]
    append!(Lambda, [suN[i] for i = 1:length(suN)])
    vec_∂ρ = [[0.0 for i = 1:num] for j = 1:para_num]

    for pa = 1:para_num
        for ra = 2:num
            vec_∂ρ[pa][ra] = (∂ρ_∂x[pa] * Lambda[ra]) |> tr |> real
        end
    end
    S = zeros(ComplexF64, num, num)
    for a = 1:num
        for b = 1:num
            S[a, b] = (Lambda[a] * Lambda[b] * ρ) |> tr
        end
    end

    accu = length(string(Int(ceil(1 / eps)))) - 1
    R = decomposition(round.(digits = accu, S))

    #=========optimization variables===========#
    V = Variable(para_num, para_num)
    X = Variable(num, para_num)
    #============add constraints===============#
    constraints = [[V X'*R'; R*X Matrix{Float64}(I, num, num)] ⪰ 0]
    for i = 1:para_num
        for j = 1:para_num
            if i == j
                constraints += [X[:, i]' * vec_∂ρ[j] == 1]
            else
                constraints += [X[:, i]' * vec_∂ρ[j] == 0]
            end
        end
    end
    problem = minimize(tr(C * V), constraints)
    Convex.solve!(problem, SCS.Optimizer(verbose = false))
    return evaluate(tr(C * V)), evaluate(X), evaluate(V)
end
