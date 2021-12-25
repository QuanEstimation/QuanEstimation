function decomposition(A)
    C = bunchkaufman(A; check=false)
    R = sqrt(Array(C.D))*C.U'C.P
    return R
end

function Holevo_bound(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}, C::Matrix{Float64}) where {T<:Complex}

    dim = size(ρ)[1]
    num = dim*dim
    para_num = length(∂ρ_∂x)
    suN = suN_generator(dim)/sqrt(2)
    Lambda = [Matrix{ComplexF64}(I,dim,dim)/sqrt(2)]
    append!(Lambda, [suN[i] for i in 1:length(suN)])
    vec_∂ρ = [[0.0 for i in 1:num] for j in 1:para_num]
    
    for pa in 1:para_num
        for ra in 2:num
            vec_∂ρ[pa][ra] = (∂ρ_∂x[pa]*Lambda[ra]) |> tr |> real
        end
    end
    S = zeros(ComplexF64, num, num)
    for a in 1:num
        for b in 1:num
            S[a, b] = (Lambda[a]*Lambda[b]*ρ) |> tr
        end
    end
    R = decomposition(S)

    #=========optimization variables===========#
    V = Variable(para_num, para_num)
    X = Variable(num, para_num)
    #============add constraints===============#
    constraints = [[V X'*R'; R*X Matrix{Float64}(I,num,num)] ⪰ 0 ]
    for i in 1:para_num
        for j in 1:para_num
            if i == j
                constraints += [X[:,i]'*vec_∂ρ[j] == 1]
            else
                constraints += [X[:,i]'*vec_∂ρ[j] == 0]
            end
        end
    end
    problem = minimize(tr(C*V), constraints)
    solve!(problem, SCS.Optimizer)
end
