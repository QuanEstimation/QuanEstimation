function decomposition(A)
    C = bunchkaufman(A; check=false)
    R = sqrt(Array(C.D))*C.U'C.P
    return R
end

function Holevo_bound(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}, C::Matrix{Float64}, accuracy=1e-6) where {T<:Complex}
    if length(∂ρ_∂x) == 1
        println("In single parameter scenario, HCRB is equivalent to QFI. This function will return the value of QFI")
        f = QFI(ρ, ∂ρ_∂x[1], accuracy)
        return f
    else
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

        accu = length(string(Int(ceil(1/accuracy))))-1
        R = decomposition(round.(digits=accu, S))

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
        solve!(problem, SCS.Optimizer(verbose=false))
        return evaluate(tr(C*V)), evaluate(X), evaluate(V)
    end
end

function Holevo_bound_tgt(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}, C::Matrix{Float64}, accuracy=1e-6) where {T<:Complex}

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

    accu = length(string(Int(ceil(1/accuracy))))-1
    R = decomposition(round.(digits=accu, S))

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
    solve!(problem, SCS.Optimizer(verbose=false))
    return evaluate(tr(C*V)), evaluate(X), evaluate(V)
end

function obj_func(x::Val{:HCRB}, ρt, ∂ρt_∂x, W, M, accuracy)
    f, X, V = Holevo_bound_tgt(ρt, ∂ρt_∂x, W, accuracy)  
    return f 
end

function obj_func(x::Val{:HCRB}, system, M)
    ρt, ∂ρt_∂x = dynamics(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, 
                system.control_Hamiltonian, system.control_coefficients, system.tspan)
    f, X, V = Holevo_bound_tgt(ρt, ∂ρt_∂x, system.W, system.accuracy)  
    return f         
end

function obj_func(x::Val{:HCRB}, system, M, control_coeff)
    ρt, ∂ρt_∂x = dynamics(system.freeHamiltonian, system.Hamiltonian_derivative, system.ρ0, system.decay_opt, system.γ, 
                system.control_Hamiltonian, control_coeff, system.tspan)
    f, X, V = Holevo_bound_tgt(ρt, ∂ρt_∂x, system.W, system.accuracy)   
    return f 
end

function obj_func(x::Val{:HCRB_TimeIndepend_noiseless}, system, M)
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(system.freeHamiltonian, system.Hamiltonian_derivative, system.psi, system.tspan)
    f, X, V = Holevo_bound_tgt(ρt, ∂ρt_∂x, system.W, system.accuracy)   
    return f        
end

function obj_func(x::Val{:HCRB_TimeIndepend_noiseless}, system, M, psi)
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(system.freeHamiltonian, system.Hamiltonian_derivative, psi, system.tspan)
    f, X, V = Holevo_bound_tgt(ρt, ∂ρt_∂x, system.W, system.accuracy)     
    return f      
end

function obj_func(x::Val{:HCRB_TimeIndepend_noise}, system, M)
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(system.freeHamiltonian, system.Hamiltonian_derivative, system.psi, system.decay_opt, system.γ, system.tspan)
    f, X, V = Holevo_bound_tgt(ρt, ∂ρt_∂x, system.W, system.accuracy)   
    return f 
end

function obj_func(x::Val{:HCRB_TimeIndepend_noise}, system, M, psi)
    ρt, ∂ρt_∂x = dynamics_TimeIndepend(system.freeHamiltonian, system.Hamiltonian_derivative, psi, system.decay_opt, system.γ, system.tspan)
    f, X, V = Holevo_bound_tgt(ρt, ∂ρt_∂x, system.W, system.accuracy)   
    return f 
end
