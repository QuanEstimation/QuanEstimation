destroy(N) = diagm(1 => [1/sqrt(n) for n in 1:N-1])

bases(dim; T=ComplexF64) = [e for e in I(dim).|>T|>eachrow]

function vec2mat(x::Vector{T}) where {T <: Number}
    reshape(x, x |> length |> sqrt |> Int, :)  
end

function vec2mat(x)
    vec2mat.(x)
end

function vec2mat(x::Matrix)
    throw(ErrorException("vec2mating a matrix of size $(size(x))"))
end

unzip(X) = map(x->getfield.(X, x), fieldnames(eltype(X)))

function Base.repeat(system, N)
    [deepcopy(system) for i in 1:N]
end

function Base.repeat(system, M, N)
    reshape(repeat(system, M * N), M, N)
end

function filterZeros!(x::Matrix{T}) where {T <: Complex}
    x[abs.(x) .< eps()] .= zero(T)
    x
end
function filterZeros!(x) 
    filterZeros!.(x)
end

function filterZeros(x::AbstractVecOrMat{T}) where T<:Number
    [x+1≈1 ? zero(T) : x for x in x]
end

function t2Num(t0, dt, t)
    Int(round((t - t0) / dt)) + 1 
end

function basis(dim, si, ::T)::Array{T} where {T <: Complex}
    result = zeros(T, dim)
    result[si] = 1.0
    result
end

function suN_generatorU(n, k)
    tmp1, tmp2 = ceil((1+sqrt(1+8k))/2), ceil((-1+sqrt(1+8k))/2) 
    i = k - tmp2*(tmp2-1)/2 |> Int
    j =  tmp1 |> Int
    return sparse([i, j], [j,i], [1, 1], n, n)
end

function suN_generatorV(n, k)
    tmp1, tmp2 = ceil((1+sqrt(1+8k))/2), ceil((-1+sqrt(1+8k))/2) 
    i = k - tmp2*(tmp2-1)/2 |> Int
    j =  tmp1 |> Int 
    return sparse([i, j], [j,i], [-im, im], n, n)
end

function suN_generatorW(n, k)
    diagw = spzeros(n)
    diagw[1:k] .=1
    diagw[k+1] = -k
    return spdiagm(n,n,diagw)
end

function suN_generator(n)
    result = Vector{SparseMatrixCSC{ComplexF64, Int64}}(undef, n^2-1)
    idx = 2
    itr = 1

    for i in 1:n-1
       idx_t = idx
       while idx_t > 0
            result[itr] = iseven(idx_t) ? suN_generatorU(n, (i*(i-1)+idx-idx_t+2)/2) : suN_generatorV(n, (i*(i-1)+idx-idx_t+1)/2)
            itr += 1
            idx_t -= 1
       end
       result[itr] = sqrt(2/(i+i*i))*suN_generatorW(n, i)
       itr += 1
       idx += 2
    end
    return result
end

function basis(dim, index)
    x = zeros(dim)
    x[index] = 1.0
    return x
end

function sic_povm(fiducial)
    """
    Generate a set of POVMs by applying the d^2 Weyl-Heisenberg displacement operators to a
    fiducial state. 
    The Weyl-Heisenberg displacement operators are constructioned by Fuchs et al. in the article
    https://doi.org/10.3390/axioms6030021 and it is realized in QBism.
    
    """
    d = length(fiducial)
    w = exp(2.0*pi*1.0im/d)
    Z = diagm([w^(i-1) for i in 1:d])
    X = zeros(ComplexF64, d, d)
    for i in 1:d
        for j in 1:d
            if j != d
                X += basis(d, j+1)*basis(d,j)'
            else
                X += basis(d, 1)* basis(d,j)'
            end
        end
    end
    X = X/d
    
    D = [[Matrix{ComplexF64}(undef,d,d) for i in 1:d] for j in 1:d]
    for a in 1:d
        for b in 1:d
            X_a = X^(b-1)
            Z_b = Z^(a-1)
            D[a][b] = (-exp(1.0im*pi/d))^((a-1)*(b-1))*X_a*Z_b
        end
    end
            
    res = Vector{Matrix{ComplexF64}}()
    for m in 1:d
        for n in 1:d
            res_tp = D[m][n]*fiducial
            res_tp = res_tp/norm(res_tp)
            push!(res, res_tp*res_tp'/d)
        end
    end
    return res
end
 
function SIC(dim)
    data = readdlm("$(pkgpath)/sic_fiducial_vectors/d$(dim).txt", '\t', Float64, '\n')
    fiducial = data[:,1]+1.0im*data[:,2]
    M = sic_povm(fiducial)
end

function AdaptiveInput(x, func, dfunc; channel="dynamics")
    para_num = length(x)
    x_size = [x[i] for i in 1:para_num]
    x_list  =  Iterators.product(x...)
    if channel == "dynamics"
        H = [func(xi) for xi in x_list]
        dH = [dfunc(xi) for xi in x_list]
        return H, dH
    elseif channel == "kraus"
        K = [func(xi) for xi in x_list]
        dK = [dfunc(xi) for xi in x_list]
        return K, dK
    else
        throw("Supported values for channel are 'dynamics' and 'kraus'")
    end
end

#### bound control coefficients ####
function bound!(ctrl::Vector{Vector{Float64}}, ctrl_bound)
    for ck in 1:length(ctrl)
        for tk in 1:length(ctrl[1])
            ctrl[ck][tk] = (x-> x < ctrl_bound[1] ? ctrl_bound[1] : x > ctrl_bound[2] ? ctrl_bound[2] : x)(ctrl[ck][tk])
        end 
    end
end

function bound!(ctrl::Vector{Float64}, ctrl_bound)
    for ck in 1:length(ctrl)
        ctrl[ck] = (x-> x < ctrl_bound[1] ? ctrl_bound[1] : x > ctrl_bound[2] ? ctrl_bound[2] : x)(ctrl[ck])
    end
end

function Adam(gt, t, para, mt, vt, ϵ, beta1, beta2, eps)
    t = t+1
    mt = beta1*mt + (1-beta1)*gt
    vt = beta2*vt + (1-beta2)*(gt*gt)
    m_cap = mt/(1-(beta1^t))
    v_cap = vt/(1-(beta2^t))
    para = para+(ϵ*m_cap)/(sqrt(v_cap)+eps)
    return para, mt, vt
end

#### bound coefficients of linear combination in Mopt ####
function bound_LC_coeff!(coefficients::Vector{Vector{Float64}})
    M_num = length(coefficients)
    basis_num = length(coefficients[1])
    for ck in 1:M_num
        for tk in 1:basis_num
            coefficients[ck][tk] = (x-> x < 0.0 ? 0.0 : x > 1.0 ? 1.0 : x)(coefficients[ck][tk])
        end
    end

    Sum_row = [sum([coefficients[m][n] for n in 1:basis_num])  for m in 1:M_num]
    for mi in 1:M_num
        if Sum_row[mi] == 0.0
            int_num = sample(1:basis_num, 1, replace=false)[1]
            coefficients[mi][int_num] = rand()
        end
    end 

    Sum_col = [sum([coefficients[m][n] for m in 1:M_num])  for n in 1:basis_num]
    for i in 1:M_num
        for j in 1:basis_num
            coefficients[i][j] = coefficients[i][j]/Sum_col[j]
        end
    end
end

#### bound coefficients of rotation in Mopt ####
function bound_rot_coeff!(coefficients::Vector{Float64})
    n = length(coefficients)
    for tk in 1:n
        coefficients[tk] = (x-> x < 0.0 ? 0.0 : x > 2*pi ? 2*pi : x)(coefficients[tk])
    end
end

function gramschmidt(A::Vector{Vector{ComplexF64}})
    n = length(A[1])
    m = length(A)
    Q = [zeros(ComplexF64,n) for i in 1:m]
    for j in 1:m
        q = A[j]
        for i in 1:j
            rij = dot(Q[i], q)
            q = q - rij*Q[i]
        end
        Q[j] = q/norm(q)
    end
    Q
end

function rotation_matrix(coefficients, Lambda)
    dim = size(Lambda[1])[1]
    U = Matrix{ComplexF64}(I,dim,dim)
    for i in 1:length(Lambda)
        U = U*exp(1.0im*coefficients[i]*Lambda[i])
    end
    U
end

#### initialization states for DE and PSO method ####
function initial_state!(psi0, dynamics, p_num, rng)
    dim = length(dynamics[1].data.ψ0)
    if length(psi0) > p_num
        psi0 = [psi0[i] for i in 1:p_num]
    end
    for pj in 1:length(psi0)
        dynamics[pj].data.ψ0 = [psi0[pj][i] for i in 1:dim]
    end
    for pj in (length(psi0)+1):p_num
        r_ini = 2*rand(rng, dim)-ones(dim)
        r = r_ini/norm(r_ini)
        phi = 2*pi*rand(rng, dim)
        dynamics[pj].data.ψ0 = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
    end
end

#### initialization control coefficients for DE and PSO method ####
function initial_ctrl!(opt, ctrl0, dynamics, p_num, rng)
    ctrl_length = length(dynamics[1].data.ctrl[1])
    ctrl_num = length(dynamics[1].data.Hc)
    if length(ctrl0) > p_num
        ctrl0 = [ctrl0[i] for i in 1:p_num]
    end
    for pj in 1:length(ctrl0)
        dynamics[pj].data.ctrl = [[ctrl0[pj][i,j] for j in 1:ctrl_length] for i in 1:ctrl_num]
    end
    if opt.ctrl_bound[1] == -Inf || opt.ctrl_bound[2] == Inf
        for pj in (length(ctrl0)+1):p_num
            dynamics[pj].data.ctrl = [[2*rand(rng)-1.0 for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
    else
        a = opt.ctrl_bound[1]
        b = opt.ctrl_bound[2]
        for pj in (length(ctrl0)+1):p_num
            dynamics[pj].data.ctrl = [[(b-a)*rand(rng)+a for j in 1:ctrl_length] for i in 1:ctrl_num]
        end
    end
end

#### initialization velocity for PSO ####
function initial_velocity_ctrl(opt, ctrl_length, ctrl_num, p_num, rng)
    if opt.ctrl_bound[1] == -Inf || opt.ctrl_bound[2] == Inf
        velocity = 0.1*(2.0*rand(rng, ctrl_num, ctrl_length, p_num)-ones(ctrl_num, ctrl_length, p_num))
    else
        a = opt.ctrl_bound[1]
        b = opt.ctrl_bound[2]
        velocity = 0.1*((b-a)*rand(rng, ctrl_num, ctrl_length, p_num)+a*ones(ctrl_num, ctrl_length, p_num))
    end
    velocity
end

#### initialization measurements for DE and PSO ####
function initial_M!(measurement0, C_all, dim, p_num, rng)
    M_num = length(measurement0[1])
    if length(measurement0) > p_num
        measurement0 = [measurement0[i] for i in 1:p_num]
    end 
    for pj in 1:length(measurement0)
        C_all[pj] = [[measurement0[pj][i,j] for j in 1:dim] for i in 1:M_num]
    end
    for pj in (length(measurement0)+1):p_num
        M_tp = [Vector{ComplexF64}(undef, dim) for i in 1:M_num]
        for mi in 1:M_num
            r_ini = 2*rand(rng, dim)-ones(dim)
            r = r_ini/norm(r_ini)
            phi = 2*pi*rand(rng, dim)
            M_tp[mi] = [r[i]*exp(1.0im*phi[i]) for i in 1:dim]
        end
        C_all[pj] = [[M_tp[i][j] for j in 1:dim] for i in 1:M_num]
        # orthogonality and normalization 
        C_all[pj] = gramschmidt(C_all[pj])
    end
end

function initial_LinearComb!(measurement0, B_all, basis_num, M_num, p_num, rng)
    if length(measurement0) > p_num
        measurement0 = [measurement0[i] for i in 1:p_num]
    end 
    for pj in 1:length(measurement0)
        B_all[pj] = [[measurement0[pj][i,j] for j in 1:dim] for i in 1:M_num]
    end

    for pj in (length(measurement0)+1):p_num
        B_all[pj] = [rand(rng, basis_num) for i in 1:M_num]
        bound_LC_coeff!(B_all[pj])
    end
end

function initial_Rotation!(measurement0, s_all, dim, p_num, rng)
    if length(measurement0) > p_num
        measurement0 = [measurement0[i] for i in 1:p_num]
    end 
    for pj in 1:length(measurement0)
        s_all[pj] = [measurement0[pj][i] for i in 1:dim*dim]
    end

    for pj in (length(measurement0)+1):p_num
        s_all[pj] = rand(rng, dim*dim)
    end
end
