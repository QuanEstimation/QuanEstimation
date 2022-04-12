function Kraus(opt::StateOpt, K, dK;rng=GLOBAL_RNG)
    (;ψ₀) = opt
    dim = size(K[1], 1)
    if ismissing(ψ₀)
        r_ini = 2*rand(rng, dim) - ones(dim)
		r = r_ini ./ norm(r_ini)
		ϕ = 2pi*rand(rng, dim)
		ψ₀ = [r*exp(im*ϕ) for (r, ϕ) in zip(r, ϕ)]
		opt.ψ₀ = ψ₀ 
    end
    
    K = complex.(K)
    dK = [complex.(dk) for dk in dK]
    ψ₀ = complex(ψ₀)

    Kraus(K, dK, ψ₀)
end

function Kraus(opt::AbstractMopt, ρ₀::AbstractMatrix, K, dK;rng=GLOBAL_RNG, eps=GLOBAL_EPS)
    dim = size(ρ₀, 1)
    _ini_measurement!(opt, dim, rng; eps=eps)

    K = complex.(K)
    dK = [complex.(dk) for dk in dK]
    ρ₀ = complex(ρ₀)
    Kraus(K, dK, ρ₀)
end


function Kraus(opt::CompOpt, K, dK;rng=GLOBAL_RNG, eps=GLOBAL_EPS)
    (;ψ₀) = opt
    dim = size(K[1], 1)
    if ismissing(ψ₀)
        r_ini = 2*rand(rng, dim) - ones(dim)
		r = r_ini ./ norm(r_ini)
		ϕ = 2pi*rand(rng, dim)
		ψ₀ = [r*exp(im*ϕ) for (r, ϕ) in zip(r, ϕ)]
		opt.ψ₀ = ψ₀ 
    end
    
    _ini_measurement!(opt, dim, rng; eps=eps)
    K = complex.(K)
    dK = [complex.(dk) for dk in dK]
    ψ₀ = complex(ψ₀)
    Kraus(K, dK, ψ₀)
end