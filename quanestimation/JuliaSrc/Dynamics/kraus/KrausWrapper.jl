function Kraus(opt::StateOpt, K, dK;rng=GLOBAL_RNG)
    (;psi) = opt
    dim = size(K[1], 1)
    if ismissing(psi)
        r_ini = 2*rand(rng, dim) - ones(dim)
		r = r_ini ./ norm(r_ini)
		ϕ = 2pi*rand(rng, dim)
		psi = [r*exp(im*ϕ) for (r, ϕ) in zip(r, ϕ)]
		opt.psi = psi 
    end
    
    K = complex.(K)
    dK = [complex.(dk) for dk in dK]
    psi = complex(psi)

    Kraus(K, dK, psi)
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
    (;psi) = opt
    dim = size(K[1], 1)
    if ismissing(psi)
        r_ini = 2*rand(rng, dim) - ones(dim)
		r = r_ini ./ norm(r_ini)
		ϕ = 2pi*rand(rng, dim)
		psi = [r*exp(im*ϕ) for (r, ϕ) in zip(r, ϕ)]
		opt.psi = psi 
    end
    
    _ini_measurement!(opt, dim, rng; eps=eps)
    K = complex.(K)
    dK = [complex.(dk) for dk in dK]
    psi = complex(psi)
    Kraus(K, dK, psi)
end