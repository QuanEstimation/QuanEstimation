"""

    Kraus(opt::StateOpt, K, dK)
    
Initialize the parameterization described by the Kraus operators for the state optimization. 
"""

function Kraus(opt::StateOpt, K, dK)
    (;psi) = opt
    dim = size(K[1], 1)
    if ismissing(psi)
        r_ini = 2*rand(opt.rng, dim) - ones(dim)
		r = r_ini ./ norm(r_ini)
		ϕ = 2pi*rand(opt.rng, dim)
		psi = [r*exp(im*ϕ) for (r, ϕ) in zip(r, ϕ)]
		opt.psi = psi 
    end
    
    K = complex.(K)
    dK = [complex.(dk) for dk in dK]
    psi = complex(psi)

    Kraus(psi, K, dK)
end
"""

    Kraus(opt::AbstractMopt, ρ₀::AbstractMatrix, K, dK; eps=GLOBAL_EPS)
    
Initialize the parameterization described by the Kraus operators for the measurement optimization. 
"""
function Kraus(opt::AbstractMopt, ρ₀::AbstractMatrix, K, dK; eps=GLOBAL_EPS)
    dim = size(ρ₀, 1)
    _ini_measurement!(opt, dim; eps=eps)

    K = complex.(K)
    dK = [complex.(dk) for dk in dK]
    ρ₀ = complex(ρ₀)
    Kraus(ρ₀, K, dK)
end

"""

    Kraus(opt::CompOpt, K, dK; eps=GLOBAL_EPS)
    
Initialize the parameterization described by the Kraus operators for the comprehensive optimization. 
"""
function Kraus(opt::CompOpt, K, dK; eps=GLOBAL_EPS)
    (;psi) = opt
    dim = size(K[1], 1)
    if ismissing(psi)
        r_ini = 2*rand(opt.rng, dim) - ones(dim)
		r = r_ini ./ norm(r_ini)
		ϕ = 2pi*rand(opt.rng, dim)
		psi = [r*exp(im*ϕ) for (r, ϕ) in zip(r, ϕ)]
		opt.psi = psi 
    end
    
    _ini_measurement!(opt, dim; eps=eps)
    K = complex.(K)
    dK = [complex.(dk) for dk in dK]
    psi = complex(psi)
    Kraus(psi, K, dK)
end