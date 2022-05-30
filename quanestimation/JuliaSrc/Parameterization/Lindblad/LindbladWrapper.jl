# wrapper for Lindblad dynamics with ControlOpt
"""

	Lindblad(opt::ControlOpt, tspan, ρ₀, H0, dH, Hc; decay=missing, eps=GLOBAL_EPS)
	
Initialize the parameterization described by the Lindblad master equation governed dynamics for the control optimization.
"""
function Lindblad(opt::ControlOpt, tspan, ρ₀, H0, dH, Hc; decay=missing, eps=GLOBAL_EPS)
	(;ctrl) = opt
	dim = size(ρ₀, 1)
	if ismissing(dH)
		dH = [zeros(ComplexF64, dim, dim)]
	end
	ctrl_num = length(Hc)
	tnum = length(tspan)
	
	if ismissing(decay)
		decay_opt = [zeros(ComplexF64, dim, dim)]
		γ = [0.0]
	else
		decay_opt = [decay[1] for decay in decay]
		γ = [decay[2] for decay in decay]
	end

	if ismissing(Hc)
		Hc = [zeros(ComplexF64, dim, dim)]
		ctrl = [zeros(tnum-1)]
	elseif ismissing(ctrl)
		ctrl = [zeros(tnum-1) for _ in 1:ctrl_num]
		opt.ctrl = ctrl
	else
		ctrl_length = length(ctrl)
		if ctrl_num < ctrl_length
			throw(ArgumentError(
			"There are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences: too many coefficients sequences"
			))
		elseif ctrl_num < ctrl_length
			throw(ArgumentError(
			"Not enough coefficients sequences: there are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences. The rest of the control sequences are set to be 0."
			))
		end
		
		ratio_num = ceil((length(tspan)-1) / length(ctrl[1]))
		if length(tspan) - 1 % length(ctrl[1])  != 0
			tnum = ratio_num * length(ctrl[1]) |> Int
			tspan = range(tspan[1], tspan[end], length=tnum+1)
		end
	end
	H0 = complex(H0)
	dH = complex.(dH)
	ρ₀ = complex(ρ₀)
	
	Lindblad(H0, dH, Hc, ctrl, ρ₀, tspan, decay_opt, γ)
end

Lindblad(opt::ControlOpt, tspan, ρ₀, H0, dH, Hc, decay; eps=GLOBAL_EPS) = 
	Lindblad(opt, tspan, ρ₀, H0, dH, Hc; decay=decay, eps=eps)
	
"""

	Lindblad(opt::StateOpt, tspan, H0, dH; Hc=missing, ctrl=missing, decay=missing, eps=GLOBAL_EPS)
	
Initialize the parameterization described by the Lindblad master equation governed dynamics for the state optimization.
"""
function Lindblad(opt::StateOpt, tspan, H0, dH; Hc=missing, ctrl=missing, decay=missing, eps=GLOBAL_EPS)
	(;psi) = opt
	dim = H0 isa AbstractVector ? size(H0[1], 1) : size(H0, 1)
	if ismissing(psi) 
		r_ini = 2*rand(opt.rng, dim) - ones(dim)
		r = r_ini ./ norm(r_ini)
		ϕ = 2pi*rand(opt.rng, dim)
		psi = [r*exp(im*ϕ) for (r, ϕ) in zip(r, ϕ)]
		opt.psi = psi 
	end

	if ismissing(dH)
		dH = [zeros(ComplexF64, dim, dim)]
	end

	if !ismissing(Hc) && !ismissing(ctrl)
		ctrl_length = length(ctrl)
		ctrl_num = length(Hc)
		if ctrl_num < ctrl_length
			throw(ArgumentError(
			"Too many contrl coefficients: there are $ctrl_num control Hamiltonians but $ctrl_length control coefficients given."
			))
		elseif ctrl_num < ctrl_length
			throw(ArgumentError(
			"Insufficient coefficients sequences: there are $ctrl_num control Hamiltonians but $ctrl_length coefficients given. The rest of the control sequences are setten to be 0."
			))
		end
		
		if length(ctrl[1]) == 1
			hc = sum([c[1]*hc for (c,hc) in zip(ctrl,Hc)]) 

			if typeof(H0) <: AbstractMatrix
				H0 = complex(H0 + hc)
			elseif typeof(H0) <: AbstractVector
				H0 = [complex(h0+hc) for h0 in H0] 
			else
				## TODO wrong type of H0
			end
		else
			ratio_num = ceil((length(tspan)-1) / length(ctrl[1]))
			
			if length(tspan) - 1 % length(ctrl[1])  != 0
				tnum = ratio_num * length(ctrl[1]) |> Int
				tspan = range(tspan[1], tspan[end], length=tnum+1)
				if typeof(H0) <: AbstractVector
					itp = interpolate((tspan,), H0,  Gridded(Linear())) 
					H0 = itp(tspan)
				end
			end

			hc = [sum([c[i]*hc for (c,hc) in zip(ctrl,Hc)]) for i in 1:length(ctrl[1]) ]

			if typeof(H0) <: AbstractMatrix
				H0 = [complex(H0+hc) for hc in hc]
			elseif typeof(H0) <: AbstractVector
				H0 = complex.(H0+hc)
			else
				## TODO wrong type of H0
			end
		end
	end
	if ismissing(decay)
		γ = [0.0]
	else
		decay_opt = [decay[1] for decay in decay]
		γ = [decay[2] for decay in decay]
	end

	dH = complex.(dH)
	psi = complex(psi)

	if all(iszero.(γ)) #  if any non-zero decay rate
		return Lindblad(H0, dH, psi, tspan)
	else
		return Lindblad(H0, dH, psi, tspan, decay_opt, γ)
	end
end

Lindblad(opt::StateOpt, tspan, H0, dH, Hc, ctrl, decay; eps=GLOBAL_EPS) =
	Lindblad(opt, tspan, H0, dH; Hc=Hc, ctrl=ctrl, decay=decay, eps=eps)

function _ini_measurement!(opt::Mopt_Projection, dim::Int; eps=GLOBAL_EPS)
	(; M) = opt
	## initialize the Mopt target M
	C = [ComplexF64[] for _ in 1:dim]
	if ismissing(M)
		for i in 1:dim
			r_ini = 2*rand(opt.rng, dim) - ones(dim)
			r = r_ini/norm(r_ini)
			ϕ = 2pi*rand(opt.rng, dim)
			C[i] = [r*exp(im*ϕ) for (r,ϕ) in zip(r,ϕ)] 
		end
		opt.M = gramschmidt(C)
	end
end

function _ini_measurement!(opt::Mopt_LinearComb, dim::Int; eps=GLOBAL_EPS)
	(;B, POVM_basis, M_num) = opt
	if ismissing(POVM_basis) 
		opt.POVM_basis = SIC(dim)
	else
		## TODO: accuracy ?
		for P in POVM_basis
			if minimum(eigvals(P)) < (-eps)
				throw(ArgumentError("The given POVMs should be semidefinite!"))
			end
		end
		if !(sum(POVM_basis) ≈ I(dim))
			throw(ArgumentError("The sum of the given POVMs should be identity matrix!"))
		end
	end

	if ismissing(B)
		opt.B = [rand(opt.rng, length(POVM_basis)) for _ in 1:M_num]
	end
end

function _ini_measurement!(opt::Mopt_Rotation, dim::Int; eps=GLOBAL_EPS)
	(;s, POVM_basis, Lambda) = opt
	if ismissing(POVM_basis)
		throw(ArgumentError("The initial POVM basis should not be empty!"))
	else
		## TODO: accuracy ?
		for P in POVM_basis
			if minimum(eigvals(P)) < -eps
				throw(ArgumentError("The given POVMs should be semidefinite!"))
			end
		end
		if !(sum(POVM_basis) ≈ I(dim))
			throw(ArgumentError("The sum of the given POVMs should be identity matrix!"))
		end
	end

	if ismissing(s)
		opt.s = rand(opt.rng, dim^2)
	end
end

"""

	Lindblad(opt::AbstractMopt, tspan, ρ₀, H0, dH; Hc=missing, ctrl=missing, decay=missing, eps=GLOBAL_EPS)
	
Initialize the parameterization described by the Lindblad master equation governed dynamics for the measurement optimization.
"""
function Lindblad(opt::AbstractMopt, tspan, ρ₀, H0, dH; Hc=missing, ctrl=missing, decay=missing, eps=GLOBAL_EPS)
	dim = size(ρ₀, 1)
	_ini_measurement!(opt, dim; eps=eps)
	
	if ismissing(dH)
		dH = [zeros(ComplexF64, dim, dim)]
	end

	if !ismissing(Hc) && !ismissing(ctrl)
		ctrl_length = length(ctrl)
		ctrl_num = length(Hc)
		if ctrl_num < ctrl_length
			throw(ArgumentError(
			"Too many contrl coefficients: there are $ctrl_num control Hamiltonians but $ctrl_length control coefficients given."
			))
		elseif ctrl_num < ctrl_length
			throw(ArgumentError(
			"Insufficient coefficients sequences: there are $ctrl_num control Hamiltonians but $ctrl_length coefficients given. The rest of the control sequences are setten to be 0."
			))
		end
		
		if length(ctrl[1]) == 1
			hc = sum([c[1]*hc for (c,hc) in zip(ctrl,Hc)]) 

			if typeof(H0) <: AbstractMatrix
				H0 = complex(H0+hc)
			elseif typeof(H0) <: AbstractVector
				H0 = [complex(h0+hc) for h0 in H0] 
			else
				## TODO wrong type of H0
			end
		else
			ratio_num = ceil((length(tspan)-1) / length(ctrl[1]))
			
			if length(tspan) - 1 % length(ctrl[1])  != 0
				tnum = ratio_num * length(ctrl[1]) |> Int
				tspan = range(tspan[1], tspan[end], length=tnum+1)
				if typeof(H0) <: AbstractVector
					itp = interpolate((tspan,), H0,  Gridded(Linear())) 
					H0 = itp(tspan)
				end
			end

			hc = [sum([c[i]*hc for (c,hc) in zip(ctrl,Hc)]) for i in 1:length(ctrl[1]) ]

			if typeof(H0) <: AbstractMatrix
				H0 = [complex(H0+hc) for hc in hc]
			elseif typeof(H0) <: AbstractVector
				H0 = complex.(H0+hc)
			else
				## TODO wrong type of H0
			end
		end
	end	
	if ismissing(decay)
		γ = [0.0]
	else
		decay_opt = [decay[1] for decay in decay]
		γ = [decay[2] for decay in decay]
	end

	dH = complex.(dH)
	ρ₀ = complex(ρ₀)

	if all(iszero.(γ)) #  if any non-zero decay rate
		return Lindblad(H0, dH, ρ₀, tspan)
	else
		return Lindblad(H0, dH, ρ₀, tspan, decay_opt, γ)
	end
end

Lindblad(opt::AbstractMopt, tspan, ρ₀, H0, dH, Hc, ctrl, decay; eps=GLOBAL_EPS) =
	Lindblad(opt, tspan, ρ₀, H0, dH; Hc=Hc, ctrl=ctrl, decay=decay, eps=eps)

function _ini_measurement!(opt::CompOpt, dim::Int; eps=GLOBAL_EPS)
	(; M) = opt
	## initialize the Mopt target M
	C = [ComplexF64[] for _ in 1:dim]
	if ismissing(M)
		for i in 1:dim
			r_ini = 2*rand(opt.rng, dim) - ones(dim)
			r = r_ini/norm(r_ini)
			ϕ = 2pi*rand(opt.rng, dim)
			C[i] = [r*exp(im*ϕ) for (r,ϕ) in zip(r,ϕ)] 
		end
		opt.M = gramschmidt(C)
	end
end

"""

	Lindblad(opt::StateControlOpt, tspan, H0, dH, Hc; decay=missing, eps=GLOBAL_EPS)
	
Initialize the parameterization described by the Lindblad master equation governed dynamics for the comprehensive optimization on state and control.
"""
function Lindblad(opt::StateControlOpt, tspan, H0, dH, Hc; decay=missing, eps=GLOBAL_EPS)
	(;psi, ctrl) = opt
	dim = H0 isa AbstractVector ? size(H0[1], 1) : size(H0, 1)
	if ismissing(psi) 
		r_ini = 2*rand(opt.rng, dim) - ones(dim)
		r = r_ini ./ norm(r_ini)
		ϕ = 2pi*rand(opt.rng, dim)
		psi = [r*exp(im*ϕ) for (r, ϕ) in zip(r, ϕ)]
		opt.psi = psi 
	end
	if ismissing(dH)
		dH = [zeros(ComplexF64, dim, dim)]
	end
	ctrl_num = length(Hc)
	tnum = length(tspan)
	
	if ismissing(decay)
		decay_opt = [zeros(ComplexF64, dim, dim)]
		γ = [0.0]
	else
		decay_opt = [decay[1] for decay in decay]
		γ = [decay[2] for decay in decay]
	end

	if ismissing(Hc)
		Hc = [zeros(ComplexF64, dim, dim)]
		opt.ctrl = [zeros(tnum-1)]
	elseif ismissing(ctrl)
		ctrl = [zeros(tnum-1) for _ in 1:ctrl_num]
		opt.ctrl = ctrl
	else
		ctrl_length = length(ctrl)
		if ctrl_num < ctrl_length
			throw(ArgumentError(
			"There are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences: too many coefficients sequences"
			))
		elseif ctrl_num < ctrl_length
			throw(ArgumentError(
			"Not enough coefficients sequences: there are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences. The rest of the control sequences are set to be 0."
			))
		end
		
		ratio_num = ceil((length(tspan)-1) / length(ctrl[1]))
		if length(tspan) - 1 % length(ctrl[1])  != 0
			tnum = ratio_num * length(ctrl[1]) |> Int
			tspan = range(tspan[1], tspan[end], length=tnum+1)
		end
	end
	H0 = complex(H0)
	dH = complex.(dH)
	psi = complex(psi)
	Lindblad(H0, dH, Hc, ctrl, psi, tspan, decay_opt, γ)
end

Lindblad(opt::StateControlOpt, tspan, H0, dH, Hc, decay; eps=GLOBAL_EPS) = 
	Lindblad(opt, tspan, H0, dH, Hc; decay=decay, eps=eps)
	
"""

	Lindblad(opt::ControlMeasurementOpt, tspan, ρ₀, H0, dH, Hc; decay=missing,eps=GLOBAL_EPS)
	
Initialize the parameterization described by the Lindblad master equation governed dynamics for the comprehensive optimization on control and measurement.
"""
function Lindblad(opt::ControlMeasurementOpt, tspan, ρ₀, H0, dH, Hc; decay=missing, eps=GLOBAL_EPS)
	(;ctrl) = opt
	dim = size(ρ₀, 1)
	_ini_measurement!(opt, dim; eps=eps)
	if ismissing(dH)
		dH = [zeros(ComplexF64, dim, dim)]
	end
	ctrl_num = length(Hc)
	tnum = length(tspan)
	
	if ismissing(decay)
		decay_opt = [zeros(ComplexF64, dim, dim)]
		γ = [0.0]
	else
		decay_opt = [decay[1] for decay in decay]
		γ = [decay[2] for decay in decay]
	end

	if ismissing(Hc)
		Hc = [zeros(ComplexF64, dim, dim)]
		opt.ctrl = [zeros(tnum-1)]
	elseif ismissing(ctrl)
		ctrl = [zeros(tnum-1) for _ in 1:ctrl_num]
		opt.ctrl = ctrl
	else
		ctrl_length = length(ctrl)
		if ctrl_num < ctrl_length
			throw(ArgumentError(
			"There are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences: too many coefficients sequences"
			))
		elseif ctrl_num < ctrl_length
			throw(ArgumentError(
			"Not enough coefficients sequences: there are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences. The rest of the control sequences are set to be 0."
			))
		end
		
		ratio_num = ceil((length(tspan)-1) / length(ctrl[1]))
		if length(tspan) - 1 % length(ctrl[1])  != 0
			tnum = ratio_num * length(ctrl[1]) |> Int
			tspan = range(tspan[1], tspan[end], length=tnum+1)
		end
	end
	H0 = complex(H0)
	dH = complex.(dH)
	ρ₀ = complex(ρ₀)
	
	Lindblad(H0, dH, Hc, ctrl, ρ₀, tspan, decay_opt, γ)
end

Lindblad(opt::ControlMeasurementOpt, tspan, ρ₀, H0, dH, Hc, decay; eps=GLOBAL_EPS) = 
	Lindblad(opt, tspan, ρ₀, H0, dH, Hc; decay=decay, eps=eps)
	
"""

	Lindblad(opt::StateMeasurementOpt, tspan, H0, dH; Hc=missing, ctrl=missing, decay=missing)
	
Initialize the parameterization described by the Lindblad master equation governed dynamics for the comprehensive optimization on state and measurement.
"""
function Lindblad(opt::StateMeasurementOpt, tspan, H0, dH; Hc=missing, ctrl=missing, decay=missing)
	(;psi) = opt
	dim = H0 isa AbstractVector ? size(H0[1], 1) : size(H0, 1)
	_ini_measurement!(opt, dim; eps=eps)
	if ismissing(psi) 
		r_ini = 2*rand(opt.rng, dim) - ones(dim)
		r = r_ini ./ norm(r_ini)
		ϕ = 2pi*rand(opt.rng, dim)
		psi = [r*exp(im*ϕ) for (r, ϕ) in zip(r, ϕ)]
		opt.psi = psi 
	end

	if ismissing(dH)
		dH = [zeros(ComplexF64, dim, dim)]
	end

	if !ismissing(Hc) && !ismissing(ctrl)
		ctrl_length = length(ctrl)
		ctrl_num = length(Hc)
		if ctrl_num < ctrl_length
			throw(ArgumentError(
			"Too many contrl coefficients: there are $ctrl_num control Hamiltonians but $ctrl_length control coefficients given."
			))
		elseif ctrl_num < ctrl_length
			throw(ArgumentError(
			"Insufficient coefficients sequences: there are $ctrl_num control Hamiltonians but $ctrl_length coefficients given. The rest of the control sequences are setten to be 0."
			))
		end
		
		if length(ctrl[1]) == 1
			hc = sum([c[1]*hc for (c,hc) in zip(ctrl,Hc)]) 

			if typeof(H0) <: AbstractMatrix
				H0 = complex(H0 + hc)
			elseif typeof(H0) <: AbstractVector
				H0 = [complex(h0+hc) for h0 in H0] 
			else
				## TODO wrong type of H0
			end
		else
			ratio_num = ceil((length(tspan)-1) / length(ctrl[1]))
			
			if length(tspan) - 1 % length(ctrl[1])  != 0
				tnum = ratio_num * length(ctrl[1]) |> Int
				tspan = range(tspan[1], tspan[end], length=tnum+1)
				if typeof(H0) <: AbstractVector
					itp = interpolate((tspan,), H0,  Gridded(Linear())) 
					H0 = itp(tspan)
				end
			end

			hc = [sum([c[i]*hc for (c,hc) in zip(ctrl,Hc)]) for i in 1:length(ctrl[1]) ]

			if typeof(H0) <: AbstractMatrix
				H0 = [complex(H0+hc) for hc in hc]
			elseif typeof(H0) <: AbstractVector
				H0 = complex.(H0+hc)
			else
				## TODO wrong type of H0
			end
		end
	end
	if ismissing(decay)
		γ = [0.0]
	else
		decay_opt = [decay[1] for decay in decay]
		γ = [decay[2] for decay in decay]
	end

	dH = complex.(dH)
	psi = complex(psi)

	if all(iszero.(γ)) #  if any non-zero decay rate
		return Lindblad(H0, dH, psi, tspan)
	else
		return Lindblad(H0, dH, psi, tspan, decay_opt, γ)
	end
end

Lindblad(opt::StateMeasurementOpt, tspan, H0, dH, Hc, ctrl, decay;eps=GLOBAL_EPS) = 
	Lindblad(opt, tspan, H0, dH; Hc=Hc, ctrl=ctrl, decay=decay, eps=eps)

"""

	Lindblad(opt::StateControlMeasurementOpt, tspan, H0, dH, Hc; decay=missing,eps=GLOBAL_EPS)
	
Initialize the parameterization described by the Lindblad master equation governed dynamics for the comprehensive optimization on state, control and measurement.
"""
function Lindblad(opt::StateControlMeasurementOpt, tspan, H0, dH, Hc; decay=missing, eps=GLOBAL_EPS)
	(;ctrl, psi) = opt
	dim = H0 isa AbstractVector ? size(H0[1], 1) : size(H0, 1)
	_ini_measurement!(opt, dim; eps=eps)

	if ismissing(psi) 
		r_ini = 2*rand(opt.rng, dim) - ones(dim)
		r = r_ini ./ norm(r_ini)
		ϕ = 2pi*rand(opt.rng, dim)
		psi = [r*exp(im*ϕ) for (r, ϕ) in zip(r, ϕ)]
		opt.psi = psi 
	end

	if ismissing(dH)
		dH = [zeros(ComplexF64, dim, dim)]
	end
	ctrl_num = length(Hc)
	tnum = length(tspan)
	
	if ismissing(decay)
		decay_opt = [zeros(ComplexF64, dim, dim)]
		γ = [0.0]
	else
		decay_opt = [decay[1] for decay in decay]
		γ = [decay[2] for decay in decay]
	end

	if ismissing(Hc)
		Hc = [zeros(ComplexF64, dim, dim)]
		opt.ctrl = [zeros(tnum-1)]
	elseif ismissing(ctrl)
		ctrl = [zeros(tnum-1) for _ in 1:ctrl_num]
		opt.ctrl = ctrl
	else
		ctrl_length = length(ctrl)
		if ctrl_num < ctrl_length
			throw(ArgumentError(
			"There are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences: too many coefficients sequences"
			))
		elseif ctrl_num < ctrl_length
			throw(ArgumentError(
			"Not enough coefficients sequences: there are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences. The rest of the control sequences are set to be 0."
			))
		end
		
		ratio_num = ceil((length(tspan)-1) / length(ctrl[1]))
		if length(tspan) - 1 % length(ctrl[1])  != 0
			tnum = ratio_num * length(ctrl[1]) |> Int
			tspan = range(tspan[1], tspan[end], length=tnum+1)
		end
	end
	H0 = complex(H0)
	dH = complex.(dH)
	psi = complex(psi)
	
	Lindblad(H0, dH, Hc, ctrl, psi, tspan, decay_opt, γ)
end

Lindblad(opt::StateControlMeasurementOpt, tspan, H0, dH, Hc, decay; eps=GLOBAL_EPS) = 
	Lindblad(opt, tspan, H0, dH, Hc; decay=decay, eps=eps)
