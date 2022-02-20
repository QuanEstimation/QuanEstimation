using LinearAlgebra
using Trapz
using Interpolations
using PyCall

include("quanestimation/JuliaSrc/Common/common.jl")
include("quanestimation/JuliaSrc/Dynamics/dynamics.jl")
include("quanestimation/JuliaSrc/AsymptoticBound/CramerRao.jl")
include("quanestimation/JuliaSrc/BayesianBound/BayesianCramerRao.jl")
include("quanestimation/JuliaSrc/BayesianBound/ZivZakai.jl")

#initial state
rho0 = 0.5*[1.0 1.0+0.0im; 1.0 1.0]
#Hamiltonian
# B = pi/2.0
sx = [0.0 1.0; 1.0 0.0im]
sy = [0.0 -1.0im; 1.0im 0.0]
sz = [1.0 0.0im; 0.0 -1.0]

function H0_res(x,B)
    return 0.5*B*(sx*cos(x)+sz*sin(x))
end
function dH_res(x,B)
    return [0.5*B*(-sx*sin(x)+sz*cos(x)), 0.5*(sx*cos(x)+sz*sin(x))]
end

#measurement
Measurement = sic_povm([0.37637620719571985+2.7760878126176896im,1.1538819157681195+0.87835540934078105im])

tspan = range(0.0, stop=1.0, length=1000)

dim = 2
para_num = 2

decay_opt = [zeros(ComplexF64,para_num,para_num)] 
γ = [0.0]

#### flat distribution ####
xspan = [range(-pi/2.0, stop=pi/2.0, length=100), range(pi/2-0.01, stop=pi/2+0.01, length=10)].|>Vector
x1, x2 = [x[1] for x in xspan],  [x[end] for x in xspan]

σ_x = 0.5
σ_B = 1.0
# ρ = 0.5
μ = [0., 0.]

BQCRB1 =  Matrix{Float64}[]
BQCRB2 =  Matrix{Float64}[]
QVTB1 =  Matrix{Float64}[]
QVTB2 =  Matrix{Float64}[]
BCRB1 =  Matrix{Float64}[]
BCRB2 =  Matrix{Float64}[]
VTB1 =  Matrix{Float64}[]
VTB2 =  Matrix{Float64}[]

for ρ in range(0.1,0.9,length=10)
    BivarNormal(x,B) = 1/(2pi*σ_x*σ_B*sqrt(1-ρ^2))*exp(-((x-μ[1])^2/σ_x^2-2ρ*(x-μ[1])*(B-μ[2])/σ_x/σ_B + (B-μ[2])^2/σ_B^2)/2/(1-ρ^2))
    function dBivarNormal(x,B)
        BN = BivarNormal(x,B)
        [BN*(2*(x-μ[1])/σ_x^2-2*ρ*(B-μ[2])/σ_B/σ_x), BN*(2*(B-μ[2])/σ_B^2-2*ρ*(x-μ[1])/σ_x/σ_B)]./(-2*(1- ρ^2))
    end
    p = [BivarNormal(x,B) for (x,B) in Iterators.product(xspan...)]
    trapzm(x, integrands, slice_dim) =  [trapz(tuple(x...), I) for I in [reshape(hcat(integrands...)[i,:], length.(x)...) for i in 1:slice_dim]]
    C = trapzm(xspan, p, 1)
    p = p./C
    dp = Matrix{Vector{Float64}}(undef, length.(xspan)...)
    rho_all = Matrix{Matrix{ComplexF64}}(undef, length.(xspan)...)
    drho_all = Matrix{Vector{Matrix{ComplexF64}}}(undef, length.(xspan)...)
    for i = 1:length(xspan[1])
        for j = 1:length(xspan[2])  
            H0 = H0_res(xspan[1][i], xspan[2][j])
            dH = dH_res(xspan[1][i], xspan[2][j])
            rho, drho  = expm(H0, dH, rho0, decay_opt, γ, [zeros(ComplexF64,2,2)], [zeros(length(tspan)-1)],tspan)
            rho_all[i,j] = rho[end]
            drho_all[i,j] = drho[end]
            dp[i,j] = dBivarNormal(xspan[1][i],xspan[2][j])./C
        end
    end
    append!(BCRB1, [BCRB(xspan, p, rho_all, drho_all, M=Measurement, eps=1e-8)])
    append!(BCRB2, [BCRB(xspan, p, rho_all, drho_all, M=Measurement, btype=2, eps=1e-8)])
    append!(BQCRB1, [BQCRB(xspan, p, rho_all, drho_all, eps=1e-8)])
    append!(BQCRB2, [BQCRB(xspan, p, rho_all, drho_all, btype=2, eps=1e-8)])
    append!(VTB1, [VTB(xspan, p, dp, rho_all, drho_all, M=Measurement, eps=1e-8)])
    append!(VTB2, [VTB(xspan, p, dp, rho_all, drho_all, btype=2, M=Measurement, eps=1e-8)])
    append!(QVTB1, [QVTB(xspan, p, dp, rho_all, drho_all, eps=1e-8)])
    append!(QVTB2, [QVTB(xspan, p, dp, rho_all, drho_all, btype=2, eps=1e-8)])
end

np = pyimport("numpy")
np.save("BCRB1.npy",BCRB1)
np.save("BCRB2.npy",BCRB2)
np.save("BQCRB1.npy",BQCRB1)
np.save("BQCRB2.npy",BQCRB2)
np.save("VTB1.npy",VTB1)
np.save("VTB2.npy",VTB2)
np.save("QVTB1.npy",QVTB1)
np.save("QVTB2.npy",QVTB2)
