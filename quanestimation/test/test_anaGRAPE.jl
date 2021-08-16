# using BenchmarkTools
#using AutoGrape
using LinearAlgebra
# for T in 1.0:1.0:30|>Array
include("AutoGrape.jl")
using AutoGrape
T = 2.0
# begin

# θ = 114514
# ψ = [1,exp(im * θ)]  |> normalize
# ρ_initial = (ψ * ψ' |> complex) 

psi0 = [1.0+0.0im,0.0+0.0im]
psi1 = [0.0+0.0im, 1.0+0.0im]
psi_p = (psi0+psi1)/sqrt(2)
psi_m = (psi0-psi1)/sqrt(2)
ρ_initial = psi_p*psi_p'

# M0 = 0.5 * [1.0 + 0.0im -1. -1. 1.]
# M1 = 0.5 * [1.0 1. 1. 1.]
# M = [M0, M1]

times = range(0, T, length=(250 * T) |> Int)
ϵ = 0.1

w = 1.
# θ = pi/4
H0 = 0.5 * w  *  sigmaz()
dH = [0.5 * sigmaz()] 
Liouville_operator = [sigmaz()]
γ = [0.05]


control_Hamiltonian = [sigmax(),sigmay(),sigmaz()]  

Hc_coeff = [[0.5 for i = 1:length(times)] for j = 1:length(control_Hamiltonian)]

# Hc_coeff = load("controls.jld")["controls"]
dim = size(H0)[1]
grape = GrapeControl(H0, dH, ρ_initial, times, Liouville_operator, γ, control_Hamiltonian, Hc_coeff, ϵ)

AutoGrape.gradient_QFI_analitical_ADAM!(grape)

println("finished test")