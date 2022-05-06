
# dynamics in Lindblad form
struct Lindblad{N,C,R,P} <: AbstractDynamics
    data::AbstractDynamicsData
    noise_type::Symbol
    ctrl_type::Symbol
    state_rep::Symbol
    para_type::Symbol
end

include("LindbladData.jl")
include("LindbladDynamics.jl")
include("LindbladWrapper.jl")

function set_ctrl(dynamics::Lindblad, ctrl)
    temp = deepcopy(dynamics)
    temp.data.ctrl = ctrl
    temp
end

function set_state(dynamics::Lindblad, state::AbstractVector)
    temp = deepcopy(dynamics)
    temp.data.ψ0 = state
    temp
end

function set_state(dynamics::Lindblad, state::AbstractMatrix)
    temp = deepcopy(dynamics)
    temp.data.ρ0 = state
    temp
end