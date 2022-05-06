# dynamics in Kraus rep.
struct Kraus{R} <: AbstractDynamics
    data::AbstractDynamicsData
    noise_type::Symbol
    ctrl_type::Symbol
    state_rep::Symbol
end

include("KrausData.jl")
include("KrausDynamics.jl")
include("KrausWrapper.jl")

function set_state(dynamics::Kraus, state::AbstractVector)
    temp = deepcopy(dynamics)
    temp.data.ψ0 = state
    temp
end
    
function set_state(dynamics::Kraus, state::AbstractMatrix)
    temp = deepcopy(dynamics)
    temp.data.ρ0 = state
    temp
end
