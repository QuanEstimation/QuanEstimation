module Liouville
function liouville_commu(H::Array{T}) where {T <: Complex}
    kron(H, one(H))- kron(one(H), transpose(H))
end



function liouville_dissip(Γ::Matrix{T}) where {T <: Complex}
    kron(Γ,conj(Γ)) - 0.5*(kron(Γ' * Γ, one(Γ)) + kron(one(Γ), transpose(Γ) * conj(Γ)))
end

end
