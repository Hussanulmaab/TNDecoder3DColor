module TNDecoder3DColor

using ITensors
@static if ITensors.version() ≥ v"0.7.0"
    using ITensorMPS
end

include("tn_generation.jl")
include("tools.jl")
include("../codes/usc_3D.jl")
include("../codes/tetrahedral_3D_cc.jl")
include("../tensor_networks/usc_3D/usc_3D_tn.jl")
include("../tensor_networks/tetrahedral_3D_cc/tetrahedral_3D_cc.tn.jl")

export usc_3D,
    usc_3D_tn,
    tetrahedral_3D_cc,
    tetrahedral_3D_cc_tn,
    tools
end
