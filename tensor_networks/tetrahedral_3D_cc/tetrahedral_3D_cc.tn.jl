module tetrahedral_3D_cc_tn

using OMEinsum
using ..TNDecoder3DColor: CSSCode
using ..tools: check_arr

include("tetrahedral_3D_cc_network_1.jl")
include("tetrahedral_3D_cc_fill_2.jl")
include("tetrahedral_3D_cc_path_3.jl")
include("tetrahedral_3D_cc_contract_4.jl")
include("tetrahedral_3D_cc_sweep_5.jl")

export empty_tn_tetrahedral_3D_cc, tensor_labels_tetrahedral_3D_cc, gen_tn_tetrahedral_3D_cc_depol,
    find_path_tetrahedral_3D_cc, read_path_tetrahedral_3D_cc, contract_tn_tetrahedral_3D_cc,
    sweep_tetrahedral_3D_cc_depol, sweep_tetrahedral_3D_cc_phaseflip, sweep_tetrahedral_3D_cc_bitflip
end
