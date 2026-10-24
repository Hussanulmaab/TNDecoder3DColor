module usc_3D_tn

using ITensors, ITensorMPS, LinearAlgebra
using ..TNDecoder3DColor: CSSCode, TensorType
using ..tools: check_arr, delta_arr

include("contract_depol_usc_3D_1.jl")
include("contraction_network_usc_3D_2.jl")
include("contract_circuit_usc_3D_3.jl")

export gen_tn_usc_3D_depol, generate_empty_cubictn, contract_cubictn_usc_3D

end