@enum TensorType TT_DELTA=1 TT_CHECK=2 DEPOL=3

struct CSSCode{N}
    n_qubits::Int
    pos_qubits::Vector{NTuple{N,Float64}}
    pos_stab_X::Vector{NTuple{N,Float64}}
    pos_stab_Z::Vector{NTuple{N,Float64}}
    H_X::Matrix{UInt8}
    H_Z::Matrix{UInt8}
    L_X::Vector{UInt8}
    L_Z::Vector{UInt8}
end
