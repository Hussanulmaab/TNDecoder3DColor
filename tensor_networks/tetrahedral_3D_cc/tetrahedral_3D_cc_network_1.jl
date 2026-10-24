# Step 1: the empty tensor network of the tetrahedral code -- its structure only, no values.
#
# The structure is the same for tnI, tnX, tnY and tnZ and for every syndrome, so it is built
# once per distance. It comes from H_X and H_Z alone (no positions). Every tensor is a list of
# labels, and every label is a bit (dimension 2):
#     label i          the z bit of qubit i: the Z part of its error, seen by the X stabilizers
#     label n + i      the x bit of qubit i: the X part of its error, seen by the Z stabilizers
#     labels > 2n      the running parities inside the checks (below)
# The tensors are
#     one per qubit    labels (i, n + i): the probabilities of its four errors I, X, Y, Z
#     check pieces     a parity tensor on the bits of a stabilizer's qubits
# A bit is shared by its qubit tensor and by every check on that qubit, so a label can sit on
# more than two tensors (a hyperedge). A check on more than MAX_CHECK_LEGS_CC qubits is split,
# exactly, into a chain of 3-leg parity tensors that pass the running parity along.

const MAX_CHECK_LEGS_CC = 3

# The bits of a check as a chain of 3-leg pieces:
#     (b1, b2, r1), (r1, b3, r2), ..., (r_last, b_end-1, b_end)
# r1 is the parity of b1 and b2, r2 the parity up to b3, and so on, so the pieces together say
# the same as one parity tensor on all the bits. The syndrome bit goes on the last piece.
# The running parities get the labels next_label, next_label + 1, ...
function split_check_CC(bits::Vector{Int}, next_label::Int)
    if length(bits) <= MAX_CHECK_LEGS_CC
        return [bits], next_label
    end

    pieces = Vector{Vector{Int}}()
    running = next_label
    next_label += 1
    push!(pieces, [bits[1], bits[2], running])
    for k = 3:(length(bits)-2)
        new_running = next_label
        next_label += 1
        push!(pieces, [running, bits[k], new_running])
        running = new_running
    end
    push!(pieces, [running, bits[end-1], bits[end]])

    return pieces, next_label
end

function empty_tn_tetrahedral_3D_cc(css_code::CSSCode{3})
    n = css_code.n_qubits
    next_label = 2n + 1

    # the pieces of each X check: parity of the z bits of its qubits
    checks_X = Vector{Vector{Vector{Int}}}()
    for j = 1:size(css_code.H_X, 1)
        bits = Int[]
        for i = 1:n
            if css_code.H_X[j, i] == 1
                push!(bits, i)
            end
        end
        pieces, next_label = split_check_CC(bits, next_label)
        push!(checks_X, pieces)
    end

    # the pieces of each Z check: parity of the x bits of its qubits
    checks_Z = Vector{Vector{Vector{Int}}}()
    for j = 1:size(css_code.H_Z, 1)
        bits = Int[]
        for i = 1:n
            if css_code.H_Z[j, i] == 1
                push!(bits, n + i)
            end
        end
        pieces, next_label = split_check_CC(bits, next_label)
        push!(checks_Z, pieces)
    end

    return n, checks_X, checks_Z
end

# Every tensor of the network as its list of labels, in the order all later steps use:
# the qubits, then the pieces of the X checks, then the pieces of the Z checks.
function tensor_labels_tetrahedral_3D_cc(empty_tn)
    n, checks_X, checks_Z = empty_tn

    tensor_labels = Vector{Vector{Int}}()
    for i = 1:n
        push!(tensor_labels, [i, n + i])
    end
    for check in checks_X, piece in check
        push!(tensor_labels, piece)
    end
    for check in checks_Z, piece in check
        push!(tensor_labels, piece)
    end

    return tensor_labels
end
