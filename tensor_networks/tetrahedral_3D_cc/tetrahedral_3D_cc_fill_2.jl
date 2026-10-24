# Step 2: the values. The empty network of file 1 is filled for one syndrome, giving the four
# networks tnI, tnX, tnY and tnZ of usc_3D. Each is a list of arrays in the order of
# tensor_labels_tetrahedral_3D_cc: the qubits, then the pieces of the X checks, then the
# pieces of the Z checks.
#
#     qubit tensor   PTensor[z+1, x+1], the probability of the error with Z part z and X part x.
#                    In tnX it is multiplied by (-1)^z on the qubits of L_X, in tnZ by (-1)^x on
#                    the qubits of L_Z, and in tnY by both.
#     check piece    1 when the parity of its bits is even; the last piece of each check is
#                    1 when the parity equals the syndrome bit
#
# Contracting the four networks gives four signed sums; combined as in run_tests_usc_3D they
# give the probabilities of the four logical classes.

function gen_tn_tetrahedral_3D_cc_general(css_code::CSSCode{3}, empty_tn, PTensor::Matrix{Float64}, syndrome_x::Vector{UInt8}, syndrome_z::Vector{UInt8})
    n, checks_X, checks_Z = empty_tn

    tnI = Vector{Array{Float64}}()
    tnX = Vector{Array{Float64}}()
    tnY = Vector{Array{Float64}}()
    tnZ = Vector{Array{Float64}}()

    # add depol tensors
    for i = 1:n
        sign_z = (css_code.L_X[i] == 1) ? [1., -1.] : [1., 1.]      # (-1)^z on the qubits of L_X
        sign_x = (css_code.L_Z[i] == 1) ? [1., -1.] : [1., 1.]      # (-1)^x on the qubits of L_Z

        push!(tnI, PTensor)
        push!(tnX, PTensor .* sign_z)
        push!(tnY, PTensor .* sign_z .* sign_x')
        push!(tnZ, PTensor .* sign_x')
    end

    # add check tensors (the same in all four networks)
    for j = 1:length(checks_X)
        pieces = checks_X[j]
        for k = 1:length(pieces)
            if k == length(pieces) && syndrome_x[j] == 1
                arr = check_arr(length(pieces[k]), 0., 1.)
            else
                arr = check_arr(length(pieces[k]), 1., 0.)
            end
            push!(tnI, arr)
            push!(tnX, arr)
            push!(tnY, arr)
            push!(tnZ, arr)
        end
    end

    for j = 1:length(checks_Z)
        pieces = checks_Z[j]
        for k = 1:length(pieces)
            if k == length(pieces) && syndrome_z[j] == 1
                arr = check_arr(length(pieces[k]), 0., 1.)
            else
                arr = check_arr(length(pieces[k]), 1., 0.)
            end
            push!(tnI, arr)
            push!(tnX, arr)
            push!(tnY, arr)
            push!(tnZ, arr)
        end
    end

    return tnI, tnX, tnY, tnZ
end

function gen_tn_tetrahedral_3D_cc_depol(css_code::CSSCode{3}, empty_tn, err_rate::Float64, syndrome_x::Vector{UInt8}, syndrome_z::Vector{UInt8})
    return gen_tn_tetrahedral_3D_cc_general(css_code, empty_tn, [(1. - err_rate) err_rate/3.; err_rate/3. err_rate/3.], syndrome_x, syndrome_z)
end
