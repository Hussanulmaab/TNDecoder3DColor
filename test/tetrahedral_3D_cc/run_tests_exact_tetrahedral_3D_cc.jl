include("../../zsrc/tndecoder3Dcolor.jl")
using .TNDecoder3DColor
using .TNDecoder3DColor.tetrahedral_3D_cc_tn
using .TNDecoder3DColor.tools

# exact contraction is possible up to d = 5 (d = 7 and 9 would need tensors of 2^61 and 2^100 numbers)
distances = [5]
physical_error_rates = [0.01, 0.03, 0.05, 0.07]
no_of_samples = 1

class_parities = [(0, 0), (1, 0), (1, 1), (0, 1)]

for d in distances
    println("-------------------distance $d----------------------")
    code = tetrahedral_3D_cc(d)
    final_result_ratios = Float64[]

    # The empty network and its contraction path are the same for every sample, so they are made
    # once per distance. The path is found by cotengra only if it has not been saved yet.
    empty_tn = empty_tn_tetrahedral_3D_cc(code)
    tensor_labels = tensor_labels_tetrahedral_3D_cc(empty_tn)
    path_file = joinpath(@__DIR__, "..", "..", "tensor_networks", "tetrahedral_3D_cc", "paths", "path_d$(d).txt")
    if !isfile(path_file)
        find_path_tetrahedral_3D_cc(empty_tn, path_file; max_time=60)
    end
    path = read_path_tetrahedral_3D_cc(path_file)

    for p in physical_error_rates
        incorrect_count = 0

        for sample in 1:no_of_samples
            # Generate one depolarizing-error sample and its syndromes.
            r = rand(code.n_qubits)
            error_x = UInt8.(r .< (2p / 3))
            error_z = UInt8.((r .>= (p / 3)) .& (r .< p))

            syndrome_x = UInt8.((code.H_X * error_z) .% 2)
            syndrome_z = UInt8.((code.H_Z * error_x) .% 2)
            actual_parities = (
                Int(sum(error_z .& code.L_X) % 2),
                Int(sum(error_x .& code.L_Z) % 2),
            )

            # Build and contract the four tensor networks.
            tnI, tnX, tnY, tnZ = gen_tn_tetrahedral_3D_cc_depol(code, empty_tn, p, syndrome_x, syndrome_z)
            resI = contract_tn_tetrahedral_3D_cc(tnI, tensor_labels, path)
            resX = contract_tn_tetrahedral_3D_cc(tnX, tensor_labels, path)
            resY = contract_tn_tetrahedral_3D_cc(tnY, tensor_labels, path)
            resZ = contract_tn_tetrahedral_3D_cc(tnZ, tensor_labels, path)

            A00 = tuple_add([resI, resY, resZ, resX])
            A01 = tuple_add([resI, (-resY[1], resY[2]), resZ, (-resX[1], resX[2])])
            A10 = tuple_add([resI, resY, (-resZ[1], resZ[2]), (-resX[1], resX[2])])
            A11 = tuple_add([resI, (-resY[1], resY[2]), (-resZ[1], resZ[2]), resX])

            # Put the class weights on the same scale and choose the largest.
            class_scores = [A00, A01, A10, A11]
            common_exp = maximum(t[2] for t in class_scores if t[1] != 0.0)
            weights = [ldexp(t[1], t[2] - common_exp) for t in class_scores]
            predicted_parities = class_parities[argmax(weights)]

            if predicted_parities != actual_parities
                incorrect_count += 1
            end
        end

        push!(final_result_ratios, incorrect_count / no_of_samples)
        println("p = $p: incorrect = $incorrect_count/$no_of_samples, correct = $(no_of_samples - incorrect_count)/$no_of_samples")
    end

    # the logical error rate is the fraction of samples decoded to the wrong logical class
    logical_error_rates = final_result_ratios
    println("physical_error_rates = ", physical_error_rates)
    println("logical_error_rates = ", logical_error_rates)
end
