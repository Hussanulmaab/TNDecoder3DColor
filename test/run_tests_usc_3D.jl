include("../zsrc/tndecoder3Dcolor.jl")
using .TNDecoder3DColor
using .TNDecoder3DColor.usc_3D_tn
using .TNDecoder3DColor.tools

distances = [3, 5, 7, 9]
physical_error_rates = [0.01, 0.03, 0.05, 0.07]
no_of_samples = 2000
max_bond_dim = 2
sc_bond_dim = 2
svd_eps = 1e-16

class_parities = [(0, 0), (1, 0), (1, 1), (0, 1)]

for d in distances
    println("-------------------distance $d----------------------")
    usc = usc_3D(d)
    final_result_ratios = Float64[]

    for p in physical_error_rates
        incorrect_count = 0

        for sample in 1:no_of_samples
            # Generate one depolarizing-error sample and its syndromes.
            r = rand(usc.n_qubits)
            error_x = UInt8.(r .< (2p / 3))
            error_z = UInt8.((r .>= (p / 3)) .& (r .< p))

            syndrome_x = UInt8.((usc.H_X * error_z) .% 2)
            syndrome_z = UInt8.((usc.H_Z * error_x) .% 2)
            actual_parities = (
                Int(sum(error_z .& usc.L_X) % 2),
                Int(sum(error_x .& usc.L_Z) % 2),
            )

            # Build and contract the four tensor networks.
            tnI, tnX, tnY, tnZ = gen_tn_usc_3D_depol(usc, d, p, syndrome_x, syndrome_z)
            resI = contract_cubictn_usc_3D(deepcopy(tnI), max_bond_dim, sc_bond_dim; svd_eps=svd_eps, split_dim=4, ignore_bond_tensors=true)
            resX = contract_cubictn_usc_3D(deepcopy(tnX), max_bond_dim, sc_bond_dim; svd_eps=svd_eps, split_dim=4, ignore_bond_tensors=true)
            resY = contract_cubictn_usc_3D(deepcopy(tnY), max_bond_dim, sc_bond_dim; svd_eps=svd_eps, split_dim=4, ignore_bond_tensors=true)
            resZ = contract_cubictn_usc_3D(deepcopy(tnZ), max_bond_dim, sc_bond_dim; svd_eps=svd_eps, split_dim=4, ignore_bond_tensors=true)

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
    println("final_result_ratios (incorrect) = ", final_result_ratios)
end
