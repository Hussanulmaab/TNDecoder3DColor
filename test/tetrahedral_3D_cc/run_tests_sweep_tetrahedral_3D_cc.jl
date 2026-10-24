include("../../zsrc/tndecoder3Dcolor.jl")
using .TNDecoder3DColor
using .TNDecoder3DColor.tetrahedral_3D_cc_tn
using .TNDecoder3DColor.tools

distances = [3, 5, 7, 9]
# p = 0.01 is left out for d = 7 and 9: almost no failures there, only hours of running
physical_error_rates = Dict(3 => [0.01, 0.03, 0.05, 0.07], 5 => [0.01, 0.03, 0.05, 0.07],
    7 => [0.03, 0.05, 0.07], 9 => [0.03, 0.05, 0.07])
no_of_samples = Dict(3 => 10000, 5 => 10000, 7 => 1000, 9 => 500)
tolerance = 1e-8        # the sweep drops entries smaller than tolerance x the largest
max_entries = 10^5      # and keeps at most this many entries (tolerance = 0.0 and a huge max_entries = exact, only up to d = 5)

# one row per (d, p), written as soon as that point finishes; a new run starts a fresh file
results_file = joinpath(@__DIR__, "..", "..", "results", "sweep_tetrahedral_3D_cc.csv")
open(results_file, "w") do io
    println(io, "d,p,samples,incorrect,logical_error_rate,tolerance,max_entries")
end

class_parities = [(0, 0), (1, 0), (1, 1), (0, 1)]

for d in distances
    println("-------------------distance $d----------------------")
    code = tetrahedral_3D_cc(d)
    final_result_ratios = Float64[]

    for p in physical_error_rates[d]
        incorrect_count = 0

        for sample in 1:no_of_samples[d]
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

            # Sweep through the network: the same four numbers as contracting tnI, tnX, tnY, tnZ.
            resI, resX, resY, resZ = sweep_tetrahedral_3D_cc_depol(code, p, syndrome_x, syndrome_z, tolerance, max_entries)

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

        push!(final_result_ratios, incorrect_count / no_of_samples[d])
        println("p = $p: incorrect = $incorrect_count/$(no_of_samples[d]), correct = $(no_of_samples[d] - incorrect_count)/$(no_of_samples[d])")
        open(results_file, "a") do io
            println(io, "$d,$p,$(no_of_samples[d]),$incorrect_count,$(incorrect_count / no_of_samples[d]),$tolerance,$max_entries")
        end
    end

    # the logical error rate is the fraction of samples decoded to the wrong logical class
    logical_error_rates = final_result_ratios
    println("physical_error_rates = ", physical_error_rates[d])
    println("logical_error_rates = ", logical_error_rates)
    println("final_result_ratios (incorrect) = ", final_result_ratios)
end
