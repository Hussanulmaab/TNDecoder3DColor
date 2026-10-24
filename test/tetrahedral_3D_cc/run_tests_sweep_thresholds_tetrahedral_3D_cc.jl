include("../../zsrc/tndecoder3Dcolor.jl")
using .TNDecoder3DColor
using .TNDecoder3DColor.tetrahedral_3D_cc_tn
using .TNDecoder3DColor.tools
using Random

# The two thresholds of the code, each from one kind of error with its own range of p (perfect
# syndromes):
#   phase flips   Z on each qubit with probability p, decoded with the cells. The logical Z is a
#                 string; the optimal threshold is about 1.9% (Kubica et al., PRL 120, 180501 (2018))
#   bit flips     X on each qubit with probability p, decoded with all the faces. The logical X is a
#                 membrane; the optimal threshold is about 27.6% (same paper)
# For independent X and Z errors the two do not affect each other, so any pair of rates (pX, pZ)
# can be read off the two curves.
noise_models = ["phaseflip", "bitflip"]
distances = [3, 5]
# Phase flips: dense around 1.9%. d = 7 and 9 leave out 0.5% and 0.8% (no failures there with these
# sample numbers); d = 9 stops below 2.5%, as its approximation is not reliable from about 2.5% on.
phase_grid = [0.005, 0.008, 0.010, 0.012, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020, 0.021, 0.022, 0.024, 0.027, 0.030]
# Bit flips: up to 33%, around 27.6%. d = 7 leaves out 5% and 10% (no failures there); d = 9 is not
# run (its exact table does not fit in memory, and the approximation is not reliable at these p).
bit_grid = [0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.33]
physical_error_rates = Dict(
    "phaseflip" => Dict(3 => phase_grid, 5 => phase_grid, 7 => phase_grid[3:end], 9 => [p for p in phase_grid[3:end] if p < 0.025]),
    "bitflip" => Dict(3 => bit_grid, 5 => bit_grid, 7 => bit_grid[3:end]))
no_of_samples = Dict(3 => 10000, 5 => 10000, 7 => 2000, 9 => 1000)
# Tolerance 0.0 and a max_entries larger than any table is exact: possible up to d = 7 for both
# (0.44 s per sample for phase flips, 0.7 s for bit flips there); d = 9 needs the approximation.
tolerance = Dict(3 => 0.0, 5 => 0.0, 7 => 0.0, 9 => 1e-8)
max_entries = Dict(3 => 10^7, 5 => 10^7, 7 => 10^7, 9 => 10^5)

# One results file per noise model, in the format of results/sweep_tetrahedral_3D_cc.csv: a row per
# (d, p), written as soon as that point finishes.
#   fresh_start = true    the files are started again: the rows of earlier runs are deleted
#   fresh_start = false   the rows are added to the files (to add a distance to an earlier run)
fresh_start = true
results_file(noise) = joinpath(@__DIR__, "..", "..", "results", "sweep_thresholds_$(noise)_tetrahedral_3D_cc.csv")
for noise in noise_models
    if fresh_start || !isfile(results_file(noise))
        open(results_file(noise), "w") do io
            println(io, "d,p,samples,incorrect,logical_error_rate,tolerance,max_entries")
        end
    end
end

class_parities = [(0, 0), (1, 0), (1, 1), (0, 1)]

for d in distances
    code = tetrahedral_3D_cc(d)

    for noise in noise_models
        # bit flips are not run at d = 9
        if !haskey(physical_error_rates[noise], d)
            continue
        end
        println("-------------------distance $d, $noise----------------------")
        final_result_ratios = Float64[]

        for p in physical_error_rates[noise][d]
            incorrect_count = 0
            # a fixed seed for every (noise, d, p), so a run can be repeated exactly (the same seed as
            # in the Butt et al. test, so a p that both have gives the same samples)
            rng = Xoshiro((noise == "phaseflip" ? 1 : 2) * 10^7 + d * 10^5 + round(Int, 10^4 * p))

            for sample in 1:no_of_samples[d]
                # One error type only: Z errors (phase flips) or X errors (bit flips), each with
                # probability p. The other error vector, and so the other syndrome, is all zero.
                flips = UInt8.(rand(rng, code.n_qubits) .< p)
                no_flips = zeros(UInt8, code.n_qubits)
                error_z = noise == "phaseflip" ? flips : no_flips
                error_x = noise == "bitflip" ? flips : no_flips

                syndrome_x = UInt8.((code.H_X * error_z) .% 2)
                syndrome_z = UInt8.((code.H_Z * error_x) .% 2)
                actual_parities = (
                    Int(sum(error_z .& code.L_X) % 2),
                    Int(sum(error_x .& code.L_Z) % 2),
                )

                # Sweep through the network: the same four numbers as for depolarizing noise.
                if noise == "phaseflip"
                    resI, resX, resY, resZ = sweep_tetrahedral_3D_cc_phaseflip(code, p, syndrome_x, syndrome_z, tolerance[d], max_entries[d])
                else
                    resI, resX, resY, resZ = sweep_tetrahedral_3D_cc_bitflip(code, p, syndrome_x, syndrome_z, tolerance[d], max_entries[d])
                end

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
            open(results_file(noise), "a") do io
                println(io, "$d,$p,$(no_of_samples[d]),$incorrect_count,$(incorrect_count / no_of_samples[d]),$(tolerance[d]),$(max_entries[d])")
            end
        end

        # the logical error rate is the fraction of samples decoded to the wrong logical class
        logical_error_rates = final_result_ratios
        println("physical_error_rates = ", physical_error_rates[noise][d])
        println("logical_error_rates = ", logical_error_rates)
        println("final_result_ratios (incorrect) = ", final_result_ratios)
    end
end
