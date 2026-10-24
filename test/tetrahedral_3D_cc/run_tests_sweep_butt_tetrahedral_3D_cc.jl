include("../../zsrc/tndecoder3Dcolor.jl")
using .TNDecoder3DColor
using .TNDecoder3DColor.tetrahedral_3D_cc_tn
using .TNDecoder3DColor.tools
using Random

# The noise of Butt et al. (PRA 113, 042416, 2026), for comparing with their decoder: two runs at
# the same p, pure phase flips (Z on each qubit with probability p) and pure bit flips (X with
# probability p), perfect syndromes. Their logical error rate is the average of the two runs.
noise_models = ["phaseflip", "bitflip"]
distances = [3, 5]
# The p of Butt et al. (their Fig. 3a, digitized in results/) from 0.26% to 3.4%, and five more near
# the crossing. d = 7 and 9 leave out the lowest two (no failures there with these sample numbers);
# d = 9 stops at 2.34%, as its approximation is not reliable from about 2.5% on.
butt_points = [0.002637, 0.003793, 0.005456, 0.007848, 0.01129, 0.01624, 0.02336, 0.0336]
grid = sort([butt_points; 0.012; 0.014; 0.018; 0.020; 0.022])
physical_error_rates = Dict(3 => grid, 5 => grid, 7 => grid[3:end], 9 => [p for p in grid[3:end] if p < 0.025])
no_of_samples = Dict(3 => 10000, 5 => 10000, 7 => 1500, 9 => 1000)
# Tolerance 0.0 and a max_entries larger than any table is exact: phase flips up to d = 7
# (0.44 s per sample there); d = 9 needs the approximation. Bit flips use the approximation from
# d = 7 on (2 ms per sample instead of 0.7 s; they hardly ever fail there anyway).
tolerance = Dict("phaseflip" => Dict(3 => 0.0, 5 => 0.0, 7 => 0.0, 9 => 1e-8),
    "bitflip" => Dict(3 => 0.0, 5 => 0.0, 7 => 1e-8, 9 => 1e-8))
max_entries = Dict("phaseflip" => Dict(3 => 10^7, 5 => 10^7, 7 => 10^7, 9 => 10^5),
    "bitflip" => Dict(3 => 10^7, 5 => 10^7, 7 => 10^5, 9 => 10^5))

# One results file per noise model, in the format of results/sweep_tetrahedral_3D_cc.csv: a row per
# (d, p), written as soon as that point finishes.
#   fresh_start = true    the files are started again: the rows of earlier runs are deleted
#   fresh_start = false   the rows are added to the files (to add a distance to an earlier run)
fresh_start = true
results_file(noise) = joinpath(@__DIR__, "..", "..", "results", "sweep_butt_$(noise)_tetrahedral_3D_cc.csv")
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
    rates = Dict{String,Vector{Float64}}()

    for noise in noise_models
        println("-------------------distance $d, $noise----------------------")
        final_result_ratios = Float64[]

        for p in physical_error_rates[d]
            incorrect_count = 0
            # a fixed seed for every (noise, d, p), so a run can be repeated exactly
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
                    resI, resX, resY, resZ = sweep_tetrahedral_3D_cc_phaseflip(code, p, syndrome_x, syndrome_z, tolerance[noise][d], max_entries[noise][d])
                else
                    resI, resX, resY, resZ = sweep_tetrahedral_3D_cc_bitflip(code, p, syndrome_x, syndrome_z, tolerance[noise][d], max_entries[noise][d])
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
                println(io, "$d,$p,$(no_of_samples[d]),$incorrect_count,$(incorrect_count / no_of_samples[d]),$(tolerance[noise][d]),$(max_entries[noise][d])")
            end
        end

        rates[noise] = final_result_ratios
        # the logical error rate is the fraction of samples decoded to the wrong logical class
        logical_error_rates = final_result_ratios
        println("physical_error_rates = ", physical_error_rates[d])
        println("logical_error_rates = ", logical_error_rates)
        println("final_result_ratios (incorrect) = ", final_result_ratios)
    end

    # what Butt et al. report: the average of the two runs
    if haskey(rates, "phaseflip") && haskey(rates, "bitflip")
        println("-------------------distance $d, average of the two (Butt et al.)----------------------")
        println("physical_error_rates = ", physical_error_rates[d])
        println("logical_error_rates = ", (rates["phaseflip"] .+ rates["bitflip"]) ./ 2)
    end
end
