# Step 3: the contraction path. The order in which the tensors are contracted is found once per
# distance by cotengra (a Python library, run by tetrahedral_3D_cc_path.py) and saved to a file
# that every run and every sample reuses. The path does not depend on the syndrome, on p, or on
# which of tnI, tnX, tnY, tnZ is contracted: they all have the structure of the empty network.
# If codes/tetrahedral_3D_cc.jl or file 1 is changed, find the paths again.

# Write the structure of the network for cotengra: one line per tensor, in the order of
# tensor_labels_tetrahedral_3D_cc, listing its labels.
function write_structure_tetrahedral_3D_cc(empty_tn, structure_file::String)

    io = open(structure_file, "w")
    for labels in tensor_labels_tetrahedral_3D_cc(empty_tn)
        println(io, join(labels, " "))
    end
    close(io)

end

# Let cotengra search for max_time seconds and save the best path it finds to path_file.
function find_path_tetrahedral_3D_cc(empty_tn, path_file::String; max_time::Int=60)

    structure_file = tempname()
    write_structure_tetrahedral_3D_cc(empty_tn, structure_file)

    script = joinpath(@__DIR__, "tetrahedral_3D_cc_path.py")
    run(`python $script $structure_file $path_file $max_time`)
    rm(structure_file)

end

# Read a saved path: the pairs (a, b) of tensors to contract, in order. The tensors are numbered
# 1, 2, ... as in tensor_labels_tetrahedral_3D_cc, and the result of the k-th pair is tensor
# number (number of tensors + k).
function read_path_tetrahedral_3D_cc(path_file::String)

    path = Vector{Tuple{Int,Int}}()
    for line in eachline(path_file)
        if !startswith(line, "#")
            a, b = parse.(Int, split(line))
            push!(path, (a, b))
        end
    end

    return path
end
