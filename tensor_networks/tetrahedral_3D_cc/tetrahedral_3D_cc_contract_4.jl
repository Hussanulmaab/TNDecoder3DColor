# Step 4: the contraction. One network (tnI, tnX, tnY or tnZ from file 2) is contracted pair by
# pair in the order of the saved path (file 3); OMEinsum does each pairwise contraction.
#
# A label shared by several tensors (a bit of a qubit) must not be summed over when the first two
# of its tensors meet, only when the last two do. So we count how many tensors still carry each
# label: the result of a pair keeps every label that another tensor still carries, and the
# labels carried by no other tensor are summed over.

function contract_tn_tetrahedral_3D_cc(tn::Vector{Array{Float64}}, tensor_labels::Vector{Vector{Int}}, path::Vector{Tuple{Int,Int}})

    tensors = copy(tn)
    labels = copy(tensor_labels)

    # how many tensors carry each label
    count = Dict{Int,Int}()
    for tensor_label in labels, x in tensor_label
        count[x] = get(count, x, 0) + 1
    end

    for (a, b) in path
        # tensors a and b are used up
        for x in labels[a]
            count[x] -= 1
        end
        for x in labels[b]
            count[x] -= 1
        end

        # the labels of the result: those another tensor still carries, in the order OMEinsum
        # builds them (only on a, only on b, on both), so it does not have to reorder the result
        new_labels = Int[]
        for x in labels[a]
            if count[x] > 0 && !(x in labels[b])
                push!(new_labels, x)
            end
        end
        for x in labels[b]
            if count[x] > 0 && !(x in labels[a])
                push!(new_labels, x)
            end
        end
        for x in labels[a]
            if count[x] > 0 && x in labels[b]
                push!(new_labels, x)
            end
        end
        for x in new_labels
            count[x] += 1
        end

        T = einsum(EinCode([labels[a], labels[b]], new_labels), (tensors[a], tensors[b]))
        push!(tensors, T)
        push!(labels, new_labels)

        # free the memory of tensors a and b
        tensors[a] = Float64[]
        tensors[b] = Float64[]
    end

    # the last result has no labels left: a single number, returned as (value, exponent)
    # like contract_cubictn_usc_3D, so the four results combine with tuple_add as in usc_3D
    return (tensors[end][], 0)
end
