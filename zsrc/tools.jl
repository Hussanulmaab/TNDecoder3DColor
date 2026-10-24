module tools

using ITensors

export delta_arr, check_arr, tuple_add, tn_details

function delta_arr(num_legs::Int64, val1::Float64, val2::Float64)
    t = zeros(2*ones(Int, num_legs)...)
    t[begin] = val1
    t[end] = val2
    return t
end

function check_arr(num_legs::Int64, val1::Float64, val2::Float64)
    t = zeros(2*ones(Int, num_legs)...)
    for i = 1:(2^num_legs)
        t[i] = (count_ones(i-1)%2 == 0) ? val1 : val2
    end
    return t
end

function tuple_add(tlist)
    all(t -> isfinite(t[1]), tlist) || error("Cannot add non-finite contraction results")
    nonzero = [t for t in tlist if t[1] != 0.0]
    isempty(nonzero) && return (0.0, 0)

    max_exp = maximum(t[2] for t in nonzero)
    base = sum(ldexp(t[1], t[2] - max_exp) for t in nonzero)
    base == 0.0 && return (0.0, 0)

    fraction, shift = frexp(base)
    return (2 * fraction, max_exp + shift - 1)
end

function tn_details(tn)
    positions = keys(tn.tensors)

    println("bonds = ", sum(length, values(tn.adj)) ÷ 2)
    println("no_of_tensors = ", length(tn.tensors))
    println("extreme_x = ", extrema(pos[1] for pos in positions))
    println("extreme_y = ", extrema(pos[2] for pos in positions))
    println("extreme_z = ", extrema(pos[3] for pos in positions))
    println("stored_bond_matrices = ", length(tn.bond_matrices))
    println("tensor_orders = ",
        sort(unique(length(inds(T)) for T in values(tn.tensors))))

    return nothing
end

end
