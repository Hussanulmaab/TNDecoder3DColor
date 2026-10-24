# Step 5 (approximate): the sweep. Instead of contracting the network along a cotengra path
# (files 3 and 4), we sweep through it qubit by qubit, from the bottom of the tetrahedron to the
# top. The qubits are numbered z, then y, then x (codes/tetrahedral_3D_cc.jl), so this is simply
# their index order.
#
# What we carry along is the boundary of the part already swept, as a table
#     key => probability
# where the key holds the partial syndrome of every open check and the two logical parities.
# Each qubit splits every entry four ways (no error, X, Y, Z). After the last qubit of a check,
# only the entries that agree with its measured syndrome bit are kept. At the end the table has
# four entries: the probabilities of the four logical classes.
#
# The approximation, two settings (like svd_eps and max_bond_dim in usc_3D). After each qubit:
#     tolerance     drop the entries smaller than tolerance times the largest entry with the same
#                   logical parities
#     max_entries   then keep at most max_entries entries, the largest ones. The tolerance alone
#                   does not limit the size of the table, which explodes at high p (d = 7, p = 0.07:
#                   65 million entries with tolerance 1e-8)
# tolerance = 0.0 with a max_entries larger than the table gives the exact result (only possible up
# to d = 5). Neither setting is the error of the result: the largest entry can itself be dropped
# later by a check, so the class weights can be off by much more (d = 5, tolerance 1e-8: up to about
# 5e-5 of the winning class). What matters is that the chosen class does not change. A class far
# below the winner can be dropped altogether; it then counts as 0.
#
# A key is one 128-bit integer. Bit 0 is the parity of the Z part of the error on L_X, bit 1 the
# parity of the X part on L_Z, and every open check has one of the other bits, which is free again
# once the check is closed. 128 bits are enough up to d = 9 (it uses bits up to 106); a larger code
# will most likely run out of bits and stop with "invalid index: nothing".
#
# For pure phase flips or pure bit flips (the noise of Butt et al.) only one kind of check matters;
# a smaller sweep for them, with two branches per qubit and two classes, is at the end of the file.

# The bit of every check (supports: the qubits of each check). A check needs its bit from its
# first to its last qubit; the bit is then reused. A check that starts at the qubit where another
# ends must get a different bit (that qubit flips both), so new bits are given before old ones
# are freed.
function check_bits_CC(supports::Vector{Vector{Int}}, n::Int)

    bit = zeros(Int, length(supports))
    in_use = falses(128)
    in_use[1] = true        # bit 0: the L_X parity
    in_use[2] = true        # bit 1: the L_Z parity

    for i = 1:n
        for c = 1:length(supports)
            if supports[c][1] == i
                free = findfirst(.!in_use)
                in_use[free] = true
                bit[c] = free - 1
            end
        end
        for c = 1:length(supports)
            if supports[c][end] == i
                in_use[bit[c]+1] = false
            end
        end
    end

    return bit
end

function sweep_tetrahedral_3D_cc_general(css_code::CSSCode{3}, PTensor::Matrix{Float64}, syndrome_x::Vector{UInt8}, syndrome_z::Vector{UInt8}, tolerance::Float64, max_entries::Int)
    n = css_code.n_qubits
    n_X = size(css_code.H_X, 1)

    # all the checks, X checks first, with their syndrome bits
    supports = Vector{Vector{Int}}()
    for j = 1:n_X
        push!(supports, findall(css_code.H_X[j, :] .== 1))
    end
    for j = 1:size(css_code.H_Z, 1)
        push!(supports, findall(css_code.H_Z[j, :] .== 1))
    end
    syndrome = [syndrome_x; syndrome_z]
    bit = check_bits_CC(supports, n)

    # the bits flipped by a Z error (mask_z) or an X error (mask_x) on each qubit: its X checks
    # and the L_X parity, or its Z checks and the L_Z parity
    mask_z = zeros(UInt128, n)
    mask_x = zeros(UInt128, n)
    for c = 1:length(supports), i in supports[c]
        if c <= n_X
            mask_z[i] |= UInt128(1) << bit[c]
        else
            mask_x[i] |= UInt128(1) << bit[c]
        end
    end
    for i = 1:n
        if css_code.L_X[i] == 1
            mask_z[i] |= UInt128(1)
        end
        if css_code.L_Z[i] == 1
            mask_x[i] |= UInt128(2)
        end
    end

    # the checks closed at each qubit (their last qubit), and their syndrome bits
    close_mask = zeros(UInt128, n)
    close_value = zeros(UInt128, n)
    for c = 1:length(supports)
        i = supports[c][end]
        close_mask[i] |= UInt128(1) << bit[c]
        if syndrome[c] == 1
            close_value[i] |= UInt128(1) << bit[c]
        end
    end

    # two tables used in turn: each step fills new_table from table, then they swap
    # (empty! keeps the memory of a table, so it is not allocated again for every qubit)
    table = Dict{UInt128,Float64}(UInt128(0) => 1.0)
    new_table = Dict{UInt128,Float64}()
    for i = 1:n
        # the four errors on qubit i, (z, x) = (0, 0) none, (0, 1) X, (1, 1) Y, (1, 0) Z,
        # with probability PTensor[z+1, x+1]
        empty!(new_table)
        for (key, weight) in table, z = 0:1, x = 0:1
            new_key = key
            if z == 1
                new_key ⊻= mask_z[i]
            end
            if x == 1
                new_key ⊻= mask_x[i]
            end
            new_table[new_key] = get(new_table, new_key, 0.0) + weight * PTensor[z+1, x+1]
        end
        table, new_table = new_table, table

        # the checks whose last qubit is i: keep the entries that agree with their syndrome bits,
        # then free their bits
        if close_mask[i] != 0
            empty!(new_table)
            for (key, weight) in table
                if key & close_mask[i] == close_value[i]
                    new_key = key & ~close_mask[i]
                    new_table[new_key] = get(new_table, new_key, 0.0) + weight
                end
            end
            table, new_table = new_table, table
        end

        # the approximation: drop the entries much smaller than the largest one with the same
        # logical parities (bits 0 and 1 of the key)
        if tolerance > 0
            largest = zeros(4)
            for (key, weight) in table
                sector = Int(key & 3) + 1
                largest[sector] = max(largest[sector], weight)
            end
            empty!(new_table)
            for (key, weight) in table
                if weight >= tolerance * largest[Int(key & 3)+1]
                    new_table[key] = weight
                end
            end
            table, new_table = new_table, table
        end

        # the cap: keep at most max_entries entries, the largest ones
        if length(table) > max_entries
            weights = collect(values(table))
            smallest_kept = partialsort!(weights, max_entries, rev=true)
            empty!(new_table)
            for (key, weight) in table
                if weight >= smallest_kept
                    new_table[key] = weight
                end
            end
            table, new_table = new_table, table
        end
    end

    # at most four entries are left (a class dropped by the approximation counts as 0), the
    # classes (a, b): a the L_X parity (bit 0), b the L_Z parity (bit 1)
    c00 = get(table, UInt128(0), 0.)
    c10 = get(table, UInt128(1), 0.)
    c01 = get(table, UInt128(2), 0.)
    c11 = get(table, UInt128(3), 0.)

    # the same four numbers as contracting tnI, tnX, tnY, tnZ (files 2 and 4): tnX has the sign
    # (-1)^a, tnZ (-1)^b and tnY (-1)^(a+b). As (value, exponent) like contract_cubictn_usc_3D,
    # so they combine with tuple_add as in usc_3D.
    resI = c00 + c10 + c01 + c11
    resX = c00 - c10 + c01 - c11
    resY = c00 - c10 - c01 + c11
    resZ = c00 + c10 - c01 - c11

    return (resI, 0), (resX, 0), (resY, 0), (resZ, 0)
end

function sweep_tetrahedral_3D_cc_depol(css_code::CSSCode{3}, err_rate::Float64, syndrome_x::Vector{UInt8}, syndrome_z::Vector{UInt8}, tolerance::Float64, max_entries::Int)
    return sweep_tetrahedral_3D_cc_general(css_code, [(1. - err_rate) err_rate/3.; err_rate/3. err_rate/3.], syndrome_x, syndrome_z, tolerance, max_entries)
end

# The sweep for one kind of error only, on each qubit with probability err_rate: Z errors, seen by
# the X checks (H_X) and changing the L_X parity, or X errors, seen by the Z checks (H_Z) and
# changing the L_Z parity. It is the sweep above with two branches per qubit (no error, error) and
# one logical parity (bit 0 of the key). It returns the probabilities of the two classes, parity 0
# and parity 1.
function sweep_tetrahedral_3D_cc_one_type(H::Matrix{UInt8}, L::Vector{UInt8}, err_rate::Float64, syndrome::Vector{UInt8}, tolerance::Float64, max_entries::Int)
    n = size(H, 2)

    # the checks with their syndrome bits
    supports = Vector{Vector{Int}}()
    for j = 1:size(H, 1)
        push!(supports, findall(H[j, :] .== 1))
    end
    bit = check_bits_CC(supports, n)

    # the bits flipped by an error on each qubit: its checks and the logical parity
    mask = zeros(UInt128, n)
    for c = 1:length(supports), i in supports[c]
        mask[i] |= UInt128(1) << bit[c]
    end
    for i = 1:n
        if L[i] == 1
            mask[i] |= UInt128(1)
        end
    end

    # the checks closed at each qubit (their last qubit), and their syndrome bits
    close_mask = zeros(UInt128, n)
    close_value = zeros(UInt128, n)
    for c = 1:length(supports)
        i = supports[c][end]
        close_mask[i] |= UInt128(1) << bit[c]
        if syndrome[c] == 1
            close_value[i] |= UInt128(1) << bit[c]
        end
    end

    table = Dict{UInt128,Float64}(UInt128(0) => 1.0)
    new_table = Dict{UInt128,Float64}()
    for i = 1:n
        # no error on qubit i (probability 1 - err_rate) or an error (probability err_rate)
        empty!(new_table)
        for (key, weight) in table
            new_table[key] = get(new_table, key, 0.0) + weight * (1. - err_rate)
            new_key = key ⊻ mask[i]
            new_table[new_key] = get(new_table, new_key, 0.0) + weight * err_rate
        end
        table, new_table = new_table, table

        # the checks whose last qubit is i, as in the sweep above
        if close_mask[i] != 0
            empty!(new_table)
            for (key, weight) in table
                if key & close_mask[i] == close_value[i]
                    new_key = key & ~close_mask[i]
                    new_table[new_key] = get(new_table, new_key, 0.0) + weight
                end
            end
            table, new_table = new_table, table
        end

        # the approximation, as in the sweep above, with two logical sectors (bit 0 of the key)
        if tolerance > 0
            largest = zeros(2)
            for (key, weight) in table
                sector = Int(key & 1) + 1
                largest[sector] = max(largest[sector], weight)
            end
            empty!(new_table)
            for (key, weight) in table
                if weight >= tolerance * largest[Int(key & 1)+1]
                    new_table[key] = weight
                end
            end
            table, new_table = new_table, table
        end

        # the cap, as in the sweep above
        if length(table) > max_entries
            weights = collect(values(table))
            smallest_kept = partialsort!(weights, max_entries, rev=true)
            empty!(new_table)
            for (key, weight) in table
                if weight >= smallest_kept
                    new_table[key] = weight
                end
            end
            table, new_table = new_table, table
        end
    end

    return get(table, UInt128(0), 0.), get(table, UInt128(1), 0.)
end

# Pure phase flips: Z errors only (syndrome_z is then all zero and not used). The classes are
# c00 and c10 (a = 0 or 1), c01 = c11 = 0, and the four numbers are those of the general sweep
# for them: resI = resZ = c00 + c10, resX = resY = c00 - c10.
function sweep_tetrahedral_3D_cc_phaseflip(css_code::CSSCode{3}, err_rate::Float64, syndrome_x::Vector{UInt8}, syndrome_z::Vector{UInt8}, tolerance::Float64, max_entries::Int)
    c00, c10 = sweep_tetrahedral_3D_cc_one_type(css_code.H_X, css_code.L_X, err_rate, syndrome_x, tolerance, max_entries)
    return (c00 + c10, 0), (c00 - c10, 0), (c00 - c10, 0), (c00 + c10, 0)
end

# Pure bit flips: X errors only (syndrome_x is then all zero and not used). The classes are
# c00 and c01 (b = 0 or 1), c10 = c11 = 0: resI = resX = c00 + c01, resY = resZ = c00 - c01.
function sweep_tetrahedral_3D_cc_bitflip(css_code::CSSCode{3}, err_rate::Float64, syndrome_x::Vector{UInt8}, syndrome_z::Vector{UInt8}, tolerance::Float64, max_entries::Int)
    c00, c01 = sweep_tetrahedral_3D_cc_one_type(css_code.H_Z, css_code.L_Z, err_rate, syndrome_z, tolerance, max_entries)
    return (c00 + c01, 0), (c00 + c01, 0), (c00 - c01, 0), (c00 - c01, 0)
end
