function qubits_usc_3D(d::Int)

    pos_qubits = Tuple{Int,Int,Int}[]
    qubits_indexing = Dict{Tuple{Int,Int,Int},Int}()
    for z = 1:(2d-1), y = 1:(2d-1), x = 1:(2d-1)

        if z%2==1
            if y%2==1 && x%2==1
                push!(pos_qubits, (x, y, z))
                qubits_indexing[(x, y, z)] = length(qubits_indexing) + 1
            elseif y%2==0 && x%2==0
                push!(pos_qubits, (x, y, z))
                qubits_indexing[(x, y, z)] = length(qubits_indexing) + 1
            end
        elseif z%2 == 0 && y%2==1 && x%2==0
            push!(pos_qubits, (x, y, z))
            qubits_indexing[(x, y, z)] = length(qubits_indexing) + 1
        end

    end

    return pos_qubits, qubits_indexing

end

function stab_X_usc_3D(d::Int, qubits_indexing::Dict{Tuple{Int,Int,Int},Int})

    pos_stab_X = Tuple{Int,Int,Int}[]
    for z = 1:(2d-1), y = 1:(2d-1), x = 1:(2d-1)
        if z%2==1 && y%2==1 && x%2==0
            push!(pos_stab_X, (x, y, z))
        end
    end

    H_X = Matrix{UInt8}(zeros(length(pos_stab_X), length(qubits_indexing)))

    for (stab_index, stab) in enumerate(pos_stab_X)
        x, y, z = stab

        neighbor_positions = [
            (x + 1, y, z),
            (x - 1, y, z),
            (x, y + 1, z),
            (x, y - 1, z),
            (x, y, z + 1),
            (x, y, z - 1),
        ]

        support_qubits_indices = Vector{Int}()

        for neighbor in neighbor_positions
            if haskey(qubits_indexing, neighbor)
                qubit_index = qubits_indexing[neighbor]
                push!(support_qubits_indices, qubit_index)
            end
        end

        H_X[stab_index, support_qubits_indices] .= 1
    end

    return pos_stab_X, H_X

end

function stab_Z_usc_3D(d::Int, qubits_indexing::Dict{Tuple{Int,Int,Int},Int})

    pos_stab_Z = Tuple{Int,Int,Int}[]
    for z = 1:(2d-1), y = 1:(2d-1), x = 1:(2d-1)
        if z%2==1 && y % 2 == 0 && x%2 == 1
            push!(pos_stab_Z, (x, y, z))
        elseif z%2==0
            if y == 1 && x%2 == 1
                push!(pos_stab_Z, (x, y, z))
            elseif y%2 == 0 && x%2 == 0
                push!(pos_stab_Z, (x, y, z))
            end
        end
    end

    H_Z = Matrix{UInt8}(zeros(length(pos_stab_Z), length(qubits_indexing)))

    for (stab_index, stab) in enumerate(pos_stab_Z)
        x, y, z = stab

        neighbor_positions = [
            (x + 1, y, z),
            (x - 1, y, z),
            (x, y + 1, z),
            (x, y - 1, z),
            (x, y, z + 1),
            (x, y, z - 1),
        ]

        support_qubits_indices = Vector{Int}()

        for neighbor in neighbor_positions
            if haskey(qubits_indexing, neighbor)
                qubit_index = qubits_indexing[neighbor]
                push!(support_qubits_indices, qubit_index)
            end
        end

        H_Z[stab_index, support_qubits_indices] .= 1
    end

    return pos_stab_Z, H_Z
end

function log_X_usc_3D(d::Int, qubits_indexing::Dict{Tuple{Int,Int,Int},Int})

    L_X = zeros(UInt8, length(qubits_indexing))

    for z in 1:2:(2d-1)
        for y in 1:2:(2d-1)
            coord = (1, y, z)
            qubit_index = qubits_indexing[coord]
            L_X[qubit_index] = 1
        end
    end

    return L_X
end

function log_Z_usc_3D(d::Int, qubits_indexing::Dict{Tuple{Int,Int,Int},Int})

    L_Z = zeros(UInt8, length(qubits_indexing))

    for x in 1:2:(2d-1)
        coord = (x, 1, 1)
        qubit_index = qubits_indexing[coord]
        L_Z[qubit_index] = 1
    end

    return L_Z
end

function usc_3D(d::Int)
    pos_qubits, qubits_indexing = qubits_usc_3D(d)
    n_qubits = length(pos_qubits)
    pos_stab_X, H_X = stab_X_usc_3D(d, qubits_indexing)
    pos_stab_Z, H_Z = stab_Z_usc_3D(d, qubits_indexing)
    L_X = log_X_usc_3D(d, qubits_indexing)
    L_Z = log_Z_usc_3D(d, qubits_indexing)

    return CSSCode{3}(
        n_qubits,
        pos_qubits,
        pos_stab_X,
        pos_stab_Z,
        H_X,
        H_Z,
        L_X,
        L_Z,
    )
end


