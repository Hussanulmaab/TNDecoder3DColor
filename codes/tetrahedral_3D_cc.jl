# Tetrahedral 3D color code, distance d (odd):  [[ (d^3+d)/2, 1, d ]]
#
# Lattice. The code lives on the bitruncated cubic honeycomb: space filled with truncated
# octahedra ("cells"). A cell has 24 corners at the permutations of (0, ±1, ±2) around its
# centre, 6 square faces and 8 hexagonal faces, and its edges have length √2. The cell centres
# form a body-centred cubic lattice of spacing 4: cells whose centres differ by 4 along an axis
# share a square, cells whose centres differ by (±2, ±2, ±2) share a hexagon.
#
# Tetrahedron. We keep the part of the honeycomb inside the tetrahedron with corners
#     (0,0,0), (0,L,L), (L,0,L), (L,L,0),     L = 2(d-1).
# Its four faces pass through lattice corners only, so cutting creates no new corners: every
# face and cell below has qubits as its corners.
#     qubits          the lattice corners inside the tetrahedron (boundary included);
#                     q1 is the tetrahedron corner (0,0,0)
#     X stabilizers   the cells reaching into the tetrahedron, acting on their corners there
#     Z stabilizers   the faces: where two of those cells meet (bulk faces), and where a cell
#                     meets a face of the tetrahedron (boundary faces)
#     logical X       all qubits on the tetrahedron face x + y + z = 2L  (a membrane)
#     logical Z       the qubits on the tetrahedron edge from (0,0,0) to (L,L,0)  (a string)
# Every stabilizer is placed at the centroid of its qubits.
#
# Axes: x to the right, y into the screen, z up.

# the 24 corners of a cell, relative to its centre: all permutations of (0, ±1, ±2)
const CELL_OFFSETS_CC = [
    (0, 1, 2), (0, 1, -2), (0, -1, 2), (0, -1, -2),
    (0, 2, 1), (0, 2, -1), (0, -2, 1), (0, -2, -1),
    (1, 0, 2), (1, 0, -2), (-1, 0, 2), (-1, 0, -2),
    (2, 0, 1), (2, 0, -1), (-2, 0, 1), (-2, 0, -1),
    (1, 2, 0), (1, -2, 0), (-1, 2, 0), (-1, -2, 0),
    (2, 1, 0), (2, -1, 0), (-2, 1, 0), (-2, -1, 0),
]

# from the centre of a bulk face its two cells lie at +direction and -direction:
# (±1, ±1, ±1) for a hexagon, (±2, 0, 0) and its permutations for a square
const FACE_DIRECTIONS_CC = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1), (2, 0, 0), (0, 2, 0), (0, 0, 2)]

# the four faces of the tetrahedron, as planes  nx*x + ny*y + nz*z = bound  with outward normal
# (nx, ny, nz); the tetrahedron is the region where  nx*x + ny*y + nz*z <= bound  for all four
boundary_planes_CC(d::Int) = [(1, 1, 1, 4d - 4), (1, -1, -1, 0), (-1, 1, -1, 0), (-1, -1, 1, 0)]

tetrahedron_corners_CC(d::Int) = [(0, 0, 0), (0, 2d - 2, 2d - 2), (2d - 2, 0, 2d - 2), (2d - 2, 2d - 2, 0)]

# cell centres: the body-centred cubic lattice of spacing 4, i.e. a cubic lattice of spacing 4 and
# its body centres ((0,3,2) = (2,1,0) + (2,2,2) mod 4), placed so that (0,0,0) is a cell corner
function is_cell_centre_CC(x::Int, y::Int, z::Int)
    r = (mod(x, 4), mod(y, 4), mod(z, 4))
    return r == (2, 1, 0) || r == (0, 3, 2)
end

# a lattice corner is a corner of some cell
function is_lattice_corner_CC(x::Int, y::Int, z::Int)
    for (a, b, c) in CELL_OFFSETS_CC
        if is_cell_centre_CC(x - a, y - b, z - c)
            return true
        end
    end
    return false
end

# inside the tetrahedron, or on its boundary
function in_tetrahedron_CC(x::Int, y::Int, z::Int, d::Int)
    for (nx, ny, nz, bound) in boundary_planes_CC(d)
        if nx * x + ny * y + nz * z > bound
            return false
        end
    end
    return true
end

# A cell belongs to the code when part of it lies inside the tetrahedron. On each plane the cell
# centres sit at odd values of nx*x + ny*y + nz*z and the corners reach 3 further, so a centre at
# bound + 1 still has corners strictly inside, while one at bound + 3 only touches the plane.
function is_cell_tetrahedral_3D_cc(x::Int, y::Int, z::Int, d::Int)
    if !is_cell_centre_CC(x, y, z)
        return false
    end
    for (nx, ny, nz, bound) in boundary_planes_CC(d)
        if nx * x + ny * y + nz * z > bound + 1
            return false
        end
    end
    return true
end

# the cells of the code, in the order of the X stabilizers
# (all their centres lie in the bounding box 0:(2d-2) of the tetrahedron)
function cell_centres_tetrahedral_3D_cc(d::Int)
    cells = Tuple{Int,Int,Int}[]
    for z = 0:(2d-2), y = 0:(2d-2), x = 0:(2d-2)
        if is_cell_tetrahedral_3D_cc(x, y, z, d)
            push!(cells, (x, y, z))
        end
    end
    return cells
end

# colour 0, 1, 2 or 3 of a cell. Neighbouring cells always differ: centres 4 apart along an axis
# change x + y + z by 4 (colour by 2), centres (±2, ±2, ±2) apart change it by 2 or 6 (colour by an odd number).
cell_colour_tetrahedral_3D_cc(cell) = mod(div(cell[1] + cell[2] + cell[3] - 3, 2), 4)

# the corners of a cell that are qubits
function cell_corners_CC(cell, qubits_indexing::Dict{Tuple{Int,Int,Int},Int})
    corners = Tuple{Int,Int,Int}[]
    for (a, b, c) in CELL_OFFSETS_CC
        corner = (cell[1] + a, cell[2] + b, cell[3] + c)
        if haskey(qubits_indexing, corner)
            push!(corners, corner)
        end
    end
    return corners
end

# centroid (average position) of a list of points
function centroid_CC(points)
    n = length(points)
    x = sum(p[1] for p in points) / n
    y = sum(p[2] for p in points) / n
    z = sum(p[3] for p in points) / n
    return (x, y, z)
end

# lattice edges: the pairs of qubits at distance √2, the edge length of the honeycomb
# (no two lattice corners are closer, and a corner has exactly 4 neighbours at √2)
function lattice_edges_tetrahedral_3D_cc(pos_qubits)
    edges = Tuple{Int,Int}[]
    for i in 1:length(pos_qubits), j in (i+1):length(pos_qubits)
        dx = pos_qubits[i][1] - pos_qubits[j][1]
        dy = pos_qubits[i][2] - pos_qubits[j][2]
        dz = pos_qubits[i][3] - pos_qubits[j][3]
        if dx^2 + dy^2 + dz^2 == 2
            push!(edges, (i, j))
        end
    end
    return edges
end

function qubits_tetrahedral_3D_cc(d::Int)

    pos_qubits = Tuple{Int,Int,Int}[]
    qubits_indexing = Dict{Tuple{Int,Int,Int},Int}()
    for z = 0:(2d-2), y = 0:(2d-2), x = 0:(2d-2)

        # a qubit is a lattice corner inside the tetrahedron (or on its boundary)
        if is_lattice_corner_CC(x, y, z) && in_tetrahedron_CC(x, y, z, d)
            push!(pos_qubits, (x, y, z))
            qubits_indexing[(x, y, z)] = length(qubits_indexing) + 1
        end

    end

    return pos_qubits, qubits_indexing

end

function stab_X_tetrahedral_3D_cc(d::Int, qubits_indexing::Dict{Tuple{Int,Int,Int},Int})

    cells = cell_centres_tetrahedral_3D_cc(d)
    pos_stab_X = NTuple{3,Float64}[]
    H_X = Matrix{UInt8}(zeros(length(cells), length(qubits_indexing)))

    for (stab_index, cell) in enumerate(cells)
        corners = cell_corners_CC(cell, qubits_indexing)
        for corner in corners
            H_X[stab_index, qubits_indexing[corner]] = 1
        end

        # the centre of the part of the cell inside the tetrahedron
        push!(pos_stab_X, centroid_CC(corners))
    end

    return pos_stab_X, H_X

end

function stab_Z_tetrahedral_3D_cc(d::Int, qubits_indexing::Dict{Tuple{Int,Int,Int},Int})

    faces = Vector{Vector{Tuple{Int,Int,Int}}}()     # the qubits (positions) of each face

    # (a) bulk faces: the face between two neighbouring cells, made of the corners they share
    for z = 0:(2d-2), y = 0:(2d-2), x = 0:(2d-2)
        for (a, b, c) in FACE_DIRECTIONS_CC
            cell_1 = (x + a, y + b, z + c)
            cell_2 = (x - a, y - b, z - c)
            if is_cell_tetrahedral_3D_cc(cell_1..., d) && is_cell_tetrahedral_3D_cc(cell_2..., d)
                shared = intersect(cell_corners_CC(cell_1, qubits_indexing), cell_corners_CC(cell_2, qubits_indexing))
                push!(faces, shared)
            end
        end
    end

    # (b) boundary faces: the corners of a cell lying on a face of the tetrahedron
    for cell in cell_centres_tetrahedral_3D_cc(d)
        for (nx, ny, nz, bound) in boundary_planes_CC(d)
            on_plane = Tuple{Int,Int,Int}[]
            for corner in cell_corners_CC(cell, qubits_indexing)
                if nx * corner[1] + ny * corner[2] + nz * corner[3] == bound
                    push!(on_plane, corner)
                end
            end
            if !isempty(on_plane)
                push!(faces, on_plane)
            end
        end
    end

    pos_stab_Z = NTuple{3,Float64}[]
    H_Z = Matrix{UInt8}(zeros(length(faces), length(qubits_indexing)))

    for (stab_index, face) in enumerate(faces)
        for corner in face
            H_Z[stab_index, qubits_indexing[corner]] = 1
        end

        # the centre of the face
        push!(pos_stab_Z, centroid_CC(face))
    end

    return pos_stab_Z, H_Z
end

function log_X_tetrahedral_3D_cc(d::Int, qubits_indexing::Dict{Tuple{Int,Int,Int},Int})

    L_X = zeros(UInt8, length(qubits_indexing))

    # membrane: every qubit on the tetrahedron face x + y + z = 4d - 4
    for y in 0:(2d-2), x in 0:(2d-2)
        coord = (x, y, 4d - 4 - x - y)
        if haskey(qubits_indexing, coord)
            qubit_index = qubits_indexing[coord]
            L_X[qubit_index] = 1
        end
    end

    return L_X
end

function log_Z_tetrahedral_3D_cc(d::Int, qubits_indexing::Dict{Tuple{Int,Int,Int},Int})

    L_Z = zeros(UInt8, length(qubits_indexing))

    # string: the tetrahedron edge from (0,0,0) to (2d-2,2d-2,0)
    for t in 0:(2d-2)
        coord = (t, t, 0)
        if haskey(qubits_indexing, coord)
            qubit_index = qubits_indexing[coord]
            L_Z[qubit_index] = 1
        end
    end

    return L_Z
end

function tetrahedral_3D_cc(d::Int)
    pos_qubits, qubits_indexing = qubits_tetrahedral_3D_cc(d)
    n_qubits = length(pos_qubits)
    pos_stab_X, H_X = stab_X_tetrahedral_3D_cc(d, qubits_indexing)
    pos_stab_Z, H_Z = stab_Z_tetrahedral_3D_cc(d, qubits_indexing)
    L_X = log_X_tetrahedral_3D_cc(d, qubits_indexing)
    L_Z = log_Z_tetrahedral_3D_cc(d, qubits_indexing)

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
