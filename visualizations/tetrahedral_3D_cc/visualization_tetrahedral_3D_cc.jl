# visualization_tetrahedral_3D_cc.jl -- interactive 3D viewer for the tetrahedral 3D color code
#
#   julia visualizations/tetrahedral_3D_cc/visualization_tetrahedral_3D_cc.jl        distance 3
#   julia visualizations/tetrahedral_3D_cc/visualization_tetrahedral_3D_cc.jl 5      distance 5
#   in the REPL:  include("visualizations/tetrahedral_3D_cc/visualization_tetrahedral_3D_cc.jl"); show_viewer(5)
#
# Everything drawn comes from codes/tetrahedral_3D_cc.jl, the same file the decoder uses: the
# qubit positions, H_X and H_Z (every stabilizer-to-qubit line is one nonzero entry of a row),
# the stabilizer positions (centroids of their qubits), the logicals, the cells and their colours.
#
# Mouse   left-drag rotate, right-drag pan, scroll zoom; a click on a qubit, lattice edge, face,
#         cell or stabilizer selects it and the panel describes it
# Panel   the distance d, what to show, a slab that cuts the tetrahedron into layers, sliders
#         that pick an X or Z stabilizer, and "isolate the selection" to look at it alone;
#         "save picture" (or the s key) saves the 3D scene as a PNG in the folder figure_exports
#         next to this file

using GLMakie
using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "zsrc", "tn_generation.jl"))      # the CSSCode struct
include(joinpath(@__DIR__, "..", "..", "codes", "tetrahedral_3D_cc.jl"))   # the code

const GB = GLMakie.Makie.GeometryBasics

# Colours. Cells, X stabilizers and X connections take the colour of their cell.
const CELL_COLOURS = [RGBf(0.86, 0.20, 0.20), RGBf(0.20, 0.66, 0.27), RGBf(0.20, 0.42, 0.88), RGBf(0.96, 0.76, 0.10)]
const CELL_COLOUR_NAMES = ["red", "green", "blue", "yellow"]
const QUBIT_COLOUR = RGBf(0.10, 0.10, 0.10)
const EDGE_COLOUR = RGBf(0.35, 0.35, 0.35)
const BULK_FACE_COLOUR = RGBf(0.60, 0.64, 0.74)
const Z_COLOUR = RGBf(0.45, 0.20, 0.75)            # Z stabilizers and Z connections: violet
const SELECTION_COLOUR = RGBf(1.00, 0.50, 0.00)    # orange
const LOGICAL_X_COLOUR = RGBf(0.00, 0.60, 0.60)    # teal
const LOGICAL_Z_COLOUR = RGBf(0.90, 0.10, 0.55)    # pink

colour_name(c) = c < 0 ? "none" : CELL_COLOUR_NAMES[c+1]

# the planes of boundary_planes_CC, written out
plane_label(d, i) = ["x + y + z = $(4d - 4)", "x - y - z = 0", "-x + y - z = 0", "-x - y + z = 0"][i]

# The slab cut keeps what lies between two values of one of these forms. "depth" is how far
# below the surface of the tetrahedron a point is, so cutting it away peels off the outside.
const CUT_FORMS = ["x+y+z", "x-y-z", "-x+y-z", "-x-y+z", "x", "y", "z", "depth"]

function cut_value(form, p, d)
    if form == "x+y+z"
        return p[1] + p[2] + p[3]
    elseif form == "x-y-z"
        return p[1] - p[2] - p[3]
    elseif form == "-x+y-z"
        return -p[1] + p[2] - p[3]
    elseif form == "-x-y+z"
        return -p[1] - p[2] + p[3]
    elseif form == "x"
        return p[1]
    elseif form == "y"
        return p[2]
    elseif form == "z"
        return p[3]
    end
    depth = Inf
    for (nx, ny, nz, bound) in boundary_planes_CC(d)
        depth = min(depth, bound - (nx * p[1] + ny * p[2] + nz * p[3]))
    end
    return depth
end

# the lowest and highest value of a form over the qubits
function form_extrema(G, form)
    values = [cut_value(form, p, G.d) for p in G.qubits]
    return round(Int, minimum(values)), round(Int, maximum(values))
end

function is_valid_distance(s)
    d = tryparse(Int, s)
    return d !== nothing && d >= 3 && isodd(d)
end

fmt(p) = "(" * join([string(round(Float64(t), digits = 3)) for t in p], ", ") * ")"

# a list of numbers as text, 10 on a line
function number_list(numbers)
    lines = String[]
    for start in 1:10:length(numbers)
        push!(lines, join(numbers[start:min(start + 9, end)], ", "))
    end
    return join(lines, ",\n    ")
end

# ------------------------------------------------------------------------- the code's geometry

# the qubits of a flat face in order around it (sorted by angle in the plane of the face)
function polygon_order(support, positions)
    points = [collect(Float64, positions[j]) for j in support]
    centre = sum(points) / length(points)
    u = points[1] - centre
    normal = zeros(3)
    for p in points
        n = cross(u, p - centre)
        if norm(n) > norm(normal)
            normal = n
        end
    end
    w = cross(normal, u)
    angles = [atan(dot(p - centre, w), dot(p - centre, u)) for p in points]
    return support[sortperm(angles)]
end

# everything the viewer draws for distance d, read from codes/tetrahedral_3D_cc.jl
function code_geometry(d)
    code = tetrahedral_3D_cc(d)
    n_X = size(code.H_X, 1)
    n_Z = size(code.H_Z, 1)

    qubits = [Point3f(p...) for p in code.pos_qubits]
    x_stabs = [Point3f(p...) for p in code.pos_stab_X]
    z_stabs = [Point3f(p...) for p in code.pos_stab_Z]

    # the qubits of each stabilizer: the nonzero entries of its row of H_X or H_Z
    x_support = [findall(code.H_X[s, :] .== 1) for s in 1:n_X]
    z_support = [findall(code.H_Z[f, :] .== 1) for f in 1:n_Z]

    cells = cell_centres_tetrahedral_3D_cc(d)       # cell s is X stabilizer s
    colour = [cell_colour_tetrahedral_3D_cc(c) for c in cells]
    edges = lattice_edges_tetrahedral_3D_cc(code.pos_qubits)
    polygons = [polygon_order(z_support[f], code.pos_qubits) for f in 1:n_Z]

    # the faces of a cell are the faces whose qubits all belong to it;
    # a bulk face then has two cells and a boundary face one
    cell_faces = [[f for f in 1:n_Z if issubset(z_support[f], x_support[s])] for s in 1:n_X]
    face_cells = [[s for s in 1:n_X if f in cell_faces[s]] for f in 1:n_Z]

    # the face of the tetrahedron each boundary face lies in (0 for bulk faces)
    face_plane = zeros(Int, n_Z)
    planes = boundary_planes_CC(d)
    for f in 1:n_Z
        if length(face_cells[f]) == 1
            for i in 1:4
                nx, ny, nz, bound = planes[i]
                values = [nx * code.pos_qubits[j][1] + ny * code.pos_qubits[j][2] + nz * code.pos_qubits[j][3] for j in z_support[f]]
                if all(values .== bound)
                    face_plane[f] = i
                end
            end
        end
    end

    # the colour missing on each face of the tetrahedron: no cell of that colour touches it
    missing_colour = Int[]
    for i in 1:4
        present = [colour[face_cells[f][1]] for f in 1:n_Z if face_plane[f] == i]
        absent = setdiff(0:3, present)
        push!(missing_colour, length(absent) == 1 ? absent[1] : -1)
    end

    logical_x = findall(code.L_X .== 1)
    logical_z = findall(code.L_Z .== 1)
    corners = [Point3f(c...) for c in tetrahedron_corners_CC(d)]

    return (; d, code, qubits, x_stabs, z_stabs, x_support, z_support, cells, colour, edges,
        polygons, cell_faces, face_cells, face_plane, missing_colour, logical_x, logical_z, corners)
end

stab_position(G, kind, s) = kind == :X ? G.x_stabs[s] : G.z_stabs[s]

# One triangle mesh for a list of faces, each face a fan of triangles around its centre.
# owner[v] is the entry of `owners` that vertex v came from, so that a click on the mesh can
# be traced back to its face or cell.
function faces_mesh(G, faces, owners)
    points = Point3f[]
    triangles = GB.GLTriangleFace[]
    owner = Int[]
    for (f, o) in zip(faces, owners)
        corners = G.qubits[G.polygons[f]]
        m = length(corners)
        start = length(points) + 1
        push!(points, sum(corners) / m)
        append!(points, corners)
        for k in 1:m
            push!(triangles, GB.GLTriangleFace(start, start + k, start + mod1(k + 1, m)))
        end
        append!(owner, fill(o, m + 1))
    end
    if isempty(triangles)              # nothing to draw: one zero-size triangle
        points = [Point3f(0, 0, 0), Point3f(0, 0, 0), Point3f(0, 0, 0)]
        triangles = [GB.GLTriangleFace(1, 2, 3)]
        owner = [0, 0, 0]
    end
    return GB.Mesh(points, triangles), owner
end

function edge_segments(G, edge_list)
    points = Point3f[]
    for e in edge_list
        a, b = G.edges[e]
        push!(points, G.qubits[a], G.qubits[b])
    end
    return points
end

# What a selection shows: its qubits, the stabilizers drawn with it, the stabilizer-to-qubit
# lines between those, its lattice edges and its faces.
function selection_parts(G, kind, k)
    qubits = Int[]
    stabilizers = Tuple{Symbol,Int}[]
    faces = Int[]

    if kind == :X
        qubits = G.x_support[k]
        stabilizers = [(:X, k)]
        faces = G.cell_faces[k]
    elseif kind == :Z
        qubits = G.z_support[k]
        stabilizers = [(:Z, k)]
        faces = [k]
    elseif kind == :qubit
        qubits = [k]
        for s in 1:length(G.x_support)
            if k in G.x_support[s]
                push!(stabilizers, (:X, s))
            end
        end
        for f in 1:length(G.z_support)
            if k in G.z_support[f]
                push!(stabilizers, (:Z, f))
            end
        end
    elseif kind == :edge
        a, b = G.edges[k]
        qubits = [a, b]
        for f in 1:length(G.z_support)
            if a in G.z_support[f] && b in G.z_support[f]
                push!(faces, f)
            end
        end
    end

    lines = Point3f[]
    for (stab_kind, s) in stabilizers
        support = stab_kind == :X ? G.x_support[s] : G.z_support[s]
        for j in qubits
            if j in support
                push!(lines, stab_position(G, stab_kind, s), G.qubits[j])
            end
        end
    end

    # the lattice edges between the selected qubits (for a qubit: the edges that meet at it)
    edges = Int[]
    for e in 1:length(G.edges)
        a, b = G.edges[e]
        if kind == :qubit
            belongs = (a == k || b == k)
        else
            belongs = (a in qubits && b in qubits)
        end
        if belongs
            push!(edges, e)
        end
    end

    return qubits, stabilizers, lines, edges, faces
end

# the text the panel shows for a selection
function selection_text(G, kind, k)
    if kind == :X
        return "X stabilizer $k, a $(colour_name(G.colour[k])) cell\n" *
               "position (centroid of its qubits) $(fmt(G.x_stabs[k]))\n" *
               "centre of the whole cell $(G.cells[k])\n" *
               "$(length(G.x_support[k])) qubits: $(number_list(G.x_support[k]))"
    elseif kind == :Z
        cells = G.face_cells[k]
        if length(cells) == 1
            place = "boundary face of X$(cells[1]) on the plane $(plane_label(G.d, G.face_plane[k]))"
        else
            difference = G.cells[cells[1]] .- G.cells[cells[2]]
            shape = maximum(abs.(difference)) == 4 ? "square" : "hexagon"
            place = "bulk $shape between X$(cells[1]) and X$(cells[2])"
        end
        return "Z stabilizer $k, $place\n" *
               "position (centroid of its qubits) $(fmt(G.z_stabs[k]))\n" *
               "$(length(G.z_support[k])) qubits: $(number_list(G.z_support[k]))"
    elseif kind == :qubit
        in_x = [s for s in 1:length(G.x_support) if k in G.x_support[s]]
        in_z = [f for f in 1:length(G.z_support) if k in G.z_support[f]]
        text = "qubit $k at $(fmt(G.qubits[k]))\n" *
               "in X stabilizers: $(number_list(in_x))\n" *
               "in Z stabilizers: $(number_list(in_z))"
        if k in G.logical_x
            text *= "\non logical X"
        end
        if k in G.logical_z
            text *= "\non logical Z"
        end
        return text
    elseif kind == :edge
        a, b = G.edges[k]
        in_z = [f for f in 1:length(G.z_support) if a in G.z_support[f] && b in G.z_support[f]]
        return "lattice edge $k between qubits $a and $b\nside of Z stabilizers: $(number_list(in_z))"
    end
    return "click a qubit, lattice edge, face, cell or stabilizer,\nor pick one with inspect X # / inspect Z #"
end

# ------------------------------------------------------------------------- the control panel

function make_toggles(position, d)
    toggle_list = [
        ("qubits", true), ("qubit numbers", d == 3),
        ("lattice edges", true), ("tetrahedron outline", true),
        ("bulk faces", false), ("boundary faces", true),
        ("cells (3D)", false), ("boundary planes", false),
        ("X stabilizers", true), ("Z stabilizers", true),
        ("X connections", false), ("Z connections", false),
        ("stabilizer numbers", false), ("x, y, z axes", false),
        ("logical X", false), ("logical Z", false),
        ("only selected connections", false), ("isolate the selection", false),
    ]
    grid = GridLayout(position)
    toggles = Dict{String,Toggle}()
    for (i, (name, active)) in enumerate(toggle_list)
        row = div(i + 1, 2)
        column = isodd(i) ? 1 : 3
        toggles[name] = Toggle(grid[row, column], active = active)
        Label(grid[row, column+1], name, halign = :left, fontsize = 13)
    end
    rowgap!(grid, 2)
    colgap!(grid, 6)
    return toggles
end

# the cell colours, the other colours, and the colour each face of the tetrahedron misses
function make_legend(position, G)
    legend = GridLayout(position)
    for c in 1:4
        Box(legend[1, 2c-1], color = CELL_COLOURS[c], width = 14, height = 14)
        Label(legend[1, 2c], CELL_COLOUR_NAMES[c], halign = :left, fontsize = 12)
    end
    others = [(Z_COLOUR, "Z stabilizer"), (SELECTION_COLOUR, "selection"), (LOGICAL_X_COLOUR, "logical X"),
        (LOGICAL_Z_COLOUR, "logical Z")]
    for (c, (colour, name)) in enumerate(others)
        Box(legend[2, 2c-1], color = colour, width = 14, height = 14)
        Label(legend[2, 2c], name, halign = :left, fontsize = 12)
    end

    missing = ["$(plane_label(G.d, i))  misses $(colour_name(G.missing_colour[i]))" for i in 1:4]
    Label(legend[3, 1:8], "$(missing[1]),   $(missing[2])\n$(missing[3]),   $(missing[4])",
        halign = :left, justification = :left, fontsize = 12)
    rowgap!(legend, 4)
end

# the control panel on the right of the window; returns its widgets
function make_panel(fig, G, selected)
    n_X = length(G.x_support)
    n_Z = length(G.z_support)
    panel = fig[1, 2] = GridLayout(tellheight = false, valign = :top)

    Label(panel[1, 1:2], "Tetrahedral 3D color code,  d = $(G.d)", fontsize = 17, font = :bold, halign = :left)
    Label(panel[2, 1:2], "$(G.code.n_qubits) qubits,  $n_X X stabilizers (cells),  $n_Z Z stabilizers (faces)",
        halign = :left, fontsize = 13)

    Label(panel[3, 1], "distance d (odd, then enter)", halign = :left)
    d_box = Textbox(panel[3, 2], stored_string = string(G.d), validator = is_valid_distance, width = 60, halign = :left)

    toggles = make_toggles(panel[4, 1:2], G.d)

    Label(panel[5, 1], "cut / slab along", halign = :left)
    cut_menu = Menu(panel[5, 2], options = CUT_FORMS, default = "x+y+z")

    low, high = form_extrema(G, "x+y+z")
    sliders = SliderGrid(panel[6, 1:2],
        (label = "face opacity", range = 0:0.05:1, startvalue = 0.3),
        (label = "cell opacity", range = 0:0.05:1, startvalue = 0.15),
        (label = "cut low", range = low:high, startvalue = low),
        (label = "cut high", range = low:high, startvalue = high),
        (label = "inspect X #", range = 0:n_X, startvalue = 0),
        (label = "inspect Z #", range = 0:n_Z, startvalue = 0))
    rowgap!(sliders.layout, 2)

    buttons = GridLayout(panel[7, 1:2])
    clear_button = Button(buttons[1, 1], label = "clear selection")
    reset_button = Button(buttons[1, 2], label = "reset view")
    save_button = Button(buttons[1, 3], label = "save picture")

    make_legend(panel[8, 1:2], G)

    info_text = lift(sel -> selection_text(G, sel[1], sel[2]), selected)
    Label(panel[9, 1:2], info_text, halign = :left, justification = :left, fontsize = 13, tellwidth = false)

    Label(panel[10, 1:2], "qubits black, lattice edges gray; cells, X stabilizers and X connections\n" *
                          "in their cell's colour.  Left-drag rotate, right-drag pan, scroll zoom, click to select.",
        halign = :left, justification = :left, fontsize = 11, color = :gray40)

    rowgap!(panel, 6)
    colsize!(panel, 2, Auto(false))         # the second column takes the rest of the panel's width
    return (; d_box, toggles, cut_menu, sliders = sliders.sliders, clear_button, reset_button, save_button)
end

# look at the tetrahedron with x to the right, y into the screen and z up
function set_view!(scene3d, d)
    L = 2d - 2
    centre = Vec3f(L / 2, L / 2, L / 2)
    eye = centre + Vec3f(0.45L, -2.3L, 0.9L)
    update_cam!(scene3d.scene, eye, centre, Vec3f(0, 0, 1))
end

# ------------------------------------------------------------------------- drawing
# V holds what the drawing functions share (see build_viewer). Every plot that can be clicked
# gets an entry in V.on_click: given the index a click reports, it returns what to select.

in_slab(V, slab, p) = slab[2] <= cut_value(slab[1], p, V.G.d) <= slab[3]

# a layer is drawn when its toggle is on, unless "isolate the selection" hides everything else
layer_visible(on, isolate, sel) = on && !(isolate && sel[1] != :none)
shown(V, name) = lift(layer_visible, V.toggles[name].active, V.toggles["isolate the selection"].active, V.selected)
is_shown(V, name) = layer_visible(V.toggles[name].active[], V.toggles["isolate the selection"].active[], V.selected[])

function draw_qubits_and_edges!(V)
    G = V.G

    # what lies outside the slab gets size 0 (or NaN lines), so the numbering of everything is kept
    sizes = lift(s -> [in_slab(V, s, p) ? V.qubit_size : 0.0 for p in G.qubits], V.slab)
    p = meshscatter!(V.scene3d, G.qubits, markersize = sizes, color = QUBIT_COLOUR, visible = shown(V, "qubits"))
    V.on_click[p] = j -> (:qubit, j)

    numbers = lift(s -> [in_slab(V, s, G.qubits[j]) ? string(j) : "" for j in 1:length(G.qubits)], V.slab)
    text!(V.scene3d, G.qubits, text = numbers, fontsize = 13, color = :black, align = (:left, :bottom),
        offset = (4, 4), visible = shown(V, "qubit numbers"))

    edge_points = lift(V.slab) do s
        points = Point3f[]
        for (a, b) in G.edges
            if in_slab(V, s, G.qubits[a]) && in_slab(V, s, G.qubits[b])
                push!(points, G.qubits[a], G.qubits[b])
            else
                push!(points, Point3f(NaN), Point3f(NaN))
            end
        end
        points
    end
    p = linesegments!(V.scene3d, edge_points, color = EDGE_COLOUR, linewidth = 2, visible = shown(V, "lattice edges"))
    V.on_click[p] = i -> (:edge, cld(i, 2))           # two points per edge
end

# One see-through mesh of `faces`; face i is drawn when centres[i] lies in the slab, and a click
# on it selects (kind, owners[i]).
function draw_faces!(V, faces, owners, centres, kind, colour, alpha, toggle_name)
    vertex_owner = Ref(Int[])
    face_mesh = lift(V.slab) do s
        keep = [i for i in 1:length(faces) if in_slab(V, s, centres[i])]
        m, owner = faces_mesh(V.G, faces[keep], owners[keep])
        vertex_owner[] = owner
        m
    end
    p = mesh!(V.scene3d, face_mesh, color = lift(a -> RGBAf(colour, a), alpha), transparency = true,
        visible = shown(V, toggle_name))
    V.on_click[p] = i -> vertex_owner[][i] > 0 ? (kind, vertex_owner[][i]) : nothing
    push!(V.see_through, p)
end

function draw_faces_and_cells!(V)
    G = V.G
    n_Z = length(G.z_support)

    bulk = [f for f in 1:n_Z if length(G.face_cells[f]) == 2]
    draw_faces!(V, bulk, bulk, G.z_stabs[bulk], :Z, BULK_FACE_COLOUR, V.face_alpha, "bulk faces")

    for colour in 0:3
        # the boundary faces of the cells of this colour
        boundary = [f for f in 1:n_Z if length(G.face_cells[f]) == 1 && G.colour[G.face_cells[f][1]] == colour]
        draw_faces!(V, boundary, boundary, G.z_stabs[boundary], :Z, CELL_COLOURS[colour+1], V.face_alpha,
            "boundary faces")

        # the cells of this colour, each drawn as all of its faces
        faces = Int[]
        owners = Int[]
        for s in 1:length(G.x_support)
            if G.colour[s] == colour
                append!(faces, G.cell_faces[s])
                append!(owners, fill(s, length(G.cell_faces[s])))
            end
        end
        draw_faces!(V, faces, owners, G.x_stabs[owners], :X, CELL_COLOURS[colour+1], V.cell_alpha, "cells (3D)")
    end
end

# the lines (stabilizer k, qubit j) with both ends in the slab; the others become NaN
function connection_points(V, slab, lines, positions)
    points = Point3f[]
    for (k, j) in lines
        if in_slab(V, slab, positions[k]) && in_slab(V, slab, V.G.qubits[j])
            push!(points, positions[k], V.G.qubits[j])
        else
            push!(points, Point3f(NaN), Point3f(NaN))
        end
    end
    return points
end

function draw_stabilizers!(V)
    G = V.G
    n_X = length(G.x_support)
    n_Z = length(G.z_support)
    x_colours = [CELL_COLOURS[c+1] for c in G.colour]

    sizes = lift(s -> [in_slab(V, s, p) ? V.x_size : 0.0 for p in G.x_stabs], V.slab)
    p = meshscatter!(V.scene3d, G.x_stabs, markersize = sizes, color = x_colours, visible = shown(V, "X stabilizers"))
    V.on_click[p] = s -> (:X, s)

    sizes = lift(s -> [in_slab(V, s, p) ? V.z_size : 0.0 for p in G.z_stabs], V.slab)
    p = meshscatter!(V.scene3d, G.z_stabs, marker = Rect3f(Vec3f(-0.5), Vec3f(1)), markersize = sizes,
        color = Z_COLOUR, visible = shown(V, "Z stabilizers"))
    V.on_click[p] = f -> (:Z, f)

    # one line for every nonzero entry of H_X and of H_Z: (stabilizer, qubit)
    x_lines = [(s, j) for s in 1:n_X for j in G.x_support[s]]
    z_lines = [(f, j) for f in 1:n_Z for j in G.z_support[f]]
    only_selected = V.toggles["only selected connections"].active

    x_line_colours = RGBAf[]
    for (s, _) in x_lines
        push!(x_line_colours, RGBAf(x_colours[s], 0.7), RGBAf(x_colours[s], 0.7))
    end
    p = linesegments!(V.scene3d, lift(s -> connection_points(V, s, x_lines, G.x_stabs), V.slab),
        color = x_line_colours, linewidth = 1.5,
        visible = lift((on, only) -> on && !only, shown(V, "X connections"), only_selected))
    V.on_click[p] = i -> (:X, x_lines[cld(i, 2)][1])

    p = linesegments!(V.scene3d, lift(s -> connection_points(V, s, z_lines, G.z_stabs), V.slab),
        color = RGBAf(Z_COLOUR, 0.6), linewidth = 1.5,
        visible = lift((on, only) -> on && !only, shown(V, "Z connections"), only_selected))
    V.on_click[p] = i -> (:Z, z_lines[cld(i, 2)][1])

    numbers = lift(s -> [in_slab(V, s, G.x_stabs[k]) ? "X$k" : "" for k in 1:n_X], V.slab)
    text!(V.scene3d, G.x_stabs, text = numbers, fontsize = 13, font = :bold, color = x_colours,
        align = (:left, :top), offset = (6, -6), visible = shown(V, "stabilizer numbers"))
    numbers = lift(s -> [in_slab(V, s, G.z_stabs[f]) ? "Z$f" : "" for f in 1:n_Z], V.slab)
    text!(V.scene3d, G.z_stabs, text = numbers, fontsize = 11, color = Z_COLOUR,
        align = (:left, :top), offset = (5, -5), visible = shown(V, "stabilizer numbers"))
end

# the tetrahedron, its four planes, the logical operators and the coordinate axes
function draw_outline_logicals_axes!(V)
    G = V.G
    outline = Point3f[]
    for i in 1:4, j in (i+1):4
        push!(outline, G.corners[i], G.corners[j])
    end
    linesegments!(V.scene3d, outline, color = :black, linewidth = 3, visible = V.toggles["tetrahedron outline"].active)

    for i in 1:4
        triangle = G.corners[setdiff(1:4, [i])]          # plane i is the face opposite corner i
        c = G.missing_colour[i]
        plane_colour = c < 0 ? RGBAf(0.5, 0.5, 0.5, 0.15) : RGBAf(CELL_COLOURS[c+1], 0.15)
        mesh!(V.scene3d, GB.Mesh(triangle, [GB.GLTriangleFace(1, 2, 3)]), color = plane_colour,
            transparency = true, visible = V.toggles["boundary planes"].active)
    end

    meshscatter!(V.scene3d, G.qubits[G.logical_x], markersize = 1.8 * V.qubit_size, color = LOGICAL_X_COLOUR,
        visible = shown(V, "logical X"))
    string_qubits = G.qubits[sort(G.logical_z, by = j -> G.qubits[j][1])]      # in order along the edge
    lines!(V.scene3d, string_qubits, color = LOGICAL_Z_COLOUR, linewidth = 6, visible = shown(V, "logical Z"))
    meshscatter!(V.scene3d, string_qubits, markersize = 1.8 * V.qubit_size, color = LOGICAL_Z_COLOUR,
        visible = shown(V, "logical Z"))

    L = 2G.d - 2
    axis_colours = [RGBf(0.80, 0.10, 0.10), RGBf(0.05, 0.55, 0.15), RGBf(0.10, 0.20, 0.85)]
    axes_on = V.toggles["x, y, z axes"].active
    for k in 1:3
        e = Point3f(k == 1, k == 2, k == 3)
        lines!(V.scene3d, [Point3f(0, 0, 0), (L + 2) * e], color = axis_colours[k], linewidth = 3, visible = axes_on)
        text!(V.scene3d, [(L + 2.5f0) * e], text = ["xyz"[k:k]], fontsize = 22, font = :bold,
            color = axis_colours[k], align = (:center, :center), visible = axes_on)
        ticks = collect(2:2:L)
        text!(V.scene3d, [Float32(t) * e for t in ticks], text = string.(ticks), fontsize = 11,
            color = axis_colours[k], align = (:right, :top), offset = (-3, -3), visible = axes_on)
    end
end

# the selection in orange, on top of everything else
function draw_selection!(V)
    G = V.G
    parts = V.parts
    chosen_x = lift(q -> [s for (kind, s) in q[2] if kind == :X], parts)
    chosen_z = lift(q -> [f for (kind, f) in q[2] if kind == :Z], parts)

    p = meshscatter!(V.scene3d, lift(q -> G.qubits[q[1]], parts), markersize = 1.7 * V.qubit_size,
        color = SELECTION_COLOUR)
    V.on_click[p] = i -> (:qubit, parts[][1][i])
    p = meshscatter!(V.scene3d, lift(c -> G.x_stabs[c], chosen_x), markersize = 1.2 * V.x_size,
        color = SELECTION_COLOUR)
    V.on_click[p] = i -> (:X, chosen_x[][i])
    p = meshscatter!(V.scene3d, lift(c -> G.z_stabs[c], chosen_z), marker = Rect3f(Vec3f(-0.5), Vec3f(1)),
        markersize = 1.2 * V.z_size, color = SELECTION_COLOUR)
    V.on_click[p] = i -> (:Z, chosen_z[][i])

    linesegments!(V.scene3d, lift(q -> q[3], parts), color = SELECTION_COLOUR, linewidth = 3)
    p = linesegments!(V.scene3d, lift(q -> edge_segments(G, q[4]), parts), color = SELECTION_COLOUR, linewidth = 5)
    V.on_click[p] = i -> (:edge, parts[][4][cld(i, 2)])

    vertex_owner = Ref(Int[])
    face_mesh = lift(parts) do q
        m, owner = faces_mesh(G, q[5], q[5])
        vertex_owner[] = owner
        m
    end
    p = mesh!(V.scene3d, face_mesh, color = RGBAf(SELECTION_COLOUR, 0.35), transparency = true)
    V.on_click[p] = i -> vertex_owner[][i] > 0 ? (:Z, vertex_owner[][i]) : nothing
    push!(V.see_through, p)

    # numbers of the selected qubits and stabilizers, also when the rest is hidden
    text!(V.scene3d, lift(q -> [(string(j), G.qubits[j]) for j in q[1]], parts), fontsize = 13, color = :black,
        align = (:left, :bottom), offset = (4, 4), visible = V.toggles["qubit numbers"].active)
    text!(V.scene3d, lift(q -> [(string(kind, s), stab_position(G, kind, s)) for (kind, s) in q[2]], parts),
        fontsize = 13, font = :bold, color = :black, align = (:left, :top), offset = (6, -6),
        visible = V.toggles["stabilizer numbers"].active)
end

# ------------------------------------------------------------------------- clicking

# where a point of the 3D scene appears in the window, in pixels
function screen_position(V, p)
    q = GLMakie.Makie.project(V.scene3d.scene, :data, :pixel, p)
    origin = GLMakie.Makie.viewport(V.scene3d.scene)[].origin
    return Point2f(q[1] + origin[1], q[2] + origin[2])
end

# distance from the point m to the segment from a to b (points on the screen)
function distance_to_segment(m, a, b)
    t = clamp(dot(m - a, b - a) / dot(b - a, b - a), 0, 1)
    return norm(m - (a + t * (b - a)))
end

# A click only "sees" the front-most object. Faces and cells are see-through, so a click on
# one of them (or on empty space) looks for a qubit or stabilizer within 6 pixels of the
# mouse, and then for a lattice edge within 4 pixels.
function object_seen_through(V, mouse)
    G = V.G
    slab = V.slab[]
    parts = V.parts[]

    markers = Tuple{Symbol,Int}[]
    if is_shown(V, "qubits")
        append!(markers, [(:qubit, j) for j in 1:length(G.qubits) if in_slab(V, slab, G.qubits[j])])
    end
    if is_shown(V, "X stabilizers")
        append!(markers, [(:X, s) for s in 1:length(G.x_stabs) if in_slab(V, slab, G.x_stabs[s])])
    end
    if is_shown(V, "Z stabilizers")
        append!(markers, [(:Z, f) for f in 1:length(G.z_stabs) if in_slab(V, slab, G.z_stabs[f])])
    end
    append!(markers, [(:qubit, j) for j in parts[1]])
    append!(markers, parts[2])

    nearest = nothing
    nearest_distance = 6.0
    for (kind, k) in markers
        p = kind == :qubit ? G.qubits[k] : stab_position(G, kind, k)
        distance = norm(screen_position(V, p) - mouse)
        if distance < nearest_distance
            nearest = (kind, k)
            nearest_distance = distance
        end
    end
    if nearest !== nothing
        return nearest
    end

    edges = copy(parts[4])
    if is_shown(V, "lattice edges")
        for e in 1:length(G.edges)
            a, b = G.edges[e]
            if in_slab(V, slab, G.qubits[a]) && in_slab(V, slab, G.qubits[b])
                push!(edges, e)
            end
        end
    end
    nearest_distance = 4.0
    for e in edges
        a, b = G.edges[e]
        distance = distance_to_segment(mouse, screen_position(V, G.qubits[a]), screen_position(V, G.qubits[b]))
        if distance < nearest_distance
            nearest = (:edge, e)
            nearest_distance = distance
        end
    end
    return nearest
end

# a click (press and release without dragging) selects what is under the mouse
function connect_mouse!(fig, V)
    press_position = Ref(Point2f(0, 0))
    on(events(fig).mousebutton, priority = 2) do event
        mouse = Point2f(events(fig).mouseposition[])
        if event.button == Mouse.left && event.action == Mouse.press
            press_position[] = mouse
        elseif event.button == Mouse.left && event.action == Mouse.release
            if norm(mouse - press_position[]) < 4 && GLMakie.Makie.is_mouseinside(V.scene3d.scene)
                plot, index = pick(V.scene3d.scene)
                choice = haskey(V.on_click, plot) ? V.on_click[plot](index) : nothing
                if choice === nothing || plot in V.see_through
                    behind = object_seen_through(V, mouse)
                    if behind !== nothing
                        choice = behind
                    end
                end
                if choice !== nothing
                    V.selected[] = choice
                end
            end
        end
        return Consume(false)
    end
end

# ------------------------------------------------------------------------- the viewer

# The 3D scene without the panel, exactly as it is seen, as a PNG at 3 times the screen resolution
# in visualizations/tetrahedral_3D_cc/figure_exports, and its camera in camera_d<d>.txt next to it
# (the figures file, tetrahedral_3D_cc_figures.jl, loads that camera with its key c).
function save_view(fig, scene3d, d)
    folder = joinpath(@__DIR__, "figure_exports")
    mkpath(folder)
    file = joinpath(folder, "tetrahedral_d$(d)_viewer_$(Libc.strftime("%Y%m%d-%H%M%S", time())).png")

    # the whole window at 3x (update = false keeps the camera), then the part of the 3D scene;
    # the rows of the image count from the top, the scene's origin from the bottom
    image = colorbuffer(fig, px_per_unit = 3, update = false)
    area = GLMakie.Makie.viewport(scene3d.scene)[]
    x, y = area.origin
    w, h = area.widths
    top = size(fig.scene)[2] - (y + h)
    save(file, image[3top+1:3(top+h), 3x+1:3(x+w)])

    cam = cameracontrols(scene3d.scene)
    open(joinpath(folder, "camera_d$(d).txt"), "w") do io
        println(io, "eyeposition ", join(cam.eyeposition[], " "))
        println(io, "lookat ", join(cam.lookat[], " "))
        println(io, "upvector ", join(cam.upvector[], " "))
        println(io, "fov ", cam.fov[])
    end
    println("saved $file")
end

# selected: (:X, s), (:Z, f), (:qubit, j), (:edge, e) or (:none, 0); passing your own
# Observable lets you change the selection from the REPL
function build_viewer(d; selected = Observable((:none, 0)))
    G = code_geometry(d)

    fig = Figure(size = (1450, 860))
    # center = false: the camera keeps the view of set_view! (Makie would re-centre it when shown);
    # backward_key = false: the s key saves a picture instead of moving the camera back
    scene3d = LScene(fig[1, 1], show_axis = false,
        scenekw = (camera = scene -> cam3d!(scene, center = false, backward_key = false),))
    ui = make_panel(fig, G, selected)
    colsize!(fig.layout, 2, Fixed(440))
    low_slider, high_slider, inspect_X, inspect_Z = ui.sliders[3], ui.sliders[4], ui.sliders[5], ui.sliders[6]

    # the slab: lowest and highest value of the chosen form (at the ends of its range, no limit)
    slab = lift(ui.cut_menu.selection, low_slider.value, high_slider.value) do form, low, high
        vmin, vmax = form_extrema(G, form)
        (form, low <= vmin ? -Inf : low, high >= vmax ? Inf : high)
    end

    V = (; G, scene3d, selected, slab, toggles = ui.toggles,
        parts = lift(sel -> selection_parts(G, sel[1], sel[2]), selected),
        face_alpha = ui.sliders[1].value, cell_alpha = ui.sliders[2].value,
        qubit_size = d <= 5 ? 0.12 : 0.09, x_size = d <= 5 ? 0.30 : 0.22, z_size = d <= 5 ? 0.20 : 0.15,
        on_click = Dict{Any,Function}(), see_through = [])

    draw_qubits_and_edges!(V)
    draw_faces_and_cells!(V)
    draw_stabilizers!(V)
    draw_outline_logicals_axes!(V)
    draw_selection!(V)
    connect_mouse!(fig, V)

    # a new form for the slab: the two cut sliders run over its values
    on(ui.cut_menu.selection) do form
        vmin, vmax = form_extrema(G, form)
        low_slider.range[] = vmin:vmax
        high_slider.range[] = vmin:vmax
        set_close_to!(low_slider, vmin)
        set_close_to!(high_slider, vmax)
    end

    on(inspect_X.value) do s
        if s > 0
            selected[] = (:X, s)
        end
    end
    on(inspect_Z.value) do f
        if f > 0
            selected[] = (:Z, f)
        end
    end
    on(ui.clear_button.clicks) do _
        set_close_to!(inspect_X, 0)
        set_close_to!(inspect_Z, 0)
        selected[] = (:none, 0)
    end
    on(ui.reset_button.clicks) do _
        set_view!(scene3d, d)
    end
    on(ui.save_button.clicks) do _
        save_view(fig, scene3d, d)
    end
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press && event.key == Keyboard.s
            save_view(fig, scene3d, d)
        end
        return Consume(false)
    end

    # a new distance rebuilds the viewer in the same window
    on(ui.d_box.stored_string) do text
        new_d = parse(Int, text)
        if new_d != d
            display(build_viewer(new_d))
        end
    end

    set_view!(scene3d, d)
    return fig
end

show_viewer(d = 3) = display(build_viewer(d))

if abspath(PROGRAM_FILE) == @__FILE__
    d = isempty(ARGS) ? 3 : parse(Int, ARGS[1])
    screen = display(build_viewer(d))
    wait(screen)
end
