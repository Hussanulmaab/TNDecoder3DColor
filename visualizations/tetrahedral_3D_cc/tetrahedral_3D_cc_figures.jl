# tetrahedral_3D_cc_figures.jl -- the 3D scene for Fig. 1: the code, its logicals, and errors with their syndrome
#
#   julia visualizations/tetrahedral_3D_cc/tetrahedral_3D_cc_figures.jl                      distance 3, no errors
#   julia visualizations/tetrahedral_3D_cc/tetrahedral_3D_cc_figures.jl 5 mode=loop          distance 5, the loop example
#   julia visualizations/tetrahedral_3D_cc/tetrahedral_3D_cc_figures.jl 5 p=0.02 seed=1 mode=random     (mode = point, loop, random or none)
#   in the REPL:  include("visualizations/tetrahedral_3D_cc/tetrahedral_3D_cc_figures.jl"); show_figures(5)
#                 then set_errors!(Z = [1, 2], X = [10], Y = [20]), example!(:point), example!(:loop),
#                 random_errors!(0.02, 1), clear_errors!()
#   without a window:  check_examples(5)
#
# The code and its geometry come from the viewer, visualization_tetrahedral_3D_cc.jl in this folder.
# The syndrome is always computed from the code: violated cells = (H_X z) mod 2, violated faces =
# (H_Z x) mod 2, where z and x are the 0/1 error vectors (a Y error sets both).
#
# Mouse   left-drag rotate, right-drag pan, scroll zoom; a click on a qubit toggles an error on it
# Keys    z / x   clicks put Z / X errors          1   point-like example (phase flips)
#         2       loop-like example (bit flips)    r   random depolarizing errors, the next seed on every press
#         0       clear the errors                 l   logical X and Z      k   cells      g   ring of violated faces
#         h       status line                      s   save a PNG and the camera          c   load the saved camera
#         v       the viewer's default view

include(joinpath(@__DIR__, "visualization_tetrahedral_3D_cc.jl"))   # GLMakie, the code and code_geometry
using Dates
using Random

# Paper colours. Blue is kept for phase flips, so the cells take muted colours.
const PAPER_PHASE = parse(RGBf, "#2a78d6")     # Z errors, violated cells, logical Z
const PAPER_BIT = parse(RGBf, "#eb6834")       # X errors, violated faces, logical X
const PAPER_CELL_COLOURS = parse.(RGBf, ["#e87ba4", "#199e70", "#c98500", "#6250d6"])   # cell colours 0 to 3
const PAPER_QUBIT = parse(RGBf, "#1c1c1c")
const PAPER_EDGE = RGBf(0.6, 0.6, 0.6)         # mid grey
const PAPER_DARK = RGBf(0.15, 0.15, 0.15)      # face outlines, the ring and the status line

const EXPORT_FOLDER = joinpath(@__DIR__, "figure_exports")
camera_file(d) = joinpath(EXPORT_FOLDER, "camera_d$(d).txt")

# ------------------------------------------------------------------------- the examples

# Point-like syndrome: Z errors on the first k qubits of logical Z, counted from the corner (0,0,0)
# along the edge. Only the cell at the open end of the string is violated.
function point_example(G)
    n = G.code.n_qubits
    string_qubits = sort(G.logical_z, by = j -> G.qubits[j][1])
    k = div(G.d + 1, 2)
    z = zeros(Int, n)
    z[string_qubits[1:k]] .= 1
    x = zeros(Int, n)
    violated = findall((G.code.H_X * z) .% 2 .== 1)
    @assert length(violated) == 1
    return z, x, k, violated
end

# Loop-like syndrome: X errors on the qubits of the largest boundary face of the plane
# x + y + z = 4d - 4 (the logical X membrane) whose centre is nearest the centre of that plane.
# The violated faces form a ring around it, none of them in the plane.
function loop_example(G)
    n = G.code.n_qubits
    bound = 4G.d - 4
    on_plane(j) = sum(G.code.pos_qubits[j]) == bound
    plane_faces = [f for f in 1:length(G.z_support) if all(on_plane, G.z_support[f])]
    largest = maximum(length(G.z_support[f]) for f in plane_faces)
    candidates = [f for f in plane_faces if length(G.z_support[f]) == largest]
    plane_centre = fill(bound / 3, 3)
    face = candidates[argmin([norm(collect(G.code.pos_stab_Z[f]) - plane_centre) for f in candidates])]
    x = zeros(Int, n)
    x[G.z_support[face]] .= 1
    z = zeros(Int, n)
    violated = findall((G.code.H_Z * x) .% 2 .== 1)
    @assert !any(all(on_plane, G.z_support[f]) for f in violated)
    return z, x, face, violated
end

face_kind(G, f) = length(G.face_cells[f]) == 2 ? "bulk" : "boundary"

# the examples and the logicals, checked without a window
function check_examples(d)
    G = code_geometry(d)
    code = G.code
    println("d = $d: n = $(code.n_qubits), $(length(G.x_support)) cells (X checks), $(length(G.z_support)) faces (Z checks)")

    z, x, k, cells = point_example(G)
    string_qubits = sort(G.logical_z, by = j -> G.qubits[j][1])
    println("  point: Z on the first k = $k qubits of logical Z $(string_qubits[1:k]) -> ",
        "$(length(cells)) violated cell: $(cells)")

    z, x, face, faces = loop_example(G)
    println("  loop:  X on the $(length(G.z_support[face])) qubits $(G.z_support[face]) of boundary face Z$face -> ",
        "$(length(faces)) violated faces: ",
        join(["Z$f (weight $(length(G.z_support[f])), $(face_kind(G, f)))" for f in faces], ", "))

    violated_by_x = count(isodd, code.H_Z * code.L_X)
    violated_by_z = count(isodd, code.H_X * code.L_Z)
    println("  logicals: X on logical X -> $violated_by_x violated faces, Z on logical Z -> $violated_by_z violated cells")
    @assert violated_by_x == 0 && violated_by_z == 0
end

# ------------------------------------------------------------------------- drawing
# F holds the window's state (see build_figures). The errors are one Observable (z, x); the
# syndrome and every drawing follow from it.

# The sides of a face, as qubit pairs (smaller index first). Where the tetrahedron cuts a cell, some
# sides are not lattice edges: they are 2√2 or 3√2 long and lie in a boundary plane, so the edges
# of the code are the lattice edges together with the sides of the faces.
function face_sides(G, f)
    corners = G.polygons[f]
    m = length(corners)
    return [minmax(corners[k], corners[mod1(k + 1, m)]) for k in 1:m]
end

# the lattice edges whose two qubits are among the given ones, and the sides of the given faces, each once
function edges_and_sides(G, qubits, faces)
    pairs = [minmax(a, b) for (a, b) in G.edges if a in qubits && b in qubits]
    for f in faces
        append!(pairs, face_sides(G, f))
    end
    return unique(pairs)
end

pair_segments(G, pairs) = [G.qubits[j] for pair in pairs for j in pair]

# the edges around each of the faces
function face_outlines(G, faces)
    points = Point3f[]
    for f in faces
        corners = G.qubits[G.polygons[f]]
        m = length(corners)
        for k in 1:m
            push!(points, corners[k], corners[mod1(k + 1, m)])
        end
    end
    return points
end

# the centres of the faces in order around their mean, in the plane that fits them best, closed
# into a ring (nothing for fewer than 3 faces)
function ring_points(G, faces)
    if length(faces) < 3
        return Point3f[]
    end
    centres = [collect(Float64, G.z_stabs[f]) for f in faces]
    mean_centre = sum(centres) / length(centres)
    fit = svd(reduce(hcat, [c - mean_centre for c in centres])')
    u, w = fit.V[:, 1], fit.V[:, 2]                    # the two directions in the best-fit plane
    angles = [atan(dot(c - mean_centre, w), dot(c - mean_centre, u)) for c in centres]
    ring = [Point3f(c...) for c in centres[sortperm(angles)]]
    return push!(ring, ring[1])
end

# qubits, the edges (lattice edges and the cut sides of the faces), the tetrahedron and its four
# planes, each in the colour it misses (the plane of logical X turns light orange when the logicals are shown)
function draw_code!(F)
    G = F.G
    p = meshscatter!(F.scene3d, G.qubits, markersize = F.qubit_size, color = PAPER_QUBIT)
    F.on_click[p] = j -> j

    # Lines lying on a boundary plane or on each other would flicker in stripes. Small depth shifts
    # draw them in a fixed order: planes, edges, the outline, then the blue lines (the violated
    # cells' edges and logical Z).
    edges = edges_and_sides(G, 1:length(G.qubits), 1:length(G.z_support))
    linesegments!(F.scene3d, pair_segments(G, edges), color = PAPER_EDGE, linewidth = 1.5, depth_shift = -1f-5)

    outline = Point3f[]
    for i in 1:4, j in (i+1):4
        push!(outline, G.corners[i], G.corners[j])
    end
    linesegments!(F.scene3d, outline, color = :black, linewidth = 2.5, depth_shift = -2f-5)

    for i in 1:4
        triangle = G.corners[setdiff(1:4, [i])]          # plane i is the face opposite corner i
        c = G.missing_colour[i]
        missing_shade = c < 0 ? RGBAf(0.5, 0.5, 0.5, 0.15) : RGBAf(PAPER_CELL_COLOURS[c+1], 0.15)
        plane_colour = missing_shade
        if i == 1                                        # x + y + z = 4d - 4
            plane_colour = lift(on -> on ? RGBAf(PAPER_BIT, 0.25) : missing_shade, F.show_logicals)
        end
        mesh!(F.scene3d, GB.Mesh(triangle, [GB.GLTriangleFace(1, 2, 3)]), color = plane_colour, transparency = true)
    end
end

# the cells as X-stabilizer spheres; while errors are shown, the cells that are not violated are faint
function draw_cells!(F)
    G = F.G
    colours = lift(F.errors, F.violated_cells) do (z, x), violated
        has_errors = any(==(1), z) || any(==(1), x)
        [RGBAf(PAPER_CELL_COLOURS[G.colour[s]+1], has_errors && !(s in violated) ? 0.2 : 1.0) for s in 1:length(G.x_stabs)]
    end
    meshscatter!(F.scene3d, G.x_stabs, markersize = F.cell_size, color = colours, transparency = true,
        visible = F.show_cells)
end

# logical X: its qubits on the membrane; logical Z: its qubits and a line along the edge
function draw_logicals!(F)
    G = F.G
    p = meshscatter!(F.scene3d, G.qubits[G.logical_x], markersize = 1.6 * F.qubit_size, color = PAPER_BIT,
        visible = F.show_logicals)
    F.on_click[p] = i -> G.logical_x[i]

    string_qubits = sort(G.logical_z, by = j -> G.qubits[j][1])     # in order along the edge
    lines!(F.scene3d, G.qubits[string_qubits], color = PAPER_PHASE, linewidth = 5, depth_shift = -3f-5,
        visible = F.show_logicals)
    p = meshscatter!(F.scene3d, G.qubits[string_qubits], markersize = 1.6 * F.qubit_size, color = PAPER_PHASE,
        visible = F.show_logicals)
    F.on_click[p] = i -> string_qubits[i]
end

# Z errors as blue cubes, X errors as orange spheres (a Y error gets both); violated cells as large
# blue spheres with their edges, violated faces as orange polygons with a dark outline
function draw_errors_and_syndrome!(F)
    G = F.G
    z_qubits = lift(e -> findall(==(1), e[1]), F.errors)
    x_qubits = lift(e -> findall(==(1), e[2]), F.errors)

    p = meshscatter!(F.scene3d, lift(q -> G.qubits[q], z_qubits), marker = Rect3f(Vec3f(-0.5), Vec3f(1)),
        markersize = 3.2 * F.qubit_size, color = PAPER_PHASE)
    F.on_click[p] = i -> z_qubits[][i]
    p = meshscatter!(F.scene3d, lift(q -> G.qubits[q], x_qubits), markersize = 1.9 * F.qubit_size, color = PAPER_BIT)
    F.on_click[p] = i -> x_qubits[][i]

    # the edges of each cell: its lattice edges and the sides of its faces (the cut sides included)
    cell_edges = [edges_and_sides(G, G.x_support[s], G.cell_faces[s]) for s in 1:length(G.x_support)]
    meshscatter!(F.scene3d, lift(c -> G.x_stabs[c], F.violated_cells), markersize = 1.5 * F.cell_size, color = PAPER_PHASE)
    linesegments!(F.scene3d, lift(c -> pair_segments(G, reduce(vcat, cell_edges[c], init = Tuple{Int,Int}[])), F.violated_cells),
        color = PAPER_PHASE, linewidth = 3.5, depth_shift = -3f-5)

    # Every violated face shows the same orange: no shading, and no see-through blending with the
    # boundary planes. The small depth shifts keep a face lying on a boundary plane, and its outline,
    # in front of that plane.
    mesh!(F.scene3d, lift(f -> faces_mesh(G, f, f)[1], F.violated_faces), color = RGBAf(PAPER_BIT, 0.85),
        transparency = false, shading = NoShading, depth_shift = -1f-4)
    linesegments!(F.scene3d, lift(f -> face_outlines(G, f), F.violated_faces), color = PAPER_DARK, linewidth = 1.5,
        depth_shift = -2f-4)

    # the ring passes through the faces, so it is drawn on top of everything
    lines!(F.scene3d, lift(f -> ring_points(G, f), F.violated_faces), color = :black, linewidth = 2.5,
        overdraw = true, visible = F.show_ring)
end

# one line in the bottom-left corner, drawn over the 3D scene
function draw_status!(F)
    G = F.G
    status = lift(F.errors, F.violated_cells, F.violated_faces, F.click_type, F.mode, F.p, F.seed) do (z, x), cells, faces, click, mode, p, seed
        n_Z = count(j -> z[j] == 1 && x[j] == 0, 1:length(z))
        n_X = count(j -> x[j] == 1 && z[j] == 0, 1:length(z))
        n_Y = count(j -> z[j] == 1 && x[j] == 1, 1:length(z))
        line = "d = $(G.d)    click: $click    Z $n_Z   X $n_X   Y $n_Y    violated cells $(length(cells))   violated faces $(length(faces))"
        mode == "random" ? line * "    p = $p   seed $seed" : line
    end
    overlay = Scene(F.fig.scene, camera = campixel!, clear = false)
    text!(overlay, Point2f(12, 10), text = status, align = (:left, :bottom), fontsize = 15, color = PAPER_DARK,
        overdraw = true, visible = F.show_status)
end

# ------------------------------------------------------------------------- errors

function put_errors!(F, z, x, mode)
    F.mode[] = mode
    F.errors[] = (z, x)
end

# exact lists of qubits (1-based, as in the code); a Y error is both a Z and an X error
function set_errors!(F::NamedTuple; Z = Int[], X = Int[], Y = Int[])
    n = F.G.code.n_qubits
    z = zeros(Int, n)
    x = zeros(Int, n)
    z[Z] .= 1
    x[X] .= 1
    z[Y] .= 1
    x[Y] .= 1
    put_errors!(F, z, x, "manual")
end

function example!(F::NamedTuple, kind::Symbol)
    if kind == :point
        z, x, k, cells = point_example(F.G)
        println("point example: Z on the first $k qubits of logical Z -> $(length(cells)) violated cell, X$(cells[1])")
    elseif kind == :loop
        z, x, face, faces = loop_example(F.G)
        println("loop example: X on the $(length(F.G.z_support[face])) qubits of boundary face Z$face -> ",
            "$(length(faces)) violated faces, none in its plane")
    else
        error("the examples are :point and :loop")
    end
    put_errors!(F, z, x, string(kind))
end

# Depolarizing noise: each qubit gets X, Y or Z with probability p/3 each, drawn as in the decoder
# tests (r < p/3 is X, r < 2p/3 is Y, r < p is Z).
function random_errors!(F::NamedTuple, p, seed)
    r = rand(Xoshiro(seed), F.G.code.n_qubits)
    x = Int.(r .< 2p / 3)
    z = Int.((r .>= p / 3) .& (r .< p))
    F.p[] = p
    F.seed[] = seed
    put_errors!(F, z, x, "random")
end

clear_errors!(F::NamedTuple) = put_errors!(F, zeros(Int, F.G.code.n_qubits), zeros(Int, F.G.code.n_qubits), "none")

function toggle_error!(F, j)
    z, x = copy(F.errors[][1]), copy(F.errors[][2])
    if F.click_type[] == :Z
        z[j] = 1 - z[j]
    else
        x[j] = 1 - x[j]
    end
    put_errors!(F, z, x, "manual")
end

# ------------------------------------------------------------------------- saving and the camera

function write_camera(F)
    cam = cameracontrols(F.scene3d.scene)
    open(camera_file(F.G.d), "w") do io
        println(io, "eyeposition ", join(cam.eyeposition[], " "))
        println(io, "lookat ", join(cam.lookat[], " "))
        println(io, "upvector ", join(cam.upvector[], " "))
        println(io, "fov ", cam.fov[])
    end
end

function load_camera!(F)
    file = camera_file(F.G.d)
    if !isfile(file)
        println("no saved camera for d = $(F.G.d) yet: $file")
        return
    end
    values = Dict{String,Vector{Float64}}()
    for line in eachline(file)
        words = split(line)
        values[words[1]] = parse.(Float64, words[2:end])
    end
    cam = cameracontrols(F.scene3d.scene)
    cam.fov[] = values["fov"][1]
    update_cam!(F.scene3d.scene, Vec3d(values["eyeposition"]...), Vec3d(values["lookat"]...), Vec3d(values["upvector"]...))
end

function reset_view!(F)
    cameracontrols(F.scene3d.scene).fov[] = F.default_fov[]
    set_view!(F.scene3d, F.G.d)
end

# the window holds only the 3D scene, so saving the figure without the status line saves exactly the view
function save_picture!(F)
    mkpath(EXPORT_FOLDER)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    file = joinpath(EXPORT_FOLDER, "tetrahedral_d$(F.G.d)_$(F.mode[])_$(stamp).png")
    status_shown = F.show_status[]
    F.show_status[] = false
    save(file, F.fig, px_per_unit = 3, update = false)       # update = false keeps the camera as it is
    F.show_status[] = status_shown
    write_camera(F)
    println("saved $file and $(camera_file(F.G.d))")
end

# ------------------------------------------------------------------------- mouse and keys

# The qubit under the mouse: of the qubits drawn within 3 pixels of it the one nearest the camera
# (the one in front), otherwise the qubit drawn nearest the mouse, within 8 pixels.
function nearest_qubit(F, mouse)
    eye = cameracontrols(F.scene3d.scene).eyeposition[]
    nearest = nothing
    best = (2, Inf)
    for j in 1:length(F.G.qubits)
        distance = norm(screen_position(F, F.G.qubits[j]) - mouse)
        rank = distance < 3 ? (0, norm(F.G.qubits[j] - eye)) : (1, distance)
        if distance < 8 && rank < best
            nearest = j
            best = rank
        end
    end
    return nearest
end

# A click (press and release without dragging) on a qubit toggles an error on it. A click on
# anything else (a face, a plane, a cell) takes the qubit drawn nearest the mouse.
function connect_mouse!(F)
    press_position = Ref(Point2f(0, 0))
    on(events(F.fig).mousebutton, priority = 2) do event
        mouse = Point2f(events(F.fig).mouseposition[])
        if event.button == Mouse.left && event.action == Mouse.press
            press_position[] = mouse
        elseif event.button == Mouse.left && event.action == Mouse.release
            if norm(mouse - press_position[]) < 4 && GLMakie.Makie.is_mouseinside(F.scene3d.scene)
                plot, index = pick(F.scene3d.scene)
                j = haskey(F.on_click, plot) ? F.on_click[plot](index) : nothing
                if j === nothing
                    j = nearest_qubit(F, mouse)
                end
                if j !== nothing
                    toggle_error!(F, j)
                end
            end
        end
        return Consume(false)
    end
end

function connect_keys!(F)
    on(events(F.fig).keyboardbutton) do event
        if event.action == Keyboard.press
            key = event.key
            if key == Keyboard.z
                F.click_type[] = :Z
            elseif key == Keyboard.x
                F.click_type[] = :X
            elseif key == Keyboard._1 || key == Keyboard.kp_1
                example!(F, :point)
            elseif key == Keyboard._2 || key == Keyboard.kp_2
                example!(F, :loop)
            elseif key == Keyboard.r
                # the next seed; the first press after other errors uses the current seed
                random_errors!(F, F.p[], F.mode[] == "random" ? F.seed[] + 1 : F.seed[])
            elseif key == Keyboard._0 || key == Keyboard.kp_0
                clear_errors!(F)
            elseif key == Keyboard.h
                F.show_status[] = !F.show_status[]
            elseif key == Keyboard.l
                F.show_logicals[] = !F.show_logicals[]
            elseif key == Keyboard.k
                F.show_cells[] = !F.show_cells[]
            elseif key == Keyboard.g
                F.show_ring[] = !F.show_ring[]
            elseif key == Keyboard.s
                save_picture!(F)
            elseif key == Keyboard.c
                load_camera!(F)
            elseif key == Keyboard.v
                reset_view!(F)
            end
        end
        return Consume(false)
    end
end

# ------------------------------------------------------------------------- the window

function build_figures(d; p = 0.02, seed = 1)
    G = code_geometry(d)
    n = G.code.n_qubits

    fig = Figure(size = (1000, 850), figure_padding = 0)
    # The camera keeps the view of set_view! (center = false) and ignores the keyboard, whose letters
    # (r, s, l, k, x, z, ...) are used here, and ctrl- and alt-clicks; the mouse controls are the viewer's.
    camera = scene -> cam3d!(scene, center = false, keyboard_rotationspeed = 0.0, keyboard_translationspeed = 0.0,
        keyboard_zoomspeed = 0.0, fix_x_key = false, fix_y_key = false, fix_z_key = false, reset = false,
        reposition_button = false)
    scene3d = LScene(fig[1, 1], show_axis = false, scenekw = (camera = camera,))

    errors = Observable((zeros(Int, n), zeros(Int, n)))
    F = (; G, fig, scene3d, errors,
        violated_cells = lift(e -> findall((G.code.H_X * e[1]) .% 2 .== 1), errors),
        violated_faces = lift(e -> findall((G.code.H_Z * e[2]) .% 2 .== 1), errors),
        click_type = Observable(:Z), mode = Observable("none"), p = Observable(Float64(p)), seed = Observable(seed),
        show_status = Observable(true), show_logicals = Observable(false), show_cells = Observable(true),
        show_ring = Observable(false),
        qubit_size = d <= 5 ? 0.12 : 0.09, cell_size = d <= 5 ? 0.30 : 0.22,
        on_click = Dict{Any,Function}(), default_fov = Ref(45.0))

    draw_code!(F)
    draw_cells!(F)
    draw_logicals!(F)
    draw_errors_and_syndrome!(F)
    draw_status!(F)
    connect_mouse!(F)
    connect_keys!(F)

    set_view!(scene3d, d)
    F.default_fov[] = cameracontrols(scene3d.scene).fov[]
    return F
end

# the window the REPL functions act on (kept when this file is included again)
if !@isdefined(FIGURES)
    const FIGURES = Ref{Any}(nothing)
end

function open_figures()
    if FIGURES[] === nothing
        error("no window yet: open one with show_figures(d)")
    end
    return FIGURES[]
end

function show_figures(d = 3; p = 0.02, seed = 1, mode = "none")
    mode = string(mode)
    if !is_valid_distance(string(d)) || !(mode in ("none", "point", "loop", "random"))
        error("d is an odd number from 3, and mode is none, point, loop or random")
    end
    F = build_figures(d; p, seed)
    FIGURES[] = F
    if mode == "point" || mode == "loop"
        example!(F, Symbol(mode))
    elseif mode == "random"
        random_errors!(F, p, seed)
    end
    return display(F.fig)
end

set_errors!(; Z = Int[], X = Int[], Y = Int[]) = set_errors!(open_figures(); Z, X, Y)
example!(kind::Symbol) = example!(open_figures(), kind)
random_errors!(p, seed) = random_errors!(open_figures(), p, seed)
clear_errors!() = clear_errors!(open_figures())

# d, p, seed and mode from the command line, e.g.  5 p=0.02 seed=1 mode=random
function command_line_settings(arguments)
    d, p, seed, mode = 3, 0.02, 1, "none"
    for argument in arguments
        if startswith(argument, "p=")
            p = parse(Float64, argument[3:end])
        elseif startswith(argument, "seed=")
            seed = parse(Int, argument[6:end])
        elseif startswith(argument, "mode=")
            mode = argument[6:end]
        else
            d = parse(Int, argument)
        end
    end
    return d, p, seed, mode
end

if abspath(PROGRAM_FILE) == @__FILE__
    d, p, seed, mode = command_line_settings(ARGS)
    screen = show_figures(d; p, seed, mode)
    wait(screen)
end
