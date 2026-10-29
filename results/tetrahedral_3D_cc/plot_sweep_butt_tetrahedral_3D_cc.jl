# plot_sweep_butt_tetrahedral_3D_cc.jl -- our decoder against Butt et al. (PRA 113, 042416, 2026) under
# their noise, from the files that test/tetrahedral_3D_cc/run_tests_sweep_butt_tetrahedral_3D_cc.jl
# writes in this folder: sweep_butt_phaseflip_tetrahedral_3D_cc.csv and sweep_butt_bitflip_tetrahedral_3D_cc.csv
#
#   julia results/tetrahedral_3D_cc/plot_sweep_butt_tetrahedral_3D_cc.jl     (saves butt_comparison_tetrahedral_3D_cc.png here)
#
# One graph in the style of their Fig. 3a: the logical error rate, averaged over the bit-flip and the
# phase-flip run as they report it, against p.
#   ours     markers with error bars (Wilson intervals, 1 sigma); a point without failures has no place
#            on a log scale and is left out. Below our lowest point a dashed line, the asymptote
#            p_L = A p^((d+1)/2) (a logical phase flip needs (d+1)/2 flips on the string of weight d; bit
#            flips are far rarer), with A fitted to our lowest points.
#   theirs   dotted lines through their Fig. 3a, digitized in butt_etal_3d_tetrahedral_digitized_20_points.csv
# The inset zooms in on the crossing, with the line p_L = p. The terminal shows where the curves of d
# and d + 2 cross and where the d = 3 curve crosses p_L = p (the pseudo-threshold), ours next to theirs.

using GLMakie

# Butt et al., Table II (tetrahedral code)
const BUTT_CROSSING = 0.0148
const BUTT_PSEUDO_THRESHOLD = 0.0113

# the rows of a results file as (d, p) => (samples, incorrect); a missing file has no rows. A point
# run more than once counts with its row with the most samples: the seeds are fixed, so a smaller
# run only repeats the first samples of a larger one (and a quick test does not hide a long run).
function read_results(file)
    results = Dict{Tuple{Int,Float64},Tuple{Int,Int}}()
    if isfile(file)
        for line in readlines(file)[2:end]
            words = split(line, ',')
            point = (parse(Int, words[1]), parse(Float64, words[2]))
            samples, incorrect = parse(Int, words[3]), parse(Int, words[4])
            if !haskey(results, point) || samples >= results[point][1]
                results[point] = (samples, incorrect)
            end
        end
    end
    return results
end

# the Wilson interval (1 sigma) of k failures in n samples
function wilson(k, n)
    centre = (k + 0.5) / (n + 1)
    half = sqrt(k * (n - k) / n + 0.25) / (n + 1)
    return centre - half, centre + half
end

# for each d, the points (p, rate, low, high), sorted by p
function curves(results)
    points = Dict{Int,Vector{NTuple{4,Float64}}}()
    for ((d, p), (n, k)) in results
        low, high = wilson(k, n)
        push!(get!(points, d, NTuple{4,Float64}[]), (p, k / n, low, high))
    end
    for d in keys(points)
        sort!(points[d])
    end
    return points
end

# the average of two runs at every (d, p) that has both: the mean of the two rates, with the error
# bars of the two runs added in quadrature (and halved, as the rates are)
function average_curves(a, b)
    points = Dict{Int,Vector{NTuple{4,Float64}}}()
    for d in keys(a)
        if haskey(b, d)
            points[d] = NTuple{4,Float64}[]
            for (p, r1, low1, high1) in a[d], (q, r2, low2, high2) in b[d]
                if p == q
                    rate = (r1 + r2) / 2
                    push!(points[d], (p, rate, rate - hypot(r1 - low1, r2 - low2) / 2, rate + hypot(high1 - r1, high2 - r2) / 2))
                end
            end
        end
    end
    return points
end

# where the curve `large` crosses the curve `small` from below: every pair of neighbouring p (of
# both curves) where it goes from under to over, with log(rate) linear in p in between. Noisy data
# can cross more than once; all crossings are returned, so that the noise shows (none: empty).
function crossing(small, large)
    rate_small = Dict(p => r for (p, r, _, _) in small)
    rate_large = Dict(p => r for (p, r, _, _) in large)
    ps = sort([p for p in keys(rate_small) if haskey(rate_large, p)])
    found = Float64[]
    for k = 1:(length(ps)-1)
        a1, b1 = rate_small[ps[k]], rate_large[ps[k]]
        a2, b2 = rate_small[ps[k+1]], rate_large[ps[k+1]]
        if 0 < b1 < a1 && 0 < a2 <= b2
            f1 = log(b1) - log(a1)
            f2 = log(b2) - log(a2)
            push!(found, ps[k] + (ps[k+1] - ps[k]) * f1 / (f1 - f2))
        end
    end
    return found
end

percent(xs) = isempty(xs) ? "none on the grid" : join([string(round(100x, digits=2), "%") for x in xs], " and ")

# A of the asymptote p_L = A p^k, from the lowest three points of a curve below p = 1% that have
# failures: A = (sum of p_L) / (sum of p^k), the best fit when the points have the same number of
# samples (it is not thrown off by a point with only a few failures); nothing if there are none
function asymptote(points, k)
    low = [q for q in points if q[2] > 0 && q[1] <= 0.01][1:min(3, end)]
    if isempty(low)
        return nothing
    end
    return sum(q[2] for q in low) / sum(q[1]^k for q in low)
end

phaseflip = curves(read_results(joinpath(@__DIR__, "sweep_butt_phaseflip_tetrahedral_3D_cc.csv")))
bitflip = curves(read_results(joinpath(@__DIR__, "sweep_butt_bitflip_tetrahedral_3D_cc.csv")))
ours = average_curves(phaseflip, bitflip)

# their curves: columns p_phys, p_L_d3, p_L_d5, p_L_d7, p_L_d9
butt_lines = readlines(joinpath(@__DIR__, "butt_etal_3d_tetrahedral_digitized_20_points.csv"))
butt_header = split(butt_lines[1], ',')
butt_rows = [parse.(Float64, split(line, ',')) for line in butt_lines[2:end]]
butt_p = [row[1] for row in butt_rows]
butt_distances = [d for d in (3, 5, 7, 9) if "p_L_d$d" in butt_header]
butt_rate(d) = [row[findfirst(==("p_L_d$d"), butt_header)] for row in butt_rows]

# the markers and colours of their figure
BUTT_MARKERS = Dict(3 => :circle, 5 => :utriangle, 7 => :cross, 9 => :xcross)
BUTT_COLOURS = Dict(3 => colorant"#1f77b4", 5 => colorant"#2ca02c", 7 => colorant"#d62728", 9 => colorant"#ff7f0e")
ZOOM = (2.2e-3, 5e-2, 1e-4, 0.6)          # the region of the inset: p from, p to, p_L from, p_L to

# our points with error bars, and theirs as dotted lines, on one axis
function draw_comparison!(ax, ours, markersize)
    for d in butt_distances
        lines!(ax, butt_p, butt_rate(d), color=BUTT_COLOURS[d], linestyle=:dot, linewidth=1.6)
    end
    for d in sort(collect(keys(ours)))
        drawn = [q for q in ours[d] if q[2] > 0]
        if isempty(drawn)
            continue
        end
        ps = [q[1] for q in drawn]
        rates = [q[2] for q in drawn]
        errorbars!(ax, ps, rates, rates .- [q[3] for q in drawn], [q[4] for q in drawn] .- rates,
            color=BUTT_COLOURS[d], whiskerwidth=5)
        scatter!(ax, ps, rates, marker=BUTT_MARKERS[d], color=BUTT_COLOURS[d], markersize=markersize)
    end
end

fig = with_theme(theme_latexfonts()) do
    fig = Figure(size=(780, 560), fontsize=18)
    ax = Axis(fig[1, 1], xscale=log10, yscale=log10, limits=(1e-4, 1e-1, 1e-12, 1),
        xlabel=L"Physical error rate $p_\mathrm{phys}$", ylabel=L"Logical error rate $p_L$",
        xticks=(10.0 .^ (-4:-1), [L"10^{%$(k)}" for k = -4:-1]),
        yticks=(10.0 .^ (-12:2:0), [L"10^{%$(k)}" for k = -12:2:0]),
        xminorticksvisible=true, xminorticks=IntervalsBetween(9),
        yminorticksvisible=true, yminorticks=10.0 .^ (-11:2:-1),
        xgridcolor=(:gray, 0.55), ygridcolor=(:gray, 0.55), xgridwidth=1.2, ygridwidth=1.2,
        spinewidth=1.5, xtickwidth=1.3, ytickwidth=1.3)

    # the asymptotes below our lowest point: a light line with a dark dashed one on top, as theirs
    for d in sort(collect(keys(ours)))
        k = div(d + 1, 2)
        A = asymptote(ours[d], k)
        drawn = [q for q in ours[d] if q[2] > 0]
        if A !== nothing
            ps = 10 .^ range(-4, log10(drawn[1][1]), length=40)
            lines!(ax, ps, A .* ps .^ k, color=(BUTT_COLOURS[d], 0.45), linewidth=4)
            lines!(ax, ps, A .* ps .^ k, color=BUTT_COLOURS[d], linestyle=:dash, linewidth=2)
        end
    end
    draw_comparison!(ax, ours, 4)

    # the distances, as in their legend (all four, so that their dotted lines have a colour too), and
    # below it which lines are ours and which theirs
    legend_distances = [d for d in (3, 7, 5, 9) if d in butt_distances || haskey(ours, d)]   # rows 3, 7 and 5, 9
    axislegend(ax, [MarkerElement(marker=BUTT_MARKERS[d], color=BUTT_COLOURS[d], markersize=16) for d in legend_distances],
        [L"$d = %$(d)$" for d in legend_distances], position=:lt, nbanks=2, labelsize=21,
        framecolor=(:gray, 0.6), backgroundcolor=(:white, 0.85), patchsize=(22, 22), colgap=22)
    axislegend(ax, [MarkerElement(marker=:circle, color=:black), LineElement(color=:black, linestyle=:dash),
        LineElement(color=:black, linestyle=:dot), LineElement(color=:gray45, linewidth=2)],
        ["this work", "this work, low-p fit", "Butt et al. (digitized)", L"$p_L = p$ (inset)"],
        position=(0.0, 0.83), labelsize=14, rowgap=0, padding=(6, 6, 3, 3),
        framecolor=(:gray, 0.6), backgroundcolor=(:white, 0.85))

    # the region of the inset
    x0, x1, y0, y1 = ZOOM
    lines!(ax, [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=:black, linestyle=:dashdot, linewidth=1.6)

    # the inset, in front of the main axis, with the line p_L = p
    inset = Axis(fig[1, 1], width=Relative(0.36), height=Relative(0.4), halign=0.95, valign=0.1,
        tellwidth=false, tellheight=false, xscale=log10, yscale=log10, limits=ZOOM,
        xticks=([1e-2], [L"10^{-2}"]), yticks=([1e-3, 1e-1], [L"10^{-3}", L"10^{-1}"]),
        xgridvisible=false, ygridvisible=false, backgroundcolor=:white, spinewidth=2)
    translate!(inset.blockscene, 0, 0, 150)
    lines!(inset, [x0, x1], [x0, x1], color=:gray45, linewidth=2)
    draw_comparison!(inset, ours, 10)

    # the lines from the region to the inset, in pixels of the figure
    connectors = lift(ax.scene.viewport, inset.scene.viewport) do main, small
        pixel(x, y) = Point2f(main.origin[1] + main.widths[1] * (log10(x) + 4) / 3,
            main.origin[2] + main.widths[2] * (log10(y) + 12) / 12)
        top = small.origin[2] + small.widths[2]
        [pixel(x0, y0), Point2f(small.origin[1], top), pixel(x1, y0), Point2f(small.origin[1] + small.widths[1], top)]
    end
    overlay = Scene(fig.scene, camera=campixel!, clear=false)
    linesegments!(overlay, connectors, color=:black, linestyle=:dash, linewidth=1.4, overdraw=true)
    fig
end

file = joinpath(@__DIR__, "butt_comparison_tetrahedral_3D_cc.png")
save(file, fig, px_per_unit=2)
println("saved $file")

println("asymptote of ours (dashed): ", join(["d = $d: p_L = $(round(asymptote(ours[d], div(d + 1, 2)), sigdigits = 3)) p^$(div(d + 1, 2))"
                                              for d in sort(collect(keys(ours))) if asymptote(ours[d], div(d + 1, 2)) !== nothing], ",  "))

# the crossings and the pseudo-threshold, ours and theirs (their curves are read off a picture, so
# quote their Table II numbers in the paper)
butt_curves = Dict(d => [(p, r, r, r) for (p, r) in zip(butt_p, butt_rate(d))] for d in butt_distances)
println("crossing of the curves of d and d + 2, average of the two runs (Butt et al., Table II: $(percent([BUTT_CROSSING])))")
for (name, points) in (("ours", ours), ("Butt et al. (digitized)", butt_curves))
    distances = sort(collect(keys(points)))
    println("  ", rpad(name, 26), join(["$d/$(d + 2): $(percent(crossing(points[d], points[d+2])))" for d in distances if haskey(points, d + 2)], "    "))
end
println("pseudo-threshold, where p_L = p for d = 3 (Butt et al., Table II: $(percent([BUTT_PSEUDO_THRESHOLD])))")
for (name, points) in (("ours", ours), ("Butt et al. (digitized)", butt_curves))
    if haskey(points, 3)
        println("  ", rpad(name, 26), percent(crossing([(p, p, p, p) for (p, _, _, _) in points[3]], points[3])))
    end
end

println("Butt et al. against ours (average of the two runs), at their p where we have failures:")
for d in sort(collect(keys(ours)))
    if d in butt_distances
        for (p, theirs) in zip(butt_p, butt_rate(d))
            for (q, mine, _, _) in ours[d]
                if q == round(p, sigdigits=4) && mine > 0
                    println("  d = $d, p = $(round(100p, digits = 2))%:  Butt et al. $(round(theirs, sigdigits = 3)),  ours ",
                        "$(round(mine, sigdigits = 3)),  theirs / ours = $(round(theirs / mine, digits = 1))")
                end
            end
        end
    end
end
