# plot_sweep_thresholds_tetrahedral_3D_cc.jl -- the two thresholds of the tetrahedral 3D color code, from
# the files that test/tetrahedral_3D_cc/run_tests_sweep_thresholds_tetrahedral_3D_cc.jl writes in this
# folder: sweep_thresholds_phaseflip_tetrahedral_3D_cc.csv and sweep_thresholds_bitflip_tetrahedral_3D_cc.csv
#
#   julia results/plot_sweep_thresholds_tetrahedral_3D_cc.jl     (saves results/thresholds_tetrahedral_3D_cc.png)
#
# Two panels, the logical error rate against p for every distance:
#   (a) phase flips (Z errors only), with the optimal threshold 1.9% of the string-like logical Z
#   (b) bit flips (X errors only), with the optimal threshold 27.6% of the membrane-like logical X
# (both optimal thresholds from Kubica et al., PRL 120, 180501 (2018)). Error bars: Wilson intervals
# (1 sigma). A point without failures is drawn as an open triangle pointing down at 3/N (N samples),
# the upper end of its 95% interval. The terminal shows where the curves of d and d + 2 cross.

using GLMakie

const OPTIMAL_PHASEFLIP = 0.019
const OPTIMAL_BITFLIP = 0.276

# the rows of a results file as (d, p) => (samples, incorrect); a missing file has no rows. A point
# run more than once counts with its row with the most samples (a smaller run only repeats the
# first samples of a larger one, as the seeds are fixed).
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

# for each d, the points (p, rate, low, high, samples), sorted by p
function curves(results)
    points = Dict{Int,Vector{NTuple{5,Float64}}}()
    for ((d, p), (n, k)) in results
        low, high = wilson(k, n)
        push!(get!(points, d, NTuple{5,Float64}[]), (p, k / n, low, high, n))
    end
    for d in keys(points)
        sort!(points[d])
    end
    return points
end

# where the curve `large` crosses the curve `small` from below: every pair of neighbouring p (of
# both curves) where it goes from under to over, with log(rate) linear in p in between. Noisy data
# can cross more than once; all crossings are returned, so that the noise shows (none: empty).
function crossing(small, large)
    rate_small = Dict(q[1] => q[2] for q in small)
    rate_large = Dict(q[1] => q[2] for q in large)
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

# the markers and colours of the comparison with Butt et al. (their Fig. 3a)
MARKERS = Dict(3 => :circle, 5 => :utriangle, 7 => :cross, 9 => :xcross)
COLOURS = Dict(3 => colorant"#1f77b4", 5 => colorant"#2ca02c", 7 => colorant"#d62728", 9 => colorant"#ff7f0e")

phaseflip = curves(read_results(joinpath(@__DIR__, "sweep_thresholds_phaseflip_tetrahedral_3D_cc.csv")))
bitflip = curves(read_results(joinpath(@__DIR__, "sweep_thresholds_bitflip_tetrahedral_3D_cc.csv")))

# the panels: title, data, x label, optimal threshold, limits, ticks of p
panels = [
    ("(a) phase flips (Z errors)", phaseflip, L"Physical error rate $p_Z$", OPTIMAL_PHASEFLIP,
        (0.0042, 0.034, 1e-5, 1), ([0.005, 0.01, 0.015, 0.02, 0.03], ["0.5%", "1%", "1.5%", "2%", "3%"])),
    ("(b) bit flips (X errors)", bitflip, L"Physical error rate $p_X$", OPTIMAL_BITFLIP,
        (0.045, 0.37, 1e-5, 1), ([0.05, 0.1, 0.2, 0.3], ["5%", "10%", "20%", "30%"]))]

fig = with_theme(theme_latexfonts()) do
    fig = Figure(size=(1150, 500), fontsize=18)
    for (i, (title, points, xlabel, optimal, limits, pticks)) in enumerate(panels)
        ax = Axis(fig[1, i], title=title, titlealign=:left, xscale=log10, yscale=log10,
            xlabel=xlabel, ylabel=L"Logical error rate $p_L$", limits=limits, xticks=pticks,
            yticks=(10.0 .^ (-5:0), [L"10^{%$(k)}" for k = -5:0]),
            xminorticksvisible=true, xminorticks=IntervalsBetween(5), yminorticksvisible=true,
            yminorticks=IntervalsBetween(9), xgridcolor=(:gray, 0.55), ygridcolor=(:gray, 0.55),
            spinewidth=1.5)

        vlines!(ax, optimal, color=:gray40, linestyle=:dash, linewidth=2)
        entries = Any[]
        labels = Any[]
        any_zero = false
        for d in sort(collect(keys(points)))
            # points with failures: markers with error bars, joined by a thin line
            drawn = [q for q in points[d] if q[2] > 0]
            if !isempty(drawn)
                ps = [q[1] for q in drawn]
                rates = [q[2] for q in drawn]
                errorbars!(ax, ps, rates, rates .- [q[3] for q in drawn], [q[4] for q in drawn] .- rates,
                    color=COLOURS[d], whiskerwidth=5)
                lines!(ax, ps, rates, color=(COLOURS[d], 0.6), linewidth=1)
                scatter!(ax, ps, rates, marker=MARKERS[d], color=COLOURS[d], markersize=4)
            end
            # points without failures: an open triangle at 3/N
            zero = [q for q in points[d] if q[2] == 0]
            if !isempty(zero)
                scatter!(ax, [q[1] for q in zero], [3 / q[5] for q in zero], marker=:dtriangle, color=:white,
                    strokecolor=COLOURS[d], strokewidth=1.5, markersize=11)
                any_zero = true
            end
            push!(entries, MarkerElement(marker=MARKERS[d], color=COLOURS[d], markersize=14))
            push!(labels, L"$d = %$(d)$")
        end
        push!(entries, LineElement(color=:gray40, linestyle=:dash, linewidth=2))
        push!(labels, "optimal $(round(100optimal, digits = 1))% (Kubica et al.)")
        if any_zero
            push!(entries, MarkerElement(marker=:dtriangle, color=:white, strokecolor=:black, strokewidth=1.5, markersize=11))
            push!(labels, L"no failures (at $3/N$)")
        end
        axislegend(ax, entries, labels, position=:rb, labelsize=15, rowgap=0,
            framecolor=(:gray, 0.6), backgroundcolor=(:white, 0.85))
    end
    fig
end

file = joinpath(@__DIR__, "thresholds_tetrahedral_3D_cc.png")
save(file, fig, px_per_unit=2)
println("saved $file")

for (title, points, _, optimal, _, _) in panels
    distances = sort(collect(keys(points)))
    pairs = ["$d/$(d + 2): $(percent(crossing(points[d], points[d+2])))" for d in distances if haskey(points, d + 2)]
    println(rpad(title, 28), "crossing of d and d + 2 (optimal $(round(100optimal, digits = 1))%):  ", join(pairs, "    "))
end
