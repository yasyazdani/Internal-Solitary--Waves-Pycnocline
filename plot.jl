# plot u field and b in the pycnoclined reagion, parameters are based on E1

using Oceananigans
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnodes, znodes
using SpecialFunctions
using CairoMakie

# ----------------------------
# INPUT
# ----------------------------
jld_in = "Mar-13-f0-Mode1_timeseries-00.jld2"
println("Reading: $jld_in")

outdir = "Mar-13-f0-Mode1_bu_snapshots-00"
mkpath(outdir)

# ----------------------------
# PARAMETERS
# ----------------------------
N2 = 0.36
f = 0
ω = sqrt((N2 + f^2) / 2)
hp = 0.02
δp = 0.01
g  = 9.81
Δp = 0.0205
N0 = 0.6
λx = 0.536

@inline function N2_of_z(z; hp, δp, g, Δp, N0)
    a = 2 * (z + hp) / δp
    N2_pyc = (g * Δp) * (2 / (sqrt(π) * δp)) * exp(-a^2)
    N2_low = (z < -hp) ? N0^2 : 0.0
    return N2_pyc + N2_low
end

bbar(z) = z >= -hp + δp/2 ? 0.0 : (g * Δp / 2) * erf((z + hp) / (δp / 2)) + (z < -hp ? N0^2 * (z + hp) : 0.0)

# ----------------------------
# LOAD b AND u
# ----------------------------
tsb = FieldTimeSeries(jld_in, "b")
tsu = FieldTimeSeries(jld_in, "u")

grid_b = tsb.grid
grid_u = tsu.grid

# b is at centers/centers
xb = xnodes(grid_b, Center())
zb = znodes(grid_b, Center())

# u is at faces/centers
xu = xnodes(grid_u, Face())
zu = znodes(grid_u, Center())

# normalized coordinates
xb_plot = xb ./ λx
zb_plot = zb ./ hp

xu_plot = xu ./ λx
zu_plot = zu ./ hp

# ----------------------------
# WINDOWS
# ----------------------------
kz_b = findall(zz -> (-2.0 <= zz <= 0.0), zb_plot)
kx_b = findall(xx -> (0 <= xx <= 7.5), xb_plot)

kz_u = findall(zz -> (-46.0 <= zz <= 0.0), zu_plot)
kx_u = findall(xx -> (0.0 <= xx <= 7.5), xu_plot)

xv_b = xb_plot[kx_b]
zv_b = zb_plot[kz_b]

xv_u = xu_plot[kx_u]
zv_u = zu_plot[kz_u]

# isopycnal levels
z_levels = range(-hp - 0.25δp, -hp + 0.05δp; length=10)
levels = bbar.(z_levels)

# initial u field
u0 = Array(interior(tsu[1])[:, 1, :])[kx_u, kz_u]
umax0 = max(maximum(abs, u0), 1e-12)

# ----------------------------
# FIGURE WITH TWO SUBPLOTS
# ----------------------------
fig = Figure(size=(700, 600))

ax1 = Axis(fig[1, 1],
           xlabel=L"x / \lambda_x",
           ylabel=L"z / h_p",
           xlabelsize=42,
           ylabelsize=42,
           titlesize=32,
           xticklabelsize=32,
           yticklabelsize=32)

ax2 = Axis(fig[2, 1],
           xlabel=L"x / \lambda_x",
           ylabel=L"z / h_p",
           xlabelsize=42,
           ylabelsize=42,
           titlesize=32,
           xticklabelsize=32,
           yticklabelsize=32)

Box(fig[1, 2], color = (:white, 0.0), strokecolor = :transparent)

ax1.limits = (0, 7.5, -2, 0)
ax2.limits = (0, 7.5, -46, 0)

# initial top subplot
B0 = Array(interior(tsb[1])[:, 1, :])[kx_b, kz_b]
contour!(ax1, xv_b, zv_b, B0; levels=levels, color=:black, linewidth=1)

# initial bottom subplot
hm = heatmap!(ax2, xv_u, zv_u, u0;
              colormap=:balance,
              colorrange=(-umax0, umax0))

Colorbar(fig[2, 2], hm,
         label="u (m/s)",
         ticklabelsize=22,
         labelsize=22)

# ----------------------------
# SAVE SNAPSHOTS
# ----------------------------
for it in 1:length(tsb.times)
    empty!(ax1)

    B = Array(interior(tsb[it])[:, 1, :])[kx_b, kz_b]
    contour!(ax1, xv_b, zv_b, B; levels=levels, color=:black, linewidth=1)
    ax1.limits = (0, 7.5, -2, 0)
    ax1.title = "t = $(round(tsb.times[it], digits=2)) s"

    u = Array(interior(tsu[it])[:, 1, :])[kx_u, kz_u]
    umax = max(maximum(abs, u), 1e-12)
    hm[3] = u
    hm.colorrange = (-umax, umax)
    ax2.limits = (0, 7.5, -46, 0)

    outfile = joinpath(outdir, "bu_" * lpad(string(it), 4, '0') * ".png")
    save(outfile, fig)
end

println("Saved snapshots in: $outdir")
