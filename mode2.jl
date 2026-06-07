# March-15-mode2.jl —  on gpu
# sponge where s is defined as sigmiod , right wall only
# Oceananigans 0.97.7
# CUDA 5.8.3
#OffsetArrays 1.17.0
#JLD2 0.5.15
# 3 layers similar to NG 2011
# compared to the previous codes: I don't have A, Cf = Uf/Tf like the paper, Uf = U0 where 
# U0 = As_target * N0 * λx_res * sin(θi)   # => U0 by Eq 20
# removing sponge in b
# the code is working but no ISW after 180 s in Feb-004 so i am moaking the envelope larger but futher away from the middle layer
#changing the angle for Uf to theta not theta_i
#changing dx from 0.002 to 0.001, and its coresponding dz

using Oceananigans
using Oceananigans.OutputWriters: JLD2Writer, TimeInterval
using Oceananigans.Grids: node
using SpecialFunctions
using CUDA

#  domain & constants (E2 ) 
Lx, H  = 3.0, 0.8
dx     = 0.001          # 2 mm
Nx     =  Int(round(Lx / dx))
Nz     = 800            # vertical count (stretched grid below sets dz min/max targets)

#Parameters
N2 =  0.36
ω = sqrt(N2 / 2) # disperssion (f = 0 , k^2 = m^2)

hp = 0.02
δp = 0.01
g  = 9.81
Δp = 0.0338
N0 = 0.6
N02 = N0^2


@inline function N2_of_z(z; hp, δp, g, Δp, N0)
    a = 2 * (z + hp) / δp
    N2_pyc = (g * Δp) * (2 / (sqrt(π) * δp)) * exp(-a^2)   # pycnocline term (all z)
    N2_low = (z < -hp) ? N0^2 : 0.0                        # lower layer term
    return N2_pyc + N2_low
end


function Ni_from_profile(; ω, hp, δp, g, Δp, N0, nint=4000)
    z1 = -hp - δp/2
    z2 = -hp + δp/2
    zs = range(z1, z2; length=nint)
    vals = similar(collect(zs))
    for (idx, z) in enumerate(zs)
      #  vals[idx] = 1 / sqrt(N2_of_z(z; hp, δp, g, Δp, N0) - ω^2)
	vals[idx] = sqrt(N2_of_z(z; hp, δp, g, Δp, N0) - ω^2)
    end
    zsv = collect(zs)
    I = sum(0.5 .* (vals[1:end-1] .+ vals[2:end]) .* diff(zsv))  # trapezoid
    #Ni2 = ω^2 + (δp / I)^2
    Ni2 = ω^2 + (I / δp)^2
    return sqrt(Ni2)
end

Ni = Ni_from_profile(; ω, hp, δp, g, Δp, N0)
θi = asin(ω / Ni)

#  λx_res so μn = 1
n = 2  # mode-2 (E2 target)
λx_res = 2 * δp / ((n - 1) * tan(θi))
k = 2π / λx_res
m = -k                 # |m| = k ⇒ 45° beam

As_target = 0.013 # 0.01 too small # 0.02 too big # based on 2014 paper
Tf = 2π / ω          # forcing period (paper's Tf)

θ = asin(ω / N0)
Uf = As_target * N0 * λx_res * sin(θ) # => U0 by Eq 20, then they used Uf ≈ U0? 
#Cf = Uf / Tf 2-Feb-004.mp4
Cf =  Uf / Tf


Nλx = Lx / λx_res


# stretched vertical grid: coarse at bottom (dzM), fine near top (dzm),
#     transition centered at z = -3hp 
function stretched_z_faces(; H, Nz, dzm, dzM, zt, w = 0.1H)
    ζ = range(0, 1; length = Nz)                          # computational coordinate (centers count)
    S = 1 ./(1 .+ exp.(-(((ζ .- (zt + H)/H) ./ (w/H)))))   # sigmoid in ζ-space
    dz = dzM .- (dzM - dzm) .* S                          # large -> small upward
    z  = -H .+ vcat(0.0, cumsum(dz))                       # preliminary faces (Nz+1)
    z  = -H .+ (z .+ H) .* (H / (z[end] + H))              # rescale to end exactly at 0
    return z
end

# E2 target: dzm = 0.3 mm, dzM = 4 mm
z_faces = stretched_z_faces(; H, Nz, dzm = 0.0003, dzM = 0.004, zt = -3hp, w = 0.1H)


# similar to jan-28 diffrent from 29 which failed
grid = RectilinearGrid(GPU(CUDA.CUDABackend()); size=(Nx, 1, Nz), x=(0, Lx), y=(0, 1), z=z_faces)




# nodes for each field location
xC, xF = xnodes(grid, Center()), xnodes(grid, Face())   # u on x-Face
zC, zF = znodes(grid, Center()), znodes(grid, Face())   # w on z-Face


# 2014_Nonlinear_generation_of... paper-style wavemaker envelope, derivatives, and phase 
@inline Fenv(x, z, p)  = exp(-((x - p.xc)^2 / (2p.σx^2) + (z - p.zc)^2 / (2p.σz^2)))
@inline Fx(x, z, p)    = -((x - p.xc) / p.σx^2) * Fenv(x, z, p)
@inline Fz(x, z, p)    = -((z - p.zc) / p.σz^2) * Fenv(x, z, p)
@inline phase(x, z, t, p) = p.k * x + p.m * z - p.ω * t  # sign of m sets up/down propagation




# --- FORCINGS (GPU-safe: get coordinates from grid, not from p.* arrays) ---
@inline function fu(i, j, k_, grid, clock, fields, p)
    x, _, _ = node(i, j, k_, grid, Face(),   Center(), Center())
    _, _, z = node(i, j, k_, grid, Face(),   Center(), Center())
    ϕ = phase(x, z, clock.time, p)
    F  = Fenv(x, z, p)
    Fx_ = Fx(x, z, p)
    Fz_ = Fz(x, z, p)
    return p.Cf * ( -F * (p.m / p.k) * cos(ϕ) -
             (1 / p.k) * Fz_ * sin(ϕ) +
             (p.m / p.k^2) * Fx_ * sin(ϕ) )
end

@inline function fw(i, j, k_, grid, clock, fields, p)
    x, _, _ = node(i, j, k_, grid, Center(), Center(), Face())
    _, _, z = node(i, j, k_, grid, Center(), Center(), Face())
    ϕ = phase(x, z, clock.time, p)
    F = Fenv(x, z, p)
    return p.Cf * F * cos(ϕ)
end

@inline function fb(i, j, k_, grid, clock, fields, p)
    x, _, _ = node(i, j, k_, grid, Center(), Center(), Center())
    _, _, z = node(i, j, k_, grid, Center(), Center(), Center())
    ϕ = phase(x, z, clock.time, p)
    F = Fenv(x, z, p)
    return p.Cf * sqrt(p.N2) * F * sin(ϕ)
end


# Feb-004-gpu.jl the one minute video
# wavemaker geometry
#xc = 0.6               # forcing near left boundary
#zc = -0.35              # forcing near top

# envelope widths (narrow Gaussian in terms of λx = Lx / Nλx)
#σz = 0.1 *  H # 0.3 * (Lx / Nλx)  
#σx = 0.3 * (Lx / Nλx)  

# new ------------------------------------
# your λx (you already have this)
λx = λx_res

# envelope widths (match Diamessis et al. 2014)
σx = 0.538 * λx
σz = 0.538 * λx

# wavemaker geometry (keep xc if you want)
xc = 0.6

# match their vertical offset from pycnocline: zcen ≈ z_pyc - 2.05 λx
zc = -hp  - 2.05 * λx
# ------------------------------------------------------


# --- sponge layer on the right: Herbaut-style relaxation ---

ℓs = 0.1    # = λx  sponge layer length
x_sponge = Lx - ℓs                 # inner boundary of sponge (facing interior)

T0 = 2π / ω                        # forcing period
τ_inner  = T0                      # relaxation time at inner edge
τ_outer  = T0 / 1000               # relaxation time at outer boundary



μu_host = zeros(length(xF))
μw_host = zeros(length(xC))

x0 = x_sponge + ℓs/2
k_sig = 10

for i in eachindex(xF)
    x = xF[i]
    if x <= x_sponge
        μu_host[i] = 0.0
    else
        s = 1 / (1 + exp(-k_sig * (x - x0)))   # logistic sigmoid
        τ = τ_inner * (1 - s) + τ_outer * s
        μu_host[i] = 1 / τ
    end
end

for i in eachindex(xC)
    x = xC[i]
    if x <= x_sponge
        μw_host[i] = 0.0
    else
        s = 1 / (1 + exp(-k_sig * (x - x0)))   # same sigmoid
        τ = τ_inner * (1 - s) + τ_outer * s
        μw_host[i] = 1 / τ
    end
end

μu = CUDA.CuArray(μu_host)
μw = CUDA.CuArray(μw_host)


# Combined forcing functions that include both wavemaker and sponge
@inline function fu_total(i, j, k_, grid, clock, fields, p)
    # Wavemaker forcing
    x, _, _ = node(i, j, k_, grid, Face(),   Center(), Center())
    _, _, z = node(i, j, k_, grid, Face(),   Center(), Center())
    ϕ = phase(x, z, clock.time, p)
    F  = Fenv(x, z, p)
    Fx_ = Fx(x, z, p)
    Fz_ = Fz(x, z, p)
    wavemaker_force = p.Cf * ( -F * (p.m / p.k) * cos(ϕ) -
                              (1 / p.k) * Fz_ * sin(ϕ) +
                              (p.m / p.k^2) * Fx_ * sin(ϕ) )
    
    # Sponge forcing
    sponge_force = @inbounds -p.μu[i] * fields.u[i, j, k_]
    
    return wavemaker_force + sponge_force
end

@inline function fw_total(i, j, k_, grid, clock, fields, p)
    # Wavemaker forcing
    x, _, _ = node(i, j, k_, grid, Center(), Center(), Face())
    _, _, z = node(i, j, k_, grid, Center(), Center(), Face())
    ϕ = phase(x, z, clock.time, p)
    F = Fenv(x, z, p)
    wavemaker_force = p.Cf * F * cos(ϕ)
    
    # Sponge forcing
    sponge_force = @inbounds -p.μw[i] * fields.w[i, j, k_]
    
    return wavemaker_force + sponge_force
end


forcing = (
  u = Forcing(fu_total; parameters=(; Cf, ω, k, m, xc, zc, σx, σz, μu), discrete_form=true),
  w = Forcing(fw_total; parameters=(; Cf, ω, k, m, xc, zc, σx, σz, μw), discrete_form=true),
  b = Forcing(fb;       parameters=(; Cf, ω, k, m, xc, zc, σx, σz, N2), discrete_form=true),
)



# --- model ---
model = NonhydrostaticModel(; grid,
    tracers  = (:b,),
    buoyancy = BuoyancyTracer(),
    closure  = ScalarDiffusivity(ν=1e-6, κ=1.43e-7),
    coriolis = nothing,
    forcing  = forcing
)



# background stratification; start at rest
bbar(z) = z >= -hp + δp/2  ? 0.0 : (g * Δp / 2) * erf((z + hp) / (δp/2)) + (z < -hp ? N0^2 * (z + hp) : 0.0)
set!(model, u=0, v=0, w=0, b=(x,y,z)->bbar(z))



stop_time = Tf * 35
# --- integrate and write fields ---
sim = Simulation(model; Δt=0.005, stop_time=stop_time)
sim.output_writers[:jld] = JLD2Writer(model,
    (; u=model.velocities.u, w=model.velocities.w, b=model.tracers.b),
    schedule = TimeInterval(1),
    filename = "Apr-m2_timeseries-00.jld2",
    overwrite_existing = true,
)
run!(sim)
println("Saved: Apr-m2_timeseries-00.jld2")
