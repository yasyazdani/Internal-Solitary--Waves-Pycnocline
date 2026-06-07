# Oceananigans 0.97.7
# CUDA 5.8.3
#OffsetArrays 1.17.0
#JLD2 0.5.15

# E1 setup if you set f= 0
# if you change f to non zero => the beam angle 45 but ω will be diffrent from the paper
# ω = sqrt((N2 + f^2) / 2)  here, paper: ω = sqrt((N2) / 2) 
# θ: beam angle 45 so => m = -k which is equivalent to  m = -k sqrt[(N^2-w^2)/(ω^2- f^2)]


# march 23: chcek the impact of thickness
# it saves file as 



using Oceananigans
using Oceananigans.OutputWriters: JLD2Writer, TimeInterval
using Oceananigans.Grids: node
using Oceananigans.Coriolis: FPlane
using SpecialFunctions
using CUDA

#  domain & constants (E2 ) 
Lx, H  = 6.0, 0.95
dx     = 0.004          # 2 mm
Nx     =  Int(round(Lx / dx))
Nz     = 238            # vertical count (stretched grid below sets dz min/max targets)

#Parameters
N2 =  0.36
f  = 0.4  #0.35
#ω = sqrt(N2 / 2) # disperssion
ω = sqrt((N2 + f^2) / 2) 


hp = 0.02
δp = 0.01  # test various thicknes δp(smaller and larger than 0.01 but not larger than 0.04 or even 0.03)
g  = 9.81
Δp = 0.0205
N0 = 0.6
N02 = N0^2


@inline function N2_of_z(z; hp, δp, g, Δp, N0)
    a = 2 * (z + hp) / δp
    N2_pyc = (g * Δp) * (2 / (sqrt(π) * δp)) * exp(-a^2)   # pycnocline term (all z)
    N2_low = (z < -hp) ? N0^2 : 0.0                        # lower layer term
    return N2_pyc + N2_low
end

#  λx_res  =  λx in Nico 2011 E1
λx_res = 0.536
k = 2π / λx_res
m = -k # * sqrt((N02 - ω^2) / (ω^2 - f^2))

As_target = 0.009 # too big : 0.02  0.013 ,12 
Tf = 2π / ω          # forcing period (paper's Tf)
θ = atan(sqrt((ω^2 - f^2) / (N02 - ω^2)))
Uf = As_target * N0 * λx_res * sin(θ) # => U0 by Eq 20, then they used Uf ≈ U0? 
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

# E1 target: dzm = 0.4 mm, dzM = 4 mm
z_faces = stretched_z_faces(; H, Nz, dzm = 0.0004, dzM = 0.004, zt = -3hp, w = 0.1H)

# Non-uniform grid
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

# based on Nico for E1
λx = λx_res
# envelope widths (match Diamessis et al. 2014)
σx = 0.538 * λx
σz = 0.538 * λx

# wavemaker geometry (keep xc if you want)
xc = 0.6

# match their vertical offset from pycnocline: zcen ≈ z_pyc - 2.05 λx
zc = - 0.8 #-hp  - 2.05 * λx
# ------------------------------------------------------


# --- sponge layer on the right: Herbaut-style relaxation ---
ℓs = 1    #   sponge layer length # same as E1
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
    coriolis = FPlane(f=f),
    forcing  = forcing
)



# background stratification; start at rest

bbar(z) = z >= -hp  ? 0.0 : (g * Δp / 2) * erf((z + hp) / (δp/2)) + (z < -hp ? N0^2 * (z + hp) : 0.0)  # top b is zero
#bbar(z) = (g * Δp / 2) * erf((z + hp) / (δp/2)) +
#          (z < -hp ? N0^2 * (z + hp) : 0.0)

set!(model, u=0, v=0, w=0, b=(x,y,z)->bbar(z))



stop_time = 350 # Tf * 20
# --- integrate and write fields ---
sim = Simulation(model; Δt=0.005, stop_time= stop_time)
sim.output_writers[:jld] = JLD2Writer(model,
    (; u=model.velocities.u, w=model.velocities.w, b=model.tracers.b),
    schedule = TimeInterval(1),
    filename = "Apr-f4-Mode1_timeseries-00.jld2",
    overwrite_existing = true,
)
run!(sim)
println("Saved: Apr-f4-Mode1_timeseries-00.jld2")
