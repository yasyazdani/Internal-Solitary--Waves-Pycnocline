# solve eigenvalue problem 
# Figure 4 in NG-2011

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS: Experiment E1
# ============================================================
sigma_redux_min = 0.05
sigma_redux_max = 5.0
n_sigma = 220

H = 0.95
hp = 0.02
delta_p = 0.01
N0 = 0.6
omega0 = 0.424
Delta_p = 0.0205
g = 9.81
lambda_x = 0.536          # m
modemax = 3
dz = 0.0004               # 0.4 mm

# beam phase speed
k_x = 2 * np.pi / lambda_x
v_phi = omega0 / k_x


# ============================================================
# YOUR N^2(z) PROFILE
# ============================================================
def N2_profile(z, H=0.95, hp=0.02, delta_p=0.01, N0=0.6, Delta_p=0.0205, g=9.81):
    z = np.asarray(z)

    gaussian = (2 / np.sqrt(np.pi)) * g * (Delta_p / delta_p) * np.exp(
        -((z + hp) / (delta_p / 2)) ** 2
    )

    background = np.zeros_like(z)
    background[(z >= -H) & (z < -hp)] = N0**2
    background[(z >= -hp) & (z <= 0)] = 0.0

    return gaussian + background


# ============================================================
# SHOOTING STEP
# bottom -> top, same recurrence logic
# ============================================================
def shoot_mode(wn_int, sigma, bvi, dz):
    C0 = -sigma**2
    nr = len(bvi)

    phi_int = np.zeros(nr)
    phi_int[0] = 0.0
    phi_int[1] = 0.01

    for k in range(1, nr - 1):
        A0 = bvi[k]**2 - sigma**2
        fun = (wn_int**2) * (-A0) / C0
        phi_int[k + 1] = phi_int[k] * (2.0 - fun * (dz**2)) - phi_int[k - 1]

        # prevent overflow during march
        if abs(phi_int[k + 1]) > 1e100:
            phi_int /= 1e100

    return phi_int


# ============================================================
# MODE COUNTER
# same style as their logic
# ============================================================
def count_no_max(phi_int):
    dphi = np.gradient(phi_int)
    dphi[dphi > 0.0] = 1.0
    dphi[dphi < 0.0] = -1.0
    ddphi = np.gradient(dphi)
    jj = np.where(ddphi != 0)[0]
    return len(jj) / 2.0


# ============================================================
# ONE-SIGMA SOLVER
# correction: uses continuation guesses from previous sigma
# ============================================================
def modes_nt_from_N2_profile(
    sigma_redux,
    H,
    hp,
    delta_p,
    N0,
    Delta_p,
    g,
    dz,
    modemax,
    omega0,
    prev_wn=None
):
    sigma = sigma_redux * omega0

    zi = np.arange(-H, 0 + dz, dz)
    nr = zi.size

    N2i = N2_profile(zi, H=H, hp=hp, delta_p=delta_p, N0=N0, Delta_p=Delta_p, g=g)
    bvi = np.sqrt(np.maximum(N2i, 0.0))

    #avbvi = np.trapz(bvi, zi) / H
    avbvi = np.trapezoid(bvi, zi)/H
    C0 = -sigma**2

    phi = np.zeros((modemax, nr))
    wn = np.full(modemax, np.nan)

    no_max_hist = np.full(500, np.nan)

    m = 1
    mm = 1

    while mm < modemax + 1 and m < 500:
        odev = (-1)**m
        incr = 0.75 / np.sqrt(m)

        # crude guess, same spirit as original code
        c_guess = avbvi * H / (m * np.pi)
        wn0 = np.sqrt(-C0) / c_guess

        # correction: if previous converged mode exists, use it
        if prev_wn is not None and (m - 1) < len(prev_wn) and np.isfinite(prev_wn[m - 1]):
            wn_int = prev_wn[m - 1]
        else:
            wn_int = wn0

        phi_int = np.zeros(nr)
        phi_int[0] = 0.0
        phi_int[1] = 0.01

        for j in range(1, 61):
            phi_int = shoot_mode(wn_int, sigma, bvi, dz)

            # same update formula, but continuation guess is now used
            base_scale = prev_wn[m - 1] if (
                prev_wn is not None and (m - 1) < len(prev_wn) and np.isfinite(prev_wn[m - 1])
            ) else wn0

            wn_int = wn_int - odev * incr * np.sign(phi_int[-1]) * base_scale / (2**j)

        maxabs = np.max(np.abs(phi_int))
        if maxabs > 0:
            phi_int = phi_int / maxabs

        no_max_hist[m - 1] = count_no_max(phi_int)

        if abs(no_max_hist[m - 1] - mm - 1) < 1e-14 and int(no_max_hist[m - 1]) <= modemax:
            mm = int(no_max_hist[m - 1])

        # slightly relaxed tolerance to avoid losing the branch
        if abs(phi_int[-1]) < 1e-5 and 1 <= mm <= modemax:
            if abs(mm - 1) < 1e-14:
                phi[mm - 1, :] = phi_int
                wn[mm - 1] = wn_int
            else:
                if np.isfinite(no_max_hist[m - 2]):
                    if (no_max_hist[m - 1] - no_max_hist[m - 2] - 1) < 1e-14:
                        phi[mm - 1, :] = phi_int
                        wn[mm - 1] = wn_int

        m += 1

    cphis = sigma / wn
    return cphis, wn, phi, zi


# ============================================================
# BUILD FIGURE 4
# correction: continuation in sigma
# ============================================================
sigma_redux_vals = np.linspace(sigma_redux_min, sigma_redux_max, n_sigma)

c1 = np.full(n_sigma, np.nan)
c2 = np.full(n_sigma, np.nan)
c3 = np.full(n_sigma, np.nan)

prev_wn = np.full(modemax, np.nan)

for i, sigma_redux in enumerate(sigma_redux_vals):
    print(f"{i+1}/{n_sigma}   Omega/omega0 = {sigma_redux:.3f}")

    cphis, wn, phi, zi = modes_nt_from_N2_profile(
        sigma_redux=sigma_redux,
        H=H,
        hp=hp,
        delta_p=delta_p,
        N0=N0,
        Delta_p=Delta_p,
        g=g,
        dz=dz,
        modemax=modemax,
        omega0=omega0,
        prev_wn=prev_wn
    )

    if np.isfinite(cphis[0]):
        c1[i] = cphis[0]
    if np.isfinite(cphis[1]):
        c2[i] = cphis[1]
    if np.isfinite(cphis[2]):
        c3[i] = cphis[2]

    # correction: carry successful roots forward
    for m in range(modemax):
        if np.isfinite(wn[m]):
            prev_wn[m] = wn[m]

x = sigma_redux_vals
y1 = c1 / v_phi
y2 = c2 / v_phi
y3 = c3 / v_phi

plt.figure(figsize=(7.5, 5.0))

plt.plot(x, y1, color='black', linewidth=1.8)
plt.plot(x, y2, color='black', linewidth=1.8)
plt.plot(x, y3, color='black', linewidth=1.8)

plt.axvline(N0 / omega0, color='black', linestyle='--', linewidth=1.0)

plt.xlim(0, 5)
plt.ylim(0, 5)

plt.xlabel(r'$\Omega/\omega_0$', fontsize=18)
plt.ylabel(r'$c_n/v_\phi$', fontsize=18)

plt.text(3.95, 1.05, r'$c_1/v_\phi$', fontsize=14)
plt.text(3.95, 0.78, r'$c_2/v_\phi$', fontsize=14)
plt.text(3.95, 0.36, r'$c_3/v_\phi$', fontsize=14)

plt.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig("Fig4_like_paper_continuation.png", dpi=300, bbox_inches="tight")

