"""
One-sided leaky mode sweep: air | n=2.0 core | air gap (t) | n=2.2 substrate
=============================================================================

Physics
-------
Because n_substrate (2.2) > n_core (2.0), the core mode has no truly guided
solution: it is evanescently confined on the *left* (air, n=1.0) but radiates
into the *right* substrate.  The correct solver for this situation uses:

    left cladding  (air, n=1.0)      → kz_physical  (evanescent decay)
    right cladding (substrate, n=2.2) → kz_outgoing  (outgoing radiation)

With the unified API, this is achieved by setting the right boundary condition
to ``BoundaryCondition.Outgoing`` (either on the MultiLayer object or as a
per-call override to ``complex_neff``).

Formerly these were separate ``leaky_neff`` / ``all_leaky_neff`` methods;  now
they are just ``complex_neff`` / ``all_complex_neff`` with ``right_bc=Outgoing``.

Sign convention
~~~~~~~~~~~~~~~
``leaky_neff`` operates with real ω and complex k∥ = neff · k₀.

  Im(neff) > 0  →  field decays along the propagation direction (+x)
                    i.e. the mode loses power as it travels down the waveguide.

This is the standard "leaky mode" convention used in waveguide optics.
Compare with ``qnm_neff``, which fixes real k∥ and uses complex ω, giving
Im(neff) < 0 for temporally decaying resonances.

Expected behaviour
~~~~~~~~~~~~~~~~~~
  Re(neff)  ≈  neff of the isolated core waveguide  (≈ 1.8043 here)
  Im(neff)  >  0                (power leaks into the substrate)
  Re(neff)  increases as t shrinks  (more high-n substrate overlap → higher n_eff)
  Im(neff)  grows ~exp(−2γt) as t shrinks (stronger tunnelling, γ ≈ 6 rad/µm)

Search strategy
~~~~~~~~~~~~~~~
The winding-number algorithm requires im_max to be large enough relative to the
actual Im(neff) value for reliable pole counting.  A two-stage approach is used:

  Stage 1  re_range = (1.7, 2.0)  — keeps TE1 (Re ≈ 1.19) out of the rectangle,
           cascade of three im windows covering Im from 1e-8 to 0.15.
           Reliably finds t = 0.2 … 0.9 µm.

  Stage 2  re_range = (1.75, 1.85)  — very tight re window around the TE0 pole,
           im_range = (1e-8, 0.01).
           Reliably finds t = 1.0 … 1.2 µm.

  Stage 3  Real-axis solver (Im effectively zero).
           Works for t ≥ 2 µm where Im < 10⁻¹⁰.

For t ≈ 1.5 µm the imaginary part is around 10⁻⁹, below the minimum search
floor.  The mode is real-axis-detectable only for t ≥ 1.8 µm; in the gap
(1.2–1.8 µm) neither solver reaches it.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

import remsol
from remsol import Polarization as pol

# ── Constants ─────────────────────────────────────────────────────────────────

OMEGA = 2.0 * math.pi / 1.55  # k0 at λ = 1.55 µm  [rad/µm]

GAP_THICKNESSES = [
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.5,
    2.0,
    5.0,
    10.0,
]

# Stage-1 im windows: broad re_range (1.7, 2.0), cascade of im windows.
# Each window is (im_min, im_max); all positive (upper half-plane).
#   Window A: strongly leaky,   Im ≈ 1e-3 … 0.15   (t ≈ 0.2–0.4 µm)
#   Window B: moderately leaky, Im ≈ 1e-5 … 0.01   (t ≈ 0.4–0.8 µm)
#   Window C: weakly leaky,     Im ≈ 1e-8 … 1e-3   (t ≈ 0.8–1.0 µm)
STAGE1_RE = (1.7, 2.0)
STAGE1_WINDOWS = [
    (1e-3, 0.15),
    (1e-5, 0.01),
    (1e-8, 1e-3),
]

# Stage-2: very tight re_range around TE0, wide im window.
# Finds very weakly leaky modes (t ≈ 1.0–1.2 µm, Im ≈ 5e-8 … 6e-7).
STAGE2_RE = (1.75, 1.85)
STAGE2_WIN = (1e-8, 0.01)


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_multilayer(t: float) -> remsol.MultiLayer:
    """Build the leaky slab for a given air-gap thickness t [µm].

    The right boundary is set to Outgoing so that ``complex_neff`` uses the
    one-sided leaky boundary condition automatically.
    """
    ml = remsol.MultiLayer(
        [
            remsol.Layer(n=1.0, d=1.0),  # air cladding (left, semi-infinite)
            remsol.Layer(n=2.0, d=0.6),  # waveguide core
            remsol.Layer(n=1.0, d=t),  # air gap — controls leakage rate
            remsol.Layer(n=2.2, d=1.0),  # substrate (right, semi-infinite)
        ]
    )
    ml.set_right_boundary(remsol.BoundaryCondition.Outgoing)
    return ml


def find_te0_leaky(ml: remsol.MultiLayer) -> tuple[float, float] | None:
    """
    Two-stage search for the TE0 one-sided leaky mode.

    Uses ``complex_neff`` with the right boundary condition already set to
    ``Outgoing`` on the MultiLayer object.  Returns
    ``(Re(neff), Im(neff))`` with ``Im(neff) > 0``, or ``None`` if
    the imaginary part is below the search floor (``Im < ~1e-11``).
    """
    # Stage 1: re_range = (1.7, 2.0), cascade of three im windows.
    for win in STAGE1_WINDOWS:
        r = ml.complex_neff(OMEGA, pol.TE, re_range=STAGE1_RE, im_range=win)
        if r is not None and r[1] > 0:
            return r

    # Stage 2: very tight re_range, wide im window.
    r = ml.complex_neff(OMEGA, pol.TE, re_range=STAGE2_RE, im_range=STAGE2_WIN)
    if r is not None and r[1] > 0:
        return r

    return None


# ── Reference: isolated-core guided mode ──────────────────────────────────────

isolated_core = remsol.MultiLayer(
    [
        remsol.Layer(n=1.0, d=1.0),
        remsol.Layer(n=2.0, d=0.6),
        remsol.Layer(n=1.0, d=1.0),
    ]
)
neff_isolated = isolated_core.neff(OMEGA, pol.TE)
neff_isolated_str = f"{neff_isolated:.8f}" if neff_isolated is not None else "N/A"

print(f"Isolated core neff (TE0, symmetric air|n=2|air): {neff_isolated_str}")
print()

# ── Main sweep ────────────────────────────────────────────────────────────────

print(f"{'gap t':>8}  {'Re(neff)':>14}  {'Im(neff)':>14}  {'source':>22}")
print("-" * 66)

# Data for plotting
gap_plot: list[float] = []
re_plot: list[float] = []
im_plot: list[float] = []
source_plot: list[str] = []  # "leaky" | "real-axis" | "gap"

for t in GAP_THICKNESSES:
    ml = make_multilayer(t)

    leaky = find_te0_leaky(ml)

    if leaky is not None:
        re, im = leaky
        print(f"t = {t:5.2f} µm  {re:14.8f}  {im:14.4e}  {'leaky solver':>22}")
        gap_plot.append(t)
        re_plot.append(re)
        im_plot.append(im)
        source_plot.append("leaky")
        continue

    # Leaky solver found nothing — try real-axis for Im ≈ 0 regime.
    guided = [x for x in ml.all_neff(OMEGA, pol.TE) if x > 1.7]
    if guided:
        re = guided[0]
        print(
            f"t = {t:5.2f} µm  {re:14.8f}  {'< 1e-10':>14}  {'real-axis  (Im ≈ 0)':>22}"
        )
        gap_plot.append(t)
        re_plot.append(re)
        im_plot.append(0.0)
        source_plot.append("real-axis")
        continue

    # Neither solver reached it (t ≈ 1.2–1.8 µm transition region).
    print(f"t = {t:5.2f} µm  {'—':>14}  {'—':>14}  {'Im < 1e-11 (gap)':>22}")

print()
print("Notes:")
print(
    "  • Im(neff) > 0  →  mode decays along the propagation direction (spatial loss)."
)
print("  • Re(neff) increases as t decreases: more high-n substrate overlap.")
print("  • Im(neff) grows ~exp(−2γt) with γ ≈ 6 rad/µm (evanescent decay in air gap).")
print("  • For t > 1.2 µm, Im < ~1e-11 — effectively zero; real-axis is sufficient.")
print("  • For t ≈ 1.5 µm, neither complex nor real-axis solver finds the mode.")

# ── Plots ─────────────────────────────────────────────────────────────────────

# Only plot points where we actually have a result.
leaky_t = [t for t, s in zip(gap_plot, source_plot) if s == "leaky"]
leaky_re = [re for re, s in zip(re_plot, source_plot) if s == "leaky"]
leaky_im = [im for im, s in zip(im_plot, source_plot) if s == "leaky"]

real_t = [t for t, s in zip(gap_plot, source_plot) if s == "real-axis"]
real_re = [re for re, s in zip(re_plot, source_plot) if s == "real-axis"]

if len(gap_plot) >= 2:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ── Left: Re(neff) vs gap ────────────────────────────────────────────────
    ax = axes[0]
    if leaky_t:
        ax.semilogx(
            leaky_t,
            leaky_re,
            "o-",
            color="steelblue",
            linewidth=1.8,
            markersize=7,
            label="leaky solver (Im > 0)",
        )
    if real_t:
        ax.semilogx(
            real_t,
            real_re,
            "s--",
            color="darkorange",
            linewidth=1.4,
            markersize=7,
            label="real-axis (Im ≈ 0)",
        )
    if neff_isolated is not None:
        ax.axhline(
            neff_isolated,
            color="gray",
            linewidth=1.2,
            linestyle=":",
            label=f"isolated core  neff = {neff_isolated:.4f}",
        )
    ax.set_xlabel("Gap thickness  t  (µm)")
    ax.set_ylabel("Re(neff)")
    ax.set_title("Re(neff) vs gap\n(increases as t shrinks — more substrate overlap)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # ── Middle: Im(neff) vs gap (log-log) ────────────────────────────────────
    ax = axes[1]
    if leaky_t:
        ax.loglog(
            leaky_t,
            leaky_im,
            "o-",
            color="steelblue",
            linewidth=1.8,
            markersize=7,
            label="|Im(neff)|",
        )
        # Overlay the exp(−2γt) guide line anchored on the first leaky point.
        gamma = OMEGA * math.sqrt(max(leaky_re) ** 2 - 1.0)
        t_fit = np.logspace(
            math.log10(min(leaky_t) * 0.8),
            math.log10(max(leaky_t) * 1.5),
            60,
        )
        t_anchor, im_anchor = leaky_t[-1], leaky_im[-1]
        ax.loglog(
            t_fit,
            im_anchor * np.exp(-2 * gamma * (t_fit - t_anchor)),
            "--",
            color="tomato",
            linewidth=1.3,
            label=rf"$\propto e^{{-2\gamma t}}$,  γ = {gamma:.2f} rad/µm",
        )
    ax.set_xlabel("Gap thickness  t  (µm)")
    ax.set_ylabel("Im(neff)")
    ax.set_title("Im(neff) vs gap  (log–log)\n(exponential tunnelling through air gap)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # ── Right: poles in the complex neff plane ────────────────────────────────
    ax2 = axes[2]
    if leaky_t:
        scatter = ax2.scatter(
            leaky_re,
            leaky_im,
            c=leaky_t,
            cmap="viridis_r",
            s=80,
            zorder=3,
        )
        for t, re, im in zip(leaky_t, leaky_re, leaky_im):
            ax2.annotate(
                f"t={t}",
                (re, im),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
            )
        cbar = fig.colorbar(scatter, ax=ax2)
        cbar.set_label("Gap t (µm)")
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    if neff_isolated is not None:
        ax2.axvline(
            neff_isolated,
            color="gray",
            linewidth=0.8,
            linestyle=":",
            label="isolated core neff",
        )
    ax2.set_xlabel("Re(neff)")
    ax2.set_ylabel("Im(neff)")
    ax2.set_title(
        "Leaky-mode poles in the complex neff plane\n"
        "(Im > 0: spatial decay along waveguide)"
    )
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("leaky_sweep.png", dpi=150)
    plt.close()
    print("\nPlot saved to leaky_sweep.png")
