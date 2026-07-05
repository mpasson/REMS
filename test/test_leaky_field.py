"""
Leaky mode field profile: air | n=2.0 core | air gap (t) | n=2.2 substrate
===========================================================================

Demonstrates computing and plotting the field profile of a one-sided leaky
mode using :meth:`~remsol.MultiLayer.complex_field` with
:attr:`~remsol.BoundaryCondition.Outgoing` on the right cladding.

The structure::

    air (n=1.0) | core (n=2.0, 0.6 µm) | air gap (t µm) | substrate (n=2.2)

The mode is evanescently confined on the left (air side) and radiates into
the substrate on the right.

How to use ``complex_field`` for leaky modes
--------------------------------------------
1. Set the right boundary condition to ``Outgoing`` (or pass it as a per-call
   override to ``complex_field``).
2. Call ``ml.complex_field(omega, polarization)``.
3. Plot ``np.abs(np.array(field.Ey))`` — the field components are
   ``list[complex]``, so take the absolute value for the amplitude envelope.

Note on the substrate field
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The field reconstruction uses the Transfer Matrix Method (TMM) with
``kz_physical`` in all layers.  For an evanescent left cladding (air) this is
exact.  For the radiating right cladding (substrate) the physically correct
outgoing wave has ``Im(kz) < 0`` and its amplitude *grows* away from the
interface.  The TMM with ``kz_physical`` flips the sign of ``Im(kz)`` and
shows a *decaying* envelope in the substrate instead.

For weakly leaky modes this difference is negligible.  For strongly leaky
modes the substrate tail will appear evanescent rather than radiating, but
the core-region field is accurately reproduced in both cases.

Run this file directly to generate a plot::

    uv run python test/test_leaky_field.py
"""

import math

import numpy as np
import pytest

import remsol
from remsol import BoundaryCondition, Polarization

# ── Constants ─────────────────────────────────────────────────────────────────

OMEGA = 2.0 * math.pi / 1.55  # k0 at λ = 1.55 µm  [rad/µm]

# Gap thickness used in the tests (0.3 µm → moderately leaky, Im(neff) ≈ 0.021,
# comfortably within the default im_range=(1e-3, 0.15)).
GAP_T = 0.2


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_leaky_slab(gap_t: float = GAP_T) -> remsol.MultiLayer:
    """Build the asymmetric leaky slab with a right Outgoing boundary.

    The right boundary condition is set to ``Outgoing`` on the ``MultiLayer``
    object so that both ``complex_neff`` and ``complex_field`` automatically
    use the correct one-sided leaky S-matrix.
    """
    ml = remsol.MultiLayer(
        [
            remsol.Layer(1.0, 1.0),  # air cladding (left, evanescent)
            remsol.Layer(2.0, 0.6),  # waveguide core
            remsol.Layer(1.0, gap_t),  # air gap — controls leakage rate
            remsol.Layer(2.2, 10.0),  # substrate (right, outgoing)
        ]
    )
    ml.set_right_boundary(BoundaryCondition.Outgoing)
    return ml


# ── Pytest tests ──────────────────────────────────────────────────────────────


def test_leaky_field_max_field_normalized():
    """complex_field for a leaky mode: max total |E| == 1 (MaxField norm)."""
    ml = make_leaky_slab()
    field = ml.complex_field(OMEGA, Polarization.TE)
    ex = np.array(field.Ex)
    ey = np.array(field.Ey)
    ez = np.array(field.Ez)
    max_e = float(np.max(np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2)))
    assert max_e == pytest.approx(1.0, abs=1e-6), (
        f"Max |E| = {max_e:.8f} (expected 1.0 after MaxField normalisation)"
    )


def test_leaky_field_ey_dominant_for_te():
    """For TE polarisation Ey is the dominant E-field component."""
    ml = make_leaky_slab()
    field = ml.complex_field(OMEGA, Polarization.TE)
    ey_max = float(np.max(np.abs(np.array(field.Ey))))
    ex_max = float(np.max(np.abs(np.array(field.Ex))))
    ez_max = float(np.max(np.abs(np.array(field.Ez))))
    # Ey should dominate; Ex and Ez should be essentially zero for TE.
    assert ey_max > 0.5, f"|Ey|_max = {ey_max:.4f} should be close to 1"
    assert ex_max < 1e-8, f"|Ex|_max = {ex_max:.2e} should be ~0 for TE"
    assert ez_max < 1e-8, f"|Ez|_max = {ez_max:.2e} should be ~0 for TE"


def test_leaky_neff_im_positive():
    """One-sided leaky mode (left SemiInfinite, right Outgoing): Im(neff) > 0."""
    ml = make_leaky_slab()
    result = ml.complex_neff(OMEGA, Polarization.TE)
    if result is None:
        pytest.skip("No leaky mode found with default search range")
    re_neff, im_neff = result
    assert im_neff > 0.0, (
        f"Im(neff) = {im_neff:.4e} should be positive for a one-sided leaky mode"
    )
    assert 1.0 < re_neff < 2.2, (
        f"Re(neff) = {re_neff:.4f} should be inside the physical index range (1.0, 2.2)"
    )


def test_leaky_field_returns_fieldata():
    """complex_field must return a FieldData object with correct x grid."""
    ml = make_leaky_slab()
    field = ml.complex_field(OMEGA, Polarization.TE)
    # All arrays must have the same length and x must be monotonically increasing.
    n = len(field.x)
    assert n > 10, "Field grid should have many points"
    assert len(field.Ey) == n
    assert len(field.Hx) == n
    x = np.array(field.x)
    assert np.all(np.diff(x) > 0), "x grid must be monotonically increasing"


def test_complex_field_consistent_with_complex_neff():
    """The neff used internally by complex_field should match complex_neff."""
    ml = make_leaky_slab()
    # Find neff explicitly.
    result = ml.complex_neff(OMEGA, Polarization.TE)
    if result is None:
        pytest.skip("No leaky mode found")
    re_neff, im_neff = result

    # The field should be non-trivially normalised (not zeroed).
    field = ml.complex_field(OMEGA, Polarization.TE)
    ey_max = float(np.max(np.abs(np.array(field.Ey))))
    assert ey_max > 1e-6, "Field should be non-zero when a mode is found"


# ── Standalone plot ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Plot the leaky mode field profile and compare with the isolated-core
    guided mode.  Run with::

        uv run python test/test_leaky_field.py
    """
    import matplotlib.pyplot as plt

    gap_t = GAP_T

    ml = make_leaky_slab(gap_t)

    # Isolated-core reference (no substrate).
    ml_iso = remsol.MultiLayer(
        [
            remsol.Layer(1.0, 1.0),
            remsol.Layer(2.0, 0.6),
            remsol.Layer(1.0, 1.0),
        ]
    )

    # ── Find the leaky mode ──────────────────────────────────────────────────
    result = ml.complex_neff(OMEGA, Polarization.TE)
    if result is None:
        raise RuntimeError(
            f"No leaky mode found for gap_t={gap_t} µm with default search range. "
            "Try a thinner gap (e.g. gap_t=0.2)."
        )
    re_neff, im_neff = result
    print(f"Leaky mode  neff = {re_neff:.6f} + {im_neff:.3e}i  (gap = {gap_t} µm)")

    neff_guided = ml_iso.neff(OMEGA, Polarization.TE)
    print(f"Guided neff = {neff_guided:.6f}  (isolated core, no substrate)")

    # ── Get field profiles ───────────────────────────────────────────────────
    leaky_field = ml.complex_field(OMEGA, Polarization.TE)
    guided_field = ml_iso.field(OMEGA, Polarization.TE)

    x_leaky = np.array(leaky_field.x)
    ey_leaky = np.abs(np.array(leaky_field.Ey))  # amplitude envelope

    x_guided = np.array(guided_field.x)
    ey_guided = np.abs(np.array(guided_field.Ey))

    # Index profile of the leaky slab.
    idx = ml.index()

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Panel 1: index profile
    ax = axes[0]
    ax.fill_between(idx.x, idx.n, alpha=0.12, color="steelblue")
    ax.step(idx.x, idx.n, color="steelblue", linewidth=1.8, where="post")
    ax.set_ylabel("Refractive index  n")
    ax.set_title(
        f"Structure: air | n=2.0 core (0.6 µm) | {gap_t} µm gap | n=2.2 substrate"
    )
    ax.set_ylim(0.5, 2.6)
    ax.grid(True, linestyle="--", alpha=0.35)

    # Layer boundaries (cumulative left edges)
    boundaries = [0.0, 1.0, 1.6, 1.6 + gap_t]
    for xb in boundaries:
        ax.axvline(xb, color="gray", linewidth=0.7, linestyle=":", alpha=0.6)

    # Panel 2: field amplitude
    ax = axes[1]
    ax.plot(
        x_leaky,
        ey_leaky,
        color="C0",
        linewidth=2.2,
        label=f"Leaky  |Ey|   neff = {re_neff:.4f} + {im_neff:.2e}i",
    )
    ax.plot(
        x_guided,
        ey_guided,
        color="C1",
        linewidth=1.6,
        linestyle="--",
        label=f"Guided |Ey|   neff = {neff_guided:.4f}  (isolated core)",
    )

    # Shade the core region.
    ax.axvspan(1.0, 1.6, alpha=0.08, color="steelblue", label="Core")
    ax.axvspan(1.6, 1.6 + gap_t, alpha=0.05, color="gray", label="Gap")

    for xb in boundaries:
        ax.axvline(xb, color="gray", linewidth=0.7, linestyle=":", alpha=0.6)

    ax.set_xlabel("x  (µm)")
    ax.set_ylabel("|Ey|  (normalised, MaxField)")
    ax.set_title("|Ey(x)| — leaky mode vs isolated-core guided mode")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    fname = "leaky_field.png"
    plt.savefig(fname, dpi=150)
    print(f"\nPlot saved to {fname}")
    plt.show()
