"""Integration tests for the adaptive complex-plane solver (no explicit ranges).

These tests verify that ``complex_neff`` / ``all_complex_neff`` return correct
results for reasonable structures **without** the user passing ``re_range`` or
``im_range``. The adaptive solver should handle the full range of mode types:

- Strongly leaky modes (large Im)
- Weakly leaky modes (Im ≈ 1e-9, previously a coverage gap)
- QNMs (both claddings Outgoing, Im < 0)
- Lossy guided modes (Im < 0, small)
- Lossless guided modes (Im = 0)

The test structure is the asymmetric leaky slab from ``test_complex.py``::

    air (n=1.0) | core (n=2.0, 0.6 µm) | air gap (t µm) | substrate (n=2.2)

As the gap ``t`` shrinks the mode couples more strongly to the substrate,
increasing the radiation loss.  The adaptive solver should find the TE0 mode
for all gap thicknesses without manual search-window tuning.
"""

import math

import pytest

import remsol as rs

OMEGA = 2.0 * math.pi / 1.55  # k0 at λ = 1.55 µm  [rad/µm]


def make_leaky_slab(gap_t: float) -> rs.MultiLayer:
    """Build the asymmetric leaky slab with right Outgoing boundary."""
    ml = rs.MultiLayer(
        [
            rs.Layer(1.0, 1.0),
            rs.Layer(2.0, 0.6),
            rs.Layer(1.0, gap_t),
            rs.Layer(2.2, 1.0),
        ]
    )
    ml.set_right_boundary(rs.BoundaryCondition.Outgoing)
    return ml


def make_multilayer_leaky_stack(*, both_outgoing: bool = False) -> rs.MultiLayer:
    """Build a coupled-core stack with five finite-material interfaces."""
    ml = rs.MultiLayer(
        [
            rs.Layer(1.0, 1.0),
            rs.Layer(2.2, 0.25),
            rs.Layer(1.45, 0.18),
            rs.Layer(2.0, 0.35),
            rs.Layer(1.0, 0.8),
            rs.Layer(2.4, 1.0),
        ]
    )
    if both_outgoing:
        ml.set_left_boundary(rs.BoundaryCondition.Outgoing)
    ml.set_right_boundary(rs.BoundaryCondition.Outgoing)
    return ml


# ── Weakly leaky mode (the former coverage gap) ──────────────────────────────


def test_adaptive_finds_weakly_leaky_mode():
    """t=1.5 µm: Im ≈ 1e-9, previously a coverage gap. Must find a mode."""
    ml = make_leaky_slab(1.5)
    result = ml.complex_neff(OMEGA, rs.Polarization.TE)
    assert result is not None, "Adaptive solver should find the weakly leaky mode"
    re_neff, im_neff = result
    assert abs(re_neff - 1.8043) < 0.01, f"Re(neff)={re_neff:.6f} should be ≈ 1.8043"
    assert abs(im_neff) < 1e-6, f"|Im(neff)|={im_neff:.2e} should be tiny"


# ── Full leaky sweep ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "gap_t", [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0, 10.0]
)
def test_adaptive_leaky_sweep_finds_mode(gap_t):
    """For every gap thickness the adaptive solver must find the TE0 mode."""
    ml = make_leaky_slab(gap_t)
    result = ml.complex_neff(OMEGA, rs.Polarization.TE)
    assert result is not None, f"No mode found for gap t={gap_t} µm"
    re_neff, im_neff = result
    # Re(neff) should be close to the isolated-system guided mode (≈ 1.8043).
    assert abs(re_neff - 1.8043) < 0.02, (
        f"Re(neff)={re_neff:.6f} should be ≈ 1.8043 for gap t={gap_t}"
    )
    # Im(neff) should be ≥ 0 (leaky mode convention) or ≈ 0 (weakly leaky / guided).
    assert im_neff >= -1e-8, (
        f"Im(neff)={im_neff:.2e} should be ≥ 0 (or ≈ 0) for leaky mode, gap t={gap_t}"
    )


def test_adaptive_leaky_im_decreases_with_gap():
    """|Im(neff)| should decrease (less leakage) as the gap grows."""
    gaps = [0.2, 0.5, 1.0]
    im_values = []
    for t in gaps:
        ml = make_leaky_slab(t)
        result = ml.complex_neff(OMEGA, rs.Polarization.TE)
        assert result is not None, f"No mode found for gap t={t}"
        _, im = result
        im_values.append(im)
    # Im should decrease (less leakage) as gap grows.
    assert im_values[0] > im_values[1] > im_values[2], (
        f"Im values should decrease with growing gap: {im_values}"
    )


# ── QNM (both claddings Outgoing) ────────────────────────────────────────────


def test_adaptive_qnm_no_ranges():
    """QNM with both Outgoing BCs, no explicit ranges: Im < 0."""
    ml = rs.MultiLayer(
        [
            rs.Layer(1.0, 1.0),
            rs.Layer(2.0, 0.6),
            rs.Layer(1.0, 0.5),
            rs.Layer(2.2, 1.0),
        ]
    )
    ml.set_left_boundary(rs.BoundaryCondition.Outgoing)
    ml.set_right_boundary(rs.BoundaryCondition.Outgoing)
    result = ml.complex_neff(OMEGA, rs.Polarization.TE)
    assert result is not None, "Adaptive solver should find the QNM"
    re_neff, im_neff = result
    assert im_neff < 0.0, f"QNM Im(neff) must be < 0, got {im_neff:.6e}"
    assert 1.0 < re_neff < 2.2, f"QNM Re(neff)={re_neff:.6f} outside (1.0, 2.2)"


def test_adaptive_multilayer_qnm_no_ranges():
    """A coupled-core QNM is found without reducing the stack to one core layer."""
    ml = make_multilayer_leaky_stack(both_outgoing=True)
    result = ml.complex_neff(OMEGA, rs.Polarization.TE)
    assert result is not None, "Adaptive solver should find the multilayer QNM"
    re_neff, im_neff = result
    assert re_neff == pytest.approx(1.6506, abs=0.01)
    assert im_neff < 0.0, f"QNM Im(neff) must be < 0, got {im_neff:.6e}"


# ── General multilayer structures ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("polarization", "expected_re"),
    [
        (rs.Polarization.TE, 1.7967),
        (rs.Polarization.TM, 1.6100),
    ],
)
def test_adaptive_coupled_core_leaky_modes(polarization, expected_re):
    """The default search handles a multilayer core for both polarizations."""
    modes = make_multilayer_leaky_stack().all_complex_neff(OMEGA, polarization)
    assert len(modes) >= 2, "Expected both coupled-core leaky modes"
    assert modes[0][0] == pytest.approx(expected_re, abs=0.01)
    assert all(im >= 0.0 for _, im in modes), modes


def test_adaptive_isolated_probe_uses_external_cladding():
    """An internal low-index spacer must remain part of the isolated device."""
    interior = [
        rs.Layer(2.25, 0.3),
        rs.Layer(1.0, 0.15),
        rs.Layer(2.05, 0.4),
        rs.Layer(1.35, 0.9),
    ]
    isolated = rs.MultiLayer([rs.Layer(1.3, 1.0), *interior, rs.Layer(1.3, 1.0)])
    reference_modes = isolated.all_neff(OMEGA, rs.Polarization.TE)
    assert len(reference_modes) >= 2

    leaky = rs.MultiLayer([rs.Layer(1.3, 1.0), *interior, rs.Layer(2.5, 1.0)])
    leaky.set_right_boundary(rs.BoundaryCondition.Outgoing)
    adaptive_modes = leaky.all_complex_neff(OMEGA, rs.Polarization.TE)

    for reference_re in reference_modes[:2]:
        assert any(abs(re - reference_re) < 0.01 for re, _ in adaptive_modes), (
            f"No adaptive leaky mode found near isolated mode {reference_re:.6f}: "
            f"{adaptive_modes}"
        )


# ── Lossy guided ─────────────────────────────────────────────────────────────


def test_adaptive_lossy_guided():
    """Lossy core slab, no explicit ranges: Im < 0, Re ≈ lossless value."""
    ml = rs.MultiLayer(
        [
            rs.Layer(1.0, 1.0),
            rs.Layer(2.0 - 0.01j, 0.6),
            rs.Layer(1.0, 1.0),
        ]
    )
    result = ml.complex_neff(OMEGA, rs.Polarization.TE)
    assert result is not None, "Adaptive solver should find the lossy mode"
    re_neff, im_neff = result
    assert abs(re_neff - 1.8043) < 0.01, f"Re(neff)={re_neff:.6f} should be ≈ 1.8043"
    assert im_neff < 0.0, f"Im(neff)={im_neff:.6e} should be < 0 for lossy core"


def test_adaptive_lossy_coupled_core_finds_multiple_modes():
    """Material loss is handled for an asymmetric multilayer coupled core."""
    ml = rs.MultiLayer(
        [
            rs.Layer(1.2, 1.0),
            rs.Layer(2.2 - 0.008j, 0.25),
            rs.Layer(1.45, 0.18),
            rs.Layer(2.0 - 0.004j, 0.35),
            rs.Layer(1.3, 1.0),
        ]
    )
    modes = ml.all_complex_neff(OMEGA, rs.Polarization.TE)
    assert len(modes) >= 2, "Expected both lossy coupled-core modes"
    assert modes[0][0] == pytest.approx(1.8106, abs=0.01)
    assert modes[1][0] == pytest.approx(1.5544, abs=0.01)
    assert all(im < 0.0 for _, im in modes), modes


# ── Lossless guided ──────────────────────────────────────────────────────────


def test_adaptive_lossless_guided():
    """Lossless slab, no explicit ranges: Im = 0."""
    ml = rs.MultiLayer(
        [
            rs.Layer(1.0, 1.0),
            rs.Layer(2.0, 0.6),
            rs.Layer(1.0, 1.0),
        ]
    )
    result = ml.complex_neff(OMEGA, rs.Polarization.TE)
    assert result is not None, "Adaptive solver should find the guided mode"
    re_neff, im_neff = result
    assert abs(re_neff - 1.8043) < 0.01, f"Re(neff)={re_neff:.6f} should be ≈ 1.8043"
    assert abs(im_neff) < 1e-10, f"Im(neff)={im_neff:.2e} should be 0 for lossless"


# ── Explicit ranges preserve legacy behaviour ────────────────────────────────


def test_explicit_ranges_preserve_legacy():
    """Passing explicit im_range should use the legacy single-rectangle path."""
    ml = make_leaky_slab(1.5)
    # The weakly leaky mode (Im ≈ 1e-9) is below im_min=1e-3.
    result = ml.complex_neff(OMEGA, rs.Polarization.TE, im_range=(1e-3, 0.15))
    # Either no mode found, or the mode has Im > 1e-3 (not the weakly leaky one).
    if result is not None:
        _, im = result
        assert abs(im) >= 1e-3, (
            f"Legacy path should not find modes below im_min=1e-3, got Im={im:.2e}"
        )
