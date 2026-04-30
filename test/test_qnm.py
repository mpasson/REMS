"""Integration tests for quasi-normal mode (QNM) and one-sided leaky mode solving.

QNMs satisfy outgoing-wave boundary conditions: instead of decaying evanescently
away from the guiding region, the field radiates outward in both claddings.  Their
effective indices are complex with Im(neff) < 0 (the mode decays in time as energy
leaks out).

With the unified API, QNMs are found by setting both claddings to
``BoundaryCondition.Outgoing`` (either on the ``MultiLayer`` object or as
per-call overrides to ``complex_neff`` / ``all_complex_neff``).

Physical test structure
-----------------------
Most tests use an *asymmetric* slab:

    air (n=1)  |  core (n=2, d=0.6 µm)  |  air gap (d=t µm)  |  substrate (n=2.2)

As the gap t shrinks the guided mode couples more strongly to the substrate,
increasing the radiation loss.  Expected behaviour:
- Im(neff) < 0 for all QNMs.
- |Im(neff)| increases as t decreases.
- Re(neff) stays close to the guided-mode value of the isolated core.
"""

import math

import pytest

import remsol as rs

# ── Shared frequency ──────────────────────────────────────────────────────────

OMEGA = 2 * math.pi / 1.55  # k0 at λ = 1.55 µm  (rad/µm)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_leaky_slab(gap_t: float) -> rs.MultiLayer:
    """Asymmetric slab with both claddings set to Outgoing (QNM boundary conditions)."""
    ml = rs.MultiLayer(
        [
            rs.Layer(1.0, 1.0),  # air cladding (left)
            rs.Layer(2.0, 0.6),  # waveguide core
            rs.Layer(1.0, gap_t),  # air gap  →  controls leakage
            rs.Layer(2.2, 1.0),  # substrate (right)
        ]
    )
    ml.set_left_boundary(rs.BoundaryCondition.Outgoing)
    ml.set_right_boundary(rs.BoundaryCondition.Outgoing)
    return ml


@pytest.fixture
def leaky_slab_thin():
    """Thin gap (0.5 µm) → strong coupling to substrate."""
    return make_leaky_slab(0.5)


@pytest.fixture
def leaky_slab_thick():
    """Thick gap (5.0 µm) → near-isolated core, very small Im(neff)."""
    return make_leaky_slab(5.0)


@pytest.fixture
def symmetric_slab():
    """Symmetric lossless slab (air / n=2 core / air) — no substrate leakage."""
    return rs.MultiLayer(
        [
            rs.Layer(1.0, 1.0),
            rs.Layer(2.0, 0.6),
            rs.Layer(1.0, 1.0),
        ]
    )


# ── Basic sanity checks ───────────────────────────────────────────────────────


def test_qnm_neff_returns_tuple_or_none(leaky_slab_thin):
    """complex_neff with Outgoing BCs must return a 2-tuple or None, never raise."""
    result = leaky_slab_thin.complex_neff(OMEGA)
    assert result is None or (isinstance(result, tuple) and len(result) == 2)


def test_all_qnm_neff_returns_list(leaky_slab_thin):
    """all_complex_neff with Outgoing BCs must return a list (possibly empty) of 2-tuples."""
    modes = leaky_slab_thin.all_complex_neff(OMEGA)
    assert isinstance(modes, list)
    for item in modes:
        assert isinstance(item, tuple) and len(item) == 2


def test_qnm_neff_no_mode_out_of_range(leaky_slab_thin):
    """Requesting a mode index far beyond the found count should return None."""
    result = leaky_slab_thin.complex_neff(OMEGA, mode=999)
    assert result is None


def test_qnm_neff_empty_search_box(leaky_slab_thin):
    """A search rectangle that contains no QNM poles should return None."""
    # Real part 0.1–0.3 is well below all material indices — no modes there.
    result = leaky_slab_thin.complex_neff(
        OMEGA,
        re_range=(0.1, 0.3),
        im_range=(-0.5, -1e-3),
    )
    assert result is None


# ── Sign of Im(neff) ─────────────────────────────────────────────────────────


def test_qnm_im_negative_thin_gap(leaky_slab_thin):
    """All QNMs found for the thin-gap leaky slab must have Im(neff) < 0."""
    # The QNM for this structure (gap=0.5 µm) sits at Im(neff) ≈ -0.003.
    # Use im_range=(-0.15, -1e-3): empirically reliable for this mode location
    # and avoids S-matrix anti-resonance instabilities that appear for deeper
    # or narrower search rectangles.
    modes = leaky_slab_thin.all_complex_neff(
        OMEGA,
        im_range=(-0.15, -1e-3),
    )
    assert len(modes) > 0, "Expected at least one QNM for thin-gap leaky slab"
    for re, im in modes:
        assert im < 0.0, f"QNM Im(neff) must be < 0, got {im:.6e}"


def test_qnm_im_negative_for_both_polarizations(leaky_slab_thin):
    """Im(neff) < 0 must hold for TE and TM QNMs alike."""
    for pol in (rs.Polarization.TE, rs.Polarization.TM):
        modes = leaky_slab_thin.all_complex_neff(
            OMEGA, polarization=pol, im_range=(-0.15, -1e-3)
        )
        for re, im in modes:
            assert im < 0.0, f"QNM Im(neff) must be < 0 for {pol}, got {im:.6e}"


def test_all_qnm_neff_symmetric_slab_all_negative_im(symmetric_slab):
    """Even for a symmetric slab, any found QNM must have Im(neff) ≤ 0."""
    modes = symmetric_slab.all_complex_neff(
        OMEGA,
        im_range=(-0.5, -1e-3),
        left_bc=rs.BoundaryCondition.Outgoing,
        right_bc=rs.BoundaryCondition.Outgoing,
    )
    for re, im in modes:
        assert im <= 0.0, f"QNM Im(neff) must be ≤ 0 for symmetric slab, got {im:.6e}"


# ── Physical range of Re(neff) ────────────────────────────────────────────────


def test_qnm_re_neff_in_physical_range(leaky_slab_thin):
    """Re(neff) should lie between the lowest and highest material index."""
    result = leaky_slab_thin.complex_neff(OMEGA, im_range=(-0.15, -1e-3))
    if result is None:
        pytest.skip("No QNM found — skipping range check")
    re, im = result
    # Core n=2.0 is the highest; air n=1.0 is the lowest real index.
    assert 0.9 < re < 2.2, f"Re(neff)={re:.4f} is outside physical range (0.9, 2.2)"


# ── Monotonic leakage with gap thickness ─────────────────────────────────────


def test_qnm_leakage_increases_as_gap_shrinks():
    """
    |Im(neff)| should increase (Im becomes more negative) as the gap shrinks.

    Gap thicknesses are chosen so that all three modes lie comfortably inside
    the reliable search window im_range=(-0.15, -1e-3):
      t=0.5 µm  →  Im(neff) ≈ -0.003
      t=0.3 µm  →  Im(neff) ≈ -0.021
      t=0.2 µm  →  Im(neff) ≈ -0.059
    """
    gaps = [0.5, 0.3, 0.2]
    im_values = {}

    for t in gaps:
        ml = make_leaky_slab(t)
        result = ml.complex_neff(OMEGA, im_range=(-0.15, -1e-3))
        if result is not None:
            re, im = result
            assert im < 0.0, f"Im(neff) must be < 0 for gap t={t}, got {im:.6e}"
            im_values[t] = im

    assert len(im_values) >= 2, (
        "At least two gap thicknesses must produce a QNM for the monotonicity test"
    )

    # Verify that Im decreases (becomes more negative) as gap shrinks.
    sorted_gaps = sorted(im_values.keys(), reverse=True)  # large → small
    for i in range(len(sorted_gaps) - 1):
        t_big = sorted_gaps[i]
        t_small = sorted_gaps[i + 1]
        assert im_values[t_small] < im_values[t_big], (
            f"Expected more leakage for smaller gap: "
            f"Im(t={t_small})={im_values[t_small]:.4e} should be < "
            f"Im(t={t_big})={im_values[t_big]:.4e}"
        )


# ── Consistency: QNM Re(neff) tracks the guided-mode Re(neff) ────────────────


def test_qnm_re_neff_tracks_guided_mode():
    """
    For a moderate gap, Re(neff_qnm) should be close to the real-axis neff
    of the guided mode of the isolated core.

    We use t=0.5 µm (the thin-gap slab) because its QNM sits at Im≈-0.003,
    well within the reliable search range.  The isolated core (air/n=2/air)
    has a guided mode whose Re(neff) should be within 0.3 of the QNM Re(neff).
    """
    t = 0.5
    ml = make_leaky_slab(t)

    # Isolated-core guided-mode neff for reference.
    isolated = rs.MultiLayer(
        [rs.Layer(1.0, 1.0), rs.Layer(2.0, 0.6), rs.Layer(1.0, 1.0)]
    )
    neff_guided = isolated.neff(OMEGA, rs.Polarization.TE)
    if neff_guided is None:
        pytest.skip("No guided mode found for isolated core — skipping")

    result = ml.complex_neff(OMEGA, im_range=(-0.15, -1e-3))
    if result is None:
        pytest.skip("No QNM found — skipping consistency check")

    re_qnm, im_qnm = result
    # Re(neff_qnm) should be reasonably close to the isolated-core guided-mode value.
    assert abs(re_qnm - neff_guided) < 0.3, (
        f"Re(neff_qnm)={re_qnm:.4f} should be near neff_guided={neff_guided:.4f}"
    )
    assert im_qnm < 0.0, f"Im(neff_qnm) must be < 0, got {im_qnm:.4e}"


# ── API consistency: mode ordering ───────────────────────────────────────────


def test_all_qnm_sorted_by_descending_re(leaky_slab_thin):
    """all_complex_neff must return QNMs sorted by descending Re(neff)."""
    modes = leaky_slab_thin.all_complex_neff(OMEGA, im_range=(-0.15, -1e-3))
    if len(modes) < 2:
        pytest.skip("Need at least 2 QNMs to test ordering")
    re_vals = [re for re, _ in modes]
    assert re_vals == sorted(re_vals, reverse=True), (
        f"Modes not sorted by descending Re(neff): {re_vals}"
    )


def test_qnm_neff_mode_0_matches_all_qnm_neff_first(leaky_slab_thin):
    """complex_neff(mode=0) should return the same result as all_complex_neff[0]."""
    all_modes = leaky_slab_thin.all_complex_neff(OMEGA, im_range=(-0.15, -1e-3))
    single = leaky_slab_thin.complex_neff(OMEGA, mode=0, im_range=(-0.15, -1e-3))
    if not all_modes:
        assert single is None
    else:
        assert single is not None
        assert abs(single[0] - all_modes[0][0]) < 1e-8
        assert abs(single[1] - all_modes[0][1]) < 1e-8
