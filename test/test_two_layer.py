import numpy as np
import pytest

import remsol
from remsol import Polarization as pol


WAVELENGTH = 1.55
OMEGA = 2.0 * np.pi / WAVELENGTH
EPSILON_D = 2.1025 + 0.0j
EPSILON_M = -116.944 + 11.223j
N_D = complex(np.sqrt(EPSILON_D))
N_M = complex(np.sqrt(EPSILON_M))
ANALYTIC_NEFF = complex(
    np.sqrt(EPSILON_D * EPSILON_M / (EPSILON_D + EPSILON_M))
)

interface = remsol.MultiLayer(
    [
        remsol.Layer(N_D, 2.0),
        remsol.Layer(N_M, 2.0),
    ]
)
reversed_interface = remsol.MultiLayer(
    [
        remsol.Layer(N_M, 2.0),
        remsol.Layer(N_D, 2.0),
    ]
)


def test_automatic_two_layer_complex_neff_matches_analytic() -> None:
    result = interface.complex_neff(OMEGA, pol.TM)

    assert result is not None
    assert result[0] == pytest.approx(ANALYTIC_NEFF.real, abs=1e-10)
    assert result[1] == pytest.approx(ANALYTIC_NEFF.imag, abs=1e-10)
    all_results = interface.all_complex_neff(OMEGA, pol.TM)
    assert len(all_results) == 1
    assert all_results[0] == pytest.approx(
        (ANALYTIC_NEFF.real, ANALYTIC_NEFF.imag), abs=1e-10
    )


def test_two_layer_explicit_ranges_filter_the_exact_mode() -> None:
    assert interface.complex_neff(
        OMEGA,
        pol.TM,
        re_range=(1.4, 1.5),
        im_range=(0.0, 0.01),
    ) == pytest.approx((ANALYTIC_NEFF.real, ANALYTIC_NEFF.imag), abs=1e-10)
    assert interface.complex_neff(
        OMEGA, pol.TM, re_range=(1.0, 1.4)
    ) is None
    assert interface.complex_neff(
        OMEGA, pol.TM, im_range=(-0.01, 0.0)
    ) is None


def test_two_layer_order_keeps_positive_propagation_mode() -> None:
    result = reversed_interface.complex_neff(OMEGA, pol.TM)

    assert result == pytest.approx(
        (ANALYTIC_NEFF.real, ANALYTIC_NEFF.imag), abs=1e-10
    )


def test_two_layer_rejects_te_and_dielectric_interface() -> None:
    assert interface.complex_neff(OMEGA, pol.TE) is None
    assert interface.all_complex_neff(OMEGA, pol.TE) == []

    dielectric_interface = remsol.MultiLayer(
        [remsol.Layer(1.45, 1.0), remsol.Layer(1.7, 1.0)]
    )
    assert dielectric_interface.complex_neff(OMEGA, pol.TM) is None


def test_lossless_two_layer_mode_is_available_on_real_axis() -> None:
    epsilon_d = 2.25 + 0.0j
    epsilon_m = -10.0 + 0.0j
    expected = complex(
        np.sqrt(epsilon_d * epsilon_m / (epsilon_d + epsilon_m))
    ).real
    lossless = remsol.MultiLayer(
        [
            remsol.Layer(complex(np.sqrt(epsilon_d)), 1.0),
            remsol.Layer(complex(np.sqrt(epsilon_m)), 1.0),
        ]
    )

    assert lossless.neff(OMEGA, pol.TM) == pytest.approx(expected, abs=1e-10)
    assert lossless.all_neff(OMEGA, pol.TM) == pytest.approx([expected], abs=1e-10)


def _electric_magnitude(field: remsol.FieldData) -> np.ndarray:
    ex = np.asarray(field.Ex)
    ey = np.asarray(field.Ey)
    ez = np.asarray(field.Ez)
    return np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2)


def test_two_layer_complex_field_is_normalized_localized_and_tm() -> None:
    field = interface.complex_field(OMEGA, pol.TM)
    x = np.asarray(field.x)
    magnitude = _electric_magnitude(field)

    for component_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        component = np.asarray(getattr(field, component_name))
        assert np.all(np.isfinite(component))
    assert np.max(magnitude) == pytest.approx(1.0, abs=1e-10)
    assert magnitude[0] < 0.3
    assert magnitude[-1] < 0.3
    assert np.allclose(field.Ey, 0.0, atol=1e-14)
    assert np.allclose(field.Hx, 0.0, atol=1e-14)
    assert np.allclose(field.Hz, 0.0, atol=1e-14)

    right = int(np.flatnonzero(x >= 0.0)[0])
    left = right - 1
    hy = np.asarray(field.Hy)
    ez = np.asarray(field.Ez)
    assert hy[left] == pytest.approx(hy[right], rel=0.01, abs=1e-8)
    assert EPSILON_D * ez[left] == pytest.approx(
        EPSILON_M * ez[right], rel=0.01, abs=1e-8
    )


def test_reversing_layers_mirrors_two_layer_field() -> None:
    field = interface.complex_field(OMEGA, pol.TM)
    reversed_field = reversed_interface.complex_field(OMEGA, pol.TM)
    x = np.asarray(field.x)
    reversed_x = np.asarray(reversed_field.x)
    magnitude = _electric_magnitude(field)
    reversed_magnitude = _electric_magnitude(reversed_field)
    # Ex is discontinuous at the interface, so exclude its single grid sample.
    interior = (np.abs(x) < 1.9) & (np.abs(x) > 2.0 * interface.plot_step)
    mirrored = np.interp(-x[interior], reversed_x, reversed_magnitude)

    assert magnitude[interior] == pytest.approx(mirrored, rel=3e-3, abs=3e-3)
