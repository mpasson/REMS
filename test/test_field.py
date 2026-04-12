import numpy as np
import pytest

import remsol
from remsol import Normalization
from remsol import Polarization as pol

slab = remsol.MultiLayer(
    [
        remsol.Layer(1, 1),
        remsol.Layer(2, 0.6),
        remsol.Layer(1, 1),
    ]
)

omega = 2.0 * np.pi / 1.55


def test_max_field_normalization():
    """Default normalization: max total |E| == 1."""
    field = slab.field(omega, pol.TE, 0)
    ex = np.array(field.Ex)
    ey = np.array(field.Ey)
    ez = np.array(field.Ez)
    max_e = np.max(np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2))
    assert max_e == pytest.approx(1.0, abs=1e-6)


def test_power_normalization():
    """Power normalization: integrated Poynting z-component == 1."""
    slab_power = remsol.MultiLayer(
        [remsol.Layer(1, 1), remsol.Layer(2, 0.6), remsol.Layer(1, 1)]
    )
    slab_power.set_normalization(Normalization.Power)
    field = slab_power.field(omega, pol.TE, 0)
    ex = np.array(field.Ex)
    hy = np.array(field.Hy)
    ey = np.array(field.Ey)
    hx = np.array(field.Hx)
    poynting = ex * np.conj(hy) - ey * np.conj(hx)
    power = np.trapz(poynting, field.x)
    assert abs(power) == pytest.approx(1.0, abs=1e-6)


def test_complex_field_max_field():
    """complex_field with a lossy core: max total |E| == 1 (default MaxField norm)."""
    lossy_slab = remsol.MultiLayer(
        [
            remsol.Layer(1.0, 1.0),
            remsol.Layer(2.0 - 0.01j, 0.6),
            remsol.Layer(1.0, 1.0),
        ]
    )
    field = lossy_slab.complex_field(omega, pol.TE, mode=0)
    ex = np.array(field.Ex)
    ey = np.array(field.Ey)
    ez = np.array(field.Ez)
    max_e = np.max(np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2))
    assert max_e == pytest.approx(1.0, abs=1e-6)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    field = slab.field(2.0 * np.pi / 1.55, pol.TE, 0)
    for component in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
        comp = getattr(field, component)
        plt.plot(field.x, np.real(comp) + np.imag(comp), label=component)
    plt.legend()
    plt.show()
