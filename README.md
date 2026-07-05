[![PyPI version](https://img.shields.io/pypi/v/remsol)](https://pypi.org/project/remsol/)
[![Crates.io version](https://img.shields.io/crates/v/remsol)](https://crates.io/crates/remsol)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# REMSOL

**REMSOL** (Rust-based Electromagnetic Multi-layer Solver) is a high-performance solver
for guided electromagnetic modes in 1D planar multilayer (slab waveguide) structures.
The core engine is written in Rust for speed and exposed to Python via PyO3 bindings,
built with [Maturin](https://github.com/PyO3/maturin).

---

## Capabilities

- **Real and complex mode solving** - computes guided, lossy, leaky, and quasi-normal
  effective indices at a free-space wavenumber `k0 = 2*pi/lambda`.
- **Exact two-layer interfaces** - a literal dielectric-metal interface uses the
  branch-aware unsquared TM eigencondition. Larger stacks use the S-matrix
  determinant and argument-principle search.
- **Complex material indices** - each layer accepts a real or complex refractive
  index and a thickness in um.
- **Both polarizations** - supports TE and TM modes. In the nonmagnetic material
  model, an isolated two-medium surface mode is TM.
- **Field profiles** - reconstructs all six field components. MaxField
  normalization is the default; unit-power normalization is available for
  lossless guided modes.
- **Two numerical backends** - TMM supports index finding and field reconstruction;
  SMM supports index finding. Complex searches always use SMM for stability.
- **Boundary conditions** - semi-infinite, PEC, and outgoing-wave boundaries are
  available for their supported solver paths.

### Known limitations

- Geometry is strictly **1D planar**; no 2D or 3D structures.
- The SMM backend does **not** support field reconstruction.
- Complex-plane searches do **not** support PEC walls; PEC is supported by the
  real-axis TMM solver.
- `index()` returns only the real part of the material index for plotting.

---

## Python package

### Installation

The package is available on PyPI:

```bash
pip install remsol
```

To build from source, see the
[documentation](https://mpasson.github.io/REMSOL/intro.html#building-from-source).

### Quick example

```python
import math
import remsol as rs

# 1D slab waveguide: air | Si (600 nm) | air
k0 = 2 * math.pi / 1.55   # free-space wavenumber at 1550 nm (rad/µm)
layers = [
    rs.Layer(1.0, 0.0),    # left cladding  (semi-infinite, thickness ignored)
    rs.Layer(3.48, 0.6),   # waveguide core (600 nm Si)
    rs.Layer(1.0, 0.0),    # right cladding (semi-infinite, thickness ignored)
]
ml = rs.MultiLayer(layers)

# Effective index of the fundamental TE mode
neff = ml.neff(k0, rs.Polarization.TE, mode=0)
print(f"neff = {neff:.6f}")

# Full vectorial field profile
field = ml.field(k0, rs.Polarization.TE, mode=0)

# Refractive index profile
index = ml.index()
```

For more examples, see the
[documentation](https://mpasson.github.io/REMSOL/examples/examples.html).

---

## Rust crate

The Rust crate is available on [crates.io](https://crates.io/crates/remsol).
API documentation is on [docs.rs](https://docs.rs/remsol/latest/remsol/index.html).

---

## Contributing

Contributions are welcome. Feel free to open an issue or a pull request on
[GitHub](https://github.com/mpasson/REMSOL).