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

- **Mode solving** — computes the effective refractive indices (`neff`) of all guided
  modes supported by a given layer stack at a given free-space wavenumber `k₀ = 2π/λ`.
- **Both polarizations** — full support for TE (transverse electric) and TM (transverse
  magnetic) polarizations.
- **Field profiles** — reconstructs the complete vectorial field distribution
  (Ex, Ey, Ez, Hx, Hy, Hz) across the cross-section for any guided mode, normalized so
  that the integrated Poynting vector equals 1.
- **Index profile** — samples the refractive index profile of the stack for plotting
  and inspection.
- **Two numerical backends**:
  - *Transfer Matrix Method (TMM)* — full-featured; supports both `neff` and `field`.
  - *Scattering Matrix Method (SMM)* — supports `neff` only; useful as a cross-check.
- **Boundary conditions** — each side of the stack can be set to:
  - *Semi-infinite cladding* (default) — field decays evanescently outward.
  - *Perfect Electric Conductor (PEC)* — tangential electric field forced to zero at
    the wall; useful for modelling symmetric structures with half-domain tricks.
- **Arbitrary stacks** — any number of layers, each with an independent real refractive
  index and thickness (in µm).

### Known limitations

- Refractive indices are **real-valued only** (lossless, non-gain media).
- Only **guided modes** (real `neff`) are computed; leaky modes are not supported.
- Geometry is strictly **1D planar**; no 2D or 3D structures.
- The SMM backend does **not** support field calculation.

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