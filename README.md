[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# REMSOL

## Overview

REMSOL (Rust-based Electromagnetic Multi-layer Solver) is a software package for finding electromagnetic modes in multi-layered media. It is written in Rust and implements both the scattering and transfer matrix method.

## Installation

For now, the package is not available on PyPI, so you will have to build it from source.

In orger to build the package, you will need:

- Rust (check [rustup](https://rustup.rs/)).
- UV ([docs.astral.sh](https://docs.astral.sh/uv/getting-started/installation/)).

after cloning the repository, build the package using uv with:

```bash
uv build
```

this will create a `.tar.gz` and a `.whl` file in the `dist` directory. You can install the package using pip:

```bash
pip install dist/<your-whl-file>.whl
```

and enjoy!

## Roadmap

- [x] Base scattering matrix method solver implementation
- [x] Base transfer matrix method solver implementation
- [x] Field calculation using transfer matrix method
- [ ] Field calculation using scattering matrix method
- [ ] Python bindings
  - [x] bare-minimum classes and methods (Polarization, Layer, MultiLayer)
  - [ ] backends methods
  - [x] field calculation methods

````

```

```

```

```
````
