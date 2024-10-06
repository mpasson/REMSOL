[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MIT](https://img.shields.io/github/license/gdsfactory/gdsfactory)](https://choosealicense.com/licenses/mit/)

# REMSOL

## Overview

REMSOL (Rust-based Electromagnetic Multi-layer Solver) is a software package for finding electromagnetic modes in multi-layered media. It is written in Rust and implements both the scattering and transfer matrix method.

## Roadmap

- [x] Base scattering matrix method solver implementation
- [x] Base transfer matrix method solver implementation
- [x] Field calculation using transfer matrix method
- [ ] Field calculation using scattering matrix method
- [ ] Python bindings
  - [x] bare-minimum classes and methods (Polarization, Layer, MultiLayer)
  - [ ] backends methods
  - [x] field calculation methods
