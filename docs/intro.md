# REMSOL Documentation

## Overview

REMSOL is a software package for the calculation of optical modes propagating in 1D multilayered structure. It is based on the transfer matrix method. Scattering matrix method is also partially supported.

The software is written in Rust for speed and provides a Python interface for ease of use.

## Installation

### PyPi

Publishing the package on PyPi is planned, but not yet available. For now, you will have to build the package from source.

### Building from source

To build from source, you need to have Rust installed on your system. You can install Rust by following the instructions on the [Rust website](https://www.rust-lang.org/tools/install).

It also reccomended to install the [UV](https://docs.astral.sh/uv/getting-started/installation/) python packge for easily managing the build process and its dependencies.

Once both are installed, you can clone the repository and build the package using uv with:

```bash
uv build
```

this will create a `.tar.gz` and a `.whl` file in the `dist` directory.

You can then use `pip install <name-of-your-whl-file>.whl` to install the package in yout main python or a dedicated virtual environment.

```{note}
Alternatively, you can use uv to automatically create a python virtuan environment with the package already installed. This is done by running `uv sync` in the root of the repository.

This will create a virtual environment in the `.venv` directory and install the package in it.

At this point, you can use the environment (after activation) to direcly run python scripts or jupyter notebooks.

Alternatively, you can use the `uv run` command to run a python script in the virtual environment without activating it.
```

## Contributing

Contributions are welcome. Plase feel free to open an issue or a pull request on the [GitHub repository](https://github.com/mpasson/REMSOL).
