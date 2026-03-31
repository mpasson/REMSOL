# CHANGELOG


## v0.2.3 (2026-03-31)

### Bug Fixes

- Define PYTHON_VERSION as a single top-level env var in publish workflow
  ([`a09343f`](https://github.com/mpasson/REMSOL/commit/a09343f17808cc68d3a44eecd0aacf8dca2f65f1))

- Fixed python version for uv environment
  ([`e81a6fd`](https://github.com/mpasson/REMSOL/commit/e81a6fd96a6d8188e92ac960e1290a4ea23350d5))

- Pin maturin to python3.13 and replace setup-python with uv on windows/macos
  ([`850a51f`](https://github.com/mpasson/REMSOL/commit/850a51f1645f99f67e1551281e8cb0b05a1cd646))


## v0.2.2 (2026-03-31)

### Bug Fixes

- Set PYO3_USE_ABI3_FORWARD_COMPATIBILITY for Python 3.14 and bump all action versions to v6
  ([`f843bcf`](https://github.com/mpasson/REMSOL/commit/f843bcf73395e4afd00b2126d782d21b8f4fb364))

### Build System

- Add .secrets to .gitignore and document GitHub API access pattern in rules
  ([`7450e81`](https://github.com/mpasson/REMSOL/commit/7450e816d167dc3fc689c401332194906bfa8c7c))

### Documentation

- Expand README with capabilities summary, quick example, and formatting improvements
  ([`a9f8e30`](https://github.com/mpasson/REMSOL/commit/a9f8e3003650ea69b9d48adb2a5671b162a7e6ed))


## v0.2.1 (2026-03-31)

### Bug Fixes

- Bump setup-uv to v6 and require clean working tree before release
  ([`0a35868`](https://github.com/mpasson/REMSOL/commit/0a35868d5de7ad1b0081da6f47712835760b5831))


## v0.2.0 (2026-03-31)

### Bug Fixes

- Set default accuracy to 10 instead of 12
  ([`14a1768`](https://github.com/mpasson/REMSOL/commit/14a176819a0de5d8c0e67a06f941a13b1d437da4))

### Build System

- Added .zed folder to tracked files
  ([`ef8efff`](https://github.com/mpasson/REMSOL/commit/ef8effff7e367e6713d1c5f5b596beb631c8dd0a))

### Features

- Added PEC as possible booundary condition on either side
  ([`030684a`](https://github.com/mpasson/REMSOL/commit/030684afd0d2692875067ccd58b6bd65b0b4b476))


## v0.1.3 (2025-05-12)

### Build System

- Added maturin to development dependencies
  ([`5e58e2b`](https://github.com/mpasson/REMSOL/commit/5e58e2ba989e84374195934ad13eed6993352293))

- Updated rust dependencies
  ([`bc69660`](https://github.com/mpasson/REMSOL/commit/bc69660e463bd253d5151a3b8e6d8817e995bd1a))


## v0.1.2 (2025-05-12)

### Bug Fixes

- Improved zero finding strategy
  ([`bd80d8c`](https://github.com/mpasson/REMSOL/commit/bd80d8cc8a38747c8c7b62b5a2846ee645c99663))

### Build System

- Added semantic release to dev dependencies
  ([`eb94ee1`](https://github.com/mpasson/REMSOL/commit/eb94ee13b7f9610722d318521fd38021a0dd89f9))

- Added semantic-release setting to update version in Cargo.toml
  ([`cee740e`](https://github.com/mpasson/REMSOL/commit/cee740e37aefd70f17dc4bd164bdc27b4cc2fdd2))

- Updated uv.lock
  ([`998a9da`](https://github.com/mpasson/REMSOL/commit/998a9da4f1001ba5afaf4e2c4f59f2957234f0bc))

### Refactoring

- Removed clippy warnings from scattering_matrix.rs and transfer_matrix.rs
  ([`245145f`](https://github.com/mpasson/REMSOL/commit/245145fcf0bb6913cc8c0bf51967302a573e9088))

- Removed non used bin files
  ([`bc00450`](https://github.com/mpasson/REMSOL/commit/bc0045061e8b719db2d1df72efcd892d9940dc70))

- Solved clippy warning for remsol.rs and multilayer.rs
  ([`ac004af`](https://github.com/mpasson/REMSOL/commit/ac004af5667366a7ea978c2448673e432c9575bd))

### Testing

- Added benchmark for performance
  ([`2727e18`](https://github.com/mpasson/REMSOL/commit/2727e1898a8609ab521aa1ed9e89109d7c1e112a))


## v0.1.1 (2024-11-04)


## v0.1.0 (2024-10-13)


## v0.0.1 (2024-10-13)
