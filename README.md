# DFT analysis utilities

Utilities for analysing results from ADF/AMS and ORCA quantum-chemistry
calculations. The import package is `dft_utils`. Originally written for my PhD research.

## Installation

Core numerical utilities:

```bash
python -m pip install .
```

For spectrum/HDF5 workflows, install the analysis extras:

```bash
python -m pip install '.[analysis,advanced,visualization]'
```

PySCF-backed readers are available with the `pyscf` extra. ADF/AMS readers
also require the external PLAMS runtime supplied by AMS.

## Development

```bash
python -m pip install -e '.[analysis,advanced,visualization,dev]'
pytest
black --check .
isort --check-only .
ruff check . --select F821,E9
```

