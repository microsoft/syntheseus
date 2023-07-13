# Syntheseus

[![CI](https://github.com/microsoft/syntheseus/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/microsoft/syntheseus/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![code style](https://img.shields.io/badge/code%20style-black-202020.svg)](https://github.com/ambv/black)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Syntheseus is a package for retrosynthetic planning.
It contains implementations of common search algorithms
and a simple API to wrap custom reaction models and write
custom algorithms.
It is meant to allow for simple benchmarking of the components
of retrosynthesis algorithms.

## Installation

Currently `syntheseus` is not hosted on PyPI
(although this will likely change in the future).
To install, please run:

```bash
# Clone and cd into the repository.
git clone https://github.com/microsoft/syntheseus.git
cd syntheseus

# Create and activate a new conda environment (or use your own).
conda env create -f environment.yml
conda activate syntheseus

# Install into the current environment.
pip install -e .
```

Syntheseus contains two subpackages: `reaction_prediction`, which deals with benchmarking single-step reaction models, and `search`, which can use any single-step model to perform multi-step search.
Each is designed to have minimal dependencies, allowing it to run in a wide range of environments.
While specific components (single-step models, policies, or value functions) can make use of Deep Learning libraries, the core of `syntheseus` does not depend on any.

If you only want to use either of the two subpackages, you can limit the dependencies further by installing the dependencies separately and then running

```bash
pip install -e .  --no-dependencies
```

See `pyproject.toml` for a list of dependencies tied to each subpackage.

## Development

Syntheseus is currently under active development and does not have a fixed API
(but we will fix it very soon).
If you want to help us develop syntheseus please install and run `pre-commit`
checks before committing code.

We use `pytest` for testing. Please make sure tests pass on your branch before
submitting a PR (and try to maintain high test coverage).

```bash
python -m pytest --cov syntheseus/tests
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
