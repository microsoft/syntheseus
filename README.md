<div align="center">
    <img src="https://github.com/microsoft/syntheseus/assets/61470923/f01a9939-61fa-4461-a124-c13eddcdd75a" height="50px">
    <h3><i>Navigating the labyrinth of synthesis planning</i></h3>
</div>

---

[![CI](https://github.com/microsoft/syntheseus/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/microsoft/syntheseus/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![code style](https://img.shields.io/badge/code%20style-black-202020.svg)](https://github.com/ambv/black)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/microsoft/syntheseus/blob/main/LICENSE)

Syntheseus is a package for end-to-end retrosynthetic planning.
- ⚒️ Combines search algorithms and reaction models in a standardized way
- 🧭 Includes implementations of common search algorithms
- 🧪 Includes wrappers for state-of-the-art reaction models
- ⚙️ Exposes a simple API to plug in custom models and algorithms
- 📈 Can be used to benchmark components of a retrosynthesis pipeline

To learn about `syntheseus`'s features and API visit [microsoft.github.io/syntheseus](https://microsoft.github.io/syntheseus).

## Quick Start

To install `syntheseus` with all the extras, run

```bash
conda env create -f environment_full.yml
conda activate syntheseus-full

pip install -e ".[all]"
```

See [documentation](https://microsoft.github.io/syntheseus/installation) if you prefer a more lightweight installation that only includes the parts you actually need.

## Development

Syntheseus is currently under active development.
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
