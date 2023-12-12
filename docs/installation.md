We support two installation modes:

- *core installation* allows you to build and benchmark your own models or search algorithms
- *full installation* also allows you to perform end-to-end search using the supported models

=== "Core installation"

    ```bash
    conda env create -f environment.yml
    conda activate syntheseus

    pip install -e .
    ```

=== "Full installation"

    ```bash
    conda env create -f environment_full.yml
    conda activate syntheseus-full

    pip install -e ".[all]"
    ```

Core installation includes only minimal dependencies (no ML libraries), while full installation includes all supported models and also dependencies for visualization/development.

Instructions above assume you already cloned the repository via

```bash
git clone https://github.com/microsoft/syntheseus.git
cd syntheseus
```

Note that `environment_full.yml` pins the CUDA version (to 11.3) for reproducibility.
If you want to use a different one, make sure to edit the environment file accordingly.

??? info "Setting up GLN"

    We also support GLN, but it requires a specialized environment and is thus not installed via `pip`.
    See [here](https://github.com/microsoft/syntheseus/blob/main/syntheseus/reaction_prediction/environment_gln/Dockerfile) for a Docker environment necessary for running GLN.

## Reducing the number of dependencies

To keep the environment smaller, you can replace the `all` option with a comma-separated subset of `{chemformer,local-retro,megan,mhn-react,retro-knn,root-aligned,viz,dev}` (`viz` and `dev` correspond to visualization and development dependencies, respectively).
For example, `pip install -e ".[local-retro,root-aligned]"` installs only LocalRetro and RootAligned.
If installing a subset of models, you can also delete the lines in `environment_full.yml` marked with names of models you do not wish to use.

If you only want to use a very specific part of `syntheseus`, you could also install it without dependencies:

```bash
pip install -e .  --no-dependencies
```

You then would need to manually install a subset of dependencies that are required for a particular functionality you want to access.
See `pyproject.toml` for a list of dependencies tied to the `search` and `reaction_prediction` subpackages.
