# Running Search

## Usage

```
syntheseus search \
    search_targets_file=[SMILES_FILE_WITH_SEARCH_TARGETS] \
    inventory_smiles_file=[SMILES_FILE_WITH_PURCHASABLE_MOLECULES] \
    model_class=[MODEL_CLASS] \
    model_dir=[MODEL_DIR] \
    time_limit_s=[NUMBER_OF_SECONDS_PER_TARGET]
```

Both the search targets and the purchasable molecules inventory are expected to be plain SMILES files, with one molecule per line.

The `search` command accepts further arguments to configure the search algorithm; see `SearchConfig` in `cli/search.py` for the complete list.

!!! info
    When using one of the natively supported single-step models you can omit `model_dir`, which will cause `syntheseus` to use a default checkpoint trained on USPTO-50K (see [here](../single_step.md) for details).

## Configuring the search algorithm

You can set the search algorithm explicitly using the `search_algorithm` argument to `retro_star` (default), `mcts` or `pdvn`.
For all of those algorithms you can vary hyperparameters such as the policy/value functions or MCTS bound type/constant.

In practice however there may be no need to override any hyperparameters, especially if combining Retro\* or MCTS with one of the natively supported models, as for those `syntheseus` will automatically choose sensible hyperparameter defaults (listed in `cli/search_config.yml`).
[In our experience](https://arxiv.org/abs/2310.19796) both Retro* and MCTS show similar performance when tuned properly, but you may want to try both for your particular usecase and see which one works best empirically.
