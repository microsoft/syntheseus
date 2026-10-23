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

## Concurrent targets and batched inference

Search remains serial by default (`max_active_searches=1`). To run independent target
searches concurrently while batching their inference requests:

```sh
syntheseus search \
    search_targets_file=targets.smi \
    inventory_smiles_file=inventory.smi \
    model_class=RetroChimera \
    max_active_searches=32 \
    inference_batch_size=8 \
    inference_batch_wait_s=0.01
```

`max_active_searches` bounds the number of active searches. `inference_batch_size` bounds
each model batch, and `inference_batch_wait_s` limits how long a request waits for other
requests to form a batch. The runner loads one backward backend and, when forward filtering
is enabled, one forward backend. Each backend has one inference worker; per-target facades
retain independent caches and reaction-model call budgets. Filter acceptance statistics,
search algorithms, and node evaluators are also independent per target.

Concurrent execution retains stereo removal, inventory checks, per-target statistics and
graphs, route extraction, and the existing aggregate statistics. Plotting is serialized.
Completed targets are still skipped on resume; failed targets retain their lockfiles so
partial outputs are removed on the next run. A failure cancels the remaining searches and
pending inference, but running model calls must finish before shutdown.

Time limits include time spent waiting for inference. Wall-clock-limited searches and
stochastic models can produce different routes or solve different targets than serial
execution. Concurrent search is opt-in and does not guarantee serial-equivalent results
or schedule-independent model augmentation. Model-specific augmentation and process
tuning remain the responsibility of the model integration.

!!! info
    When using one of the natively supported single-step models you can omit `model_dir`, which will cause `syntheseus` to use a default checkpoint (see [here](../single_step.md) for details).

## Configuring the search algorithm

You can set the search algorithm explicitly using the `search_algorithm` argument to `retro_star` (default), `mcts` or `pdvn`.
For all of those algorithms you can vary hyperparameters such as the policy/value functions or MCTS bound type/constant.

In practice however there may be no need to override any hyperparameters, especially if combining Retro\* or MCTS with one of the natively supported models, as for those `syntheseus` will automatically choose sensible hyperparameter defaults (listed in `cli/search_config.yml`).
[In our experience](https://arxiv.org/abs/2310.19796) both Retro* and MCTS show similar performance when tuned properly, but you may want to try both for your particular usecase and see which one works best empirically.
