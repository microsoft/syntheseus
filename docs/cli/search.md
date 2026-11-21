# Running Search

## Usage

```
syntheseus search \
    search_targets_file=[SMILES_FILE_WITH_SEARCH_TARGETS] \
    inventory_file=[MOL_INVENTORY_FILE] \
    model_class=[MODEL_CLASS] \
    model_dir=[MODEL_DIR] \
    time_limit_s=[NUMBER_OF_SECONDS_PER_TARGET]
```

The search targets are expected to be a plain SMILES file, with one molecule per line. The purchasable molecules are a `MolInventory` file, prebuilt once from a SMILES file:

```python
from syntheseus.search.mol_inventory import IdentityStandardiser, MolInventoryBuilder

builder = MolInventoryBuilder(IdentityStandardiser())
inventory = builder.build_from_file(
    "inventory.smi", name="my-inventory", version="1", file_format="smi"
)
builder.save(inventory, "inventory.molinv")
```

Passing a plain SMILES file with `inventory_smiles_file` instead is deprecated.

The `search` command accepts further arguments to configure the search algorithm; see `SearchConfig` in `cli/search.py` for the complete list.

## Concurrent targets and batched inference

Search remains serial by default (`max_active_searches=1`). To run independent target
searches concurrently while batching their inference requests:

```sh
syntheseus search \
    search_targets_file=targets.smi \
    inventory_file=inventory.molinv \
    model_class=RetroChimera \
    max_active_searches=32 \
    inference_batch_size=8 \
    inference_batch_wait_s=0.01  # 10 ms
```

`max_active_searches` limits the number of targets searched at once. `inference_batch_size`
limits the number of molecules predicted together, while `inference_batch_wait_s` sets the
maximum wait for other requests to fill a batch. Higher concurrency or larger batches may
improve throughput, but can also increase memory use and per-target latency.

Per-target time limits include waiting for inference. Concurrent execution can produce
different routes or solve different targets than serial execution, especially with time-limited
searches or stochastic models. Output layout and resume behavior remain unchanged. If a target
fails, remaining work is cancelled; running model calls must finish before the command exits.

!!! info
    When using one of the natively supported single-step models you can omit `model_dir`, which will cause `syntheseus` to use a default checkpoint (see [here](../single_step.md) for details).

## Configuring the search algorithm

You can set the search algorithm explicitly using the `search_algorithm` argument to `retro_star` (default), `mcts` or `pdvn`.
For all of those algorithms you can vary hyperparameters such as the policy/value functions or MCTS bound type/constant.

In practice however there may be no need to override any hyperparameters, especially if combining Retro\* or MCTS with one of the natively supported models, as for those `syntheseus` will automatically choose sensible hyperparameter defaults (listed in `cli/search_config.yml`).
[In our experience](https://arxiv.org/abs/2310.19796) both Retro* and MCTS show similar performance when tuned properly, but you may want to try both for your particular usecase and see which one works best empirically.
