Syntheseus currently supports 11 single-step models: RetroChimera (which we recommend as default), its two submodels, and 8 additional external models.

For convenience, for each model we include a default checkpoint (trained on Pistachio in the case of RetroChimera, and on USPTO-50K otherwise).
If no checkpoint directory is provided during model loading, `syntheseus` will automatically download a default checkpoint and cache it on disk for future use.
The default path for the cache is `$HOME/.cache/torch/syntheseus`, but it can be overriden by setting the `SYNTHESEUS_CACHE_DIR` environment variable.
See tables below for the links to the default checkpoints.

#### Recommended models (trained on Pistachio)

| Model checkpoint link                                          | Source |
|----------------------------------------------------------------|--------|
| [RetroChimera](https://figshare.com/ndownloader/files/59468882) | trained by us |
| [RetroChimeraEdit](https://figshare.com/ndownloader/files/61362964) | trained by us |
| [RetroChimeraDeNovo](https://figshare.com/ndownloader/files/61360657) | trained by us |

#### Other models (trained on USPTO-50K)

| Model checkpoint link                                          | Source |
|----------------------------------------------------------------|--------|
| [Chemformer](https://figshare.com/ndownloader/files/42009888)  | finetuned by us starting from checkpoint released by authors |
| [GLN](https://figshare.com/ndownloader/files/45882867)         | released by authors |
| [Graph2Edits](https://figshare.com/ndownloader/files/44194301) | released by authors |
| [LocalRetro](https://figshare.com/ndownloader/files/42287319)  | trained by us |
| [MEGAN](https://figshare.com/ndownloader/files/42012732)       | trained by us |
| [MHNreact](https://figshare.com/ndownloader/files/42012777)    | trained by us |
| [RetroKNN](https://figshare.com/ndownloader/files/45662430)    | trained by us |
| [RootAligned](https://figshare.com/ndownloader/files/42012792) | released by authors |

??? note "Choice of dataset"

    The USPTO-50K dataset is well-established but relatively small. Advanced users may prefer to either use our Pistachio-trained models, or retrain any model class of interest on their own data. To do that, please follow the instructions in the original model repositories.

#### Forward models

In `syntheseus/cli/eval_single_step.py`, a forward model can be used for computing back-translation (round-trip) accuracy.

| Model checkpoint link                                                   | Training data |
|-------------------------------------------------------------------------|---------------|
| [ForwardChimeraDeNovo](https://figshare.com/ndownloader/files/66654872) | Pistachio     |
| [Chemformer](https://figshare.com/ndownloader/files/42012708)           | USPTO-50K     |

??? info "Licenses"
    All checkpoints were produced in a way that involved external model repositories, hence may be affected by the exact license each model was released with.
    For more details about a particular model see the top of the corresponding model wrapper file in `reaction_prediction/inference/`.

## Sharing batched inference

`InferenceBroker` collects compatible requests from independent callers and runs the
backend on a single worker. Use one `BrokeredBackwardReactionModel` per search to keep
caches, call counts, and resets independent:

```python
from concurrent.futures import ThreadPoolExecutor

from syntheseus import Molecule
from syntheseus.reaction_prediction.inference.toy_models import LinearMoleculesToyModel
from syntheseus.reaction_prediction.utils.batching import (
    BrokeredBackwardReactionModel,
    InferenceBroker,
)

backend = LinearMoleculesToyModel(use_cache=False)
with InferenceBroker(backend, batch_size=8, batch_wait_s=0.01, max_queue_size=32) as broker:
    models = [BrokeredBackwardReactionModel(broker, use_cache=True) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        predictions = list(
            executor.map(lambda model, smiles: model([Molecule(smiles)]), models, ["COCS", "CC"])
        )
```

`BrokeredForwardReactionModel` provides the same isolation for forward models. Keep
filter wrappers per search to retain independent acceptance statistics.

Calls block until predictions are ready. `num_results=None` uses the backend's default
result count. Each caller receives independent prediction objects.

If a call fails during admission, its not-yet-running requests are cancelled; running
inference is left to finish. Normal context exit or `close()` finishes other accepted requests. Exiting the context with an
exception, or calling `close(cancel_pending=True)`, aborts unfinished requests with
`CancelledError`. Running inference cannot be interrupted. Inference failures reach
waiting callers and reject later calls.

When shutting down multiple brokers, call `close(cancel_pending=True, wait=False)` on
all of them before waiting with `close()`.
