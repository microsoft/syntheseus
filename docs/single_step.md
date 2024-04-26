Syntheseus currently supports 8 established single-step models.

For convenience, for each model we include a default checkpoint trained on USPTO-50K.
If no checkpoint directory is provided during model loading, `syntheseus` will automatically download a default checkpoint and cache it on disk for future use.
The default path for the cache is `$HOME/.cache/torch/syntheseus`, but it can be overriden by setting the `SYNTHESEUS_CACHE_DIR` environment variable.
See table below for the links to the default checkpoints.

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

??? note "More advanced datasets"

    The USPTO-50K dataset is well-established but relatively small. Advanced users may prefer to retrain their models of interest on a larger dataset, such as USPTO-FULL or Pistachio. To do that, please follow the instructions in the original model repositories.

In `reaction_prediction/cli/eval.py` a forward model can be used for computing back-translation (round-trip) accuracy.
See [here](https://figshare.com/ndownloader/files/42012708) for a Chemformer checkpoint finetuned for forward prediction on USPTO-50K. As for the backward direction, pretrained weights released by original authors were used as a starting point.

??? info "Licenses"
    All checkpoints were produced in a way that involved external model repositories, hence may be affected by the exact license each model was released with.
    For more details about a particular model see the top of the corresponding model wrapper file in `reaction_prediction/inference/`.
