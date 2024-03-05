# Single-step Evaluation

## Usage

```
syntheseus eval-single-step \
    data_dir=[DATA_DIR] \
    fold=[TRAIN, VAL or TEST] \
    model_class=[MODEL_CLASS] \
    model_dir=[MODEL_DIR]
```

The `eval-single-step` command accepts further arguments to customize the evaluation; see `BaseEvalConfig` in `cli/eval_single_step.py` for the complete list.

The code will scan `data_dir` looking for files matching `*{train, val, test}.{jsonl, csv, smi}` and select the right data format based on the file extension. An error will be raised in case of ambiguity. Only the fold that was selected for evaluation has to be present.

## Data format

The single-step evaluation script supports reaction data in one of three formats.

### JSONL

Our internal format based on `*.jsonl` files in which each line is a JSON representation of a single reaction, for example:
```json
{"reactants": [{"smiles": "Cc1ccc(Br)cc1"}, {"smiles": "Cc1ccc(B(O)O)cc1"}], "products": [{"smiles": "Cc1ccc(-c2ccc(C)cc2)cc1"}]}
```
This format is designed to be flexible at the expense of taking more disk space. The JSON is parsed into a `ReactionSample` object, so it can include additional metadata such as template information, while the reactants and products can include other fields accepted by the `Molecule` object. The evaluation script will only use reactant and product SMILES to compute the metrics.

Unlike the other formats below, reactants and products in this format are assumed to be already stripped of atom mapping, which leads to slightly faster data loading as that avoids extra calls to `rdkit`.

### CSV

This format is based on `*.csv` files and is commonly used to store raw USPTO data, e.g. as released by [Dai et al.](https://github.com/Hanjun-Dai/GLN):

```
id,class,reactants>reagents>production
ID,UNK,[cH:1]1[cH:2][c:3]([CH3:4])[cH:5][cH:6][c:7]1Br.B(O)(O)[c:8]1[cH:9][cH:10][c:11]([CH3:12])[cH:13][cH:14]1>>[cH:1]1[cH:2][c:3]([CH3:4])[cH:5][cH:6][c:7]1[c:8]2[cH:14][cH:13][c:11]([CH3:12])[cH:10][cH:9]2
```

The evaluation script will look for the `reactants>reagents>production` column to extract the reaction SMILES, which are stripped of atom mapping and canonicalized before being fed to the model.

### SMILES

The most compact format is to list reaction SMILES line-by-line in a `*.smi` file:

```
[cH:1]1[cH:2][c:3]([CH3:4])[cH:5][cH:6][c:7]1Br.B(O)(O)[c:8]1[cH:9][cH:10][c:11]([CH3:12])[cH:13][cH:14]1>>[cH:1]1[cH:2][c:3]([CH3:4])[cH:5][cH:6][c:7]1[c:8]2[cH:14][cH:13][c:11]([CH3:12])[cH:10][cH:9]2
```

The data will be handled in the same way as for the CSV format, i.e. it will only be fed into model after removing atom mapping and canonicalization.
