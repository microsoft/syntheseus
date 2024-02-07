from __future__ import annotations

import csv
import json
import logging
from abc import abstractmethod
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Dict, Generic, Iterable, List, Optional, Type, TypeVar, Union

from more_itertools import ilen

from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample
from syntheseus.reaction_prediction.utils.misc import asdict_extended, parallelize

logger = logging.getLogger(__file__)

SampleType = TypeVar("SampleType", bound=ReactionSample)


CSV_REACTION_SMILES_COLUMN_NAME = "reactants>reagents>production"


class DataFold(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class DataFormat(Enum):
    JSONL = "jsonl"
    CSV = "csv"
    SMILES = "smi"


class ReactionDataset(Generic[SampleType]):
    """Dataset holding raw reactions split into folds."""

    @abstractmethod
    def __getitem__(self, fold: DataFold) -> Iterable[SampleType]:
        pass

    @abstractmethod
    def get_num_samples(self, fold: DataFold) -> int:
        pass


class DiskReactionDataset(ReactionDataset[SampleType]):
    def __init__(
        self,
        data_dir: Union[str, Path],
        sample_cls: Type[SampleType],
        num_processes: int = 0,
        data_format: Optional[DataFormat] = None,
    ):
        self._data_dir = Path(data_dir)
        self._sample_cls = sample_cls
        self._num_processes = num_processes

        paths = list(self._data_dir.iterdir())

        if data_format is None:
            logger.info(f"Detecting data format from files: {[path.name for path in paths]}")
            formats_to_try = list(DataFormat)
        else:
            formats_to_try = [data_format]

        matches = {
            format: DiskReactionDataset.match_paths_to_folds(format=format, paths=paths)
            for format in formats_to_try
        }
        matches = {key: values for key, values in matches.items() if values}

        if data_format is None:
            if len(matches) != 1:
                raise ValueError(
                    f"Format detection failed (formats matching the file list: {[f.name for f in matches]})"
                )
        elif not matches:
            raise ValueError(
                f"No files matching *{{train, val, test}}.{data_format.value} were found"
            )

        [(self._data_format, self._fold_to_path)] = matches.items()

        if data_format is None:
            logger.info(f"Detected format: {self._data_format.name}")

        logger.info(f"Loading data from files {self._fold_to_path}")
        self._num_samples: Dict[DataFold, int] = {}

    def _get_lines(self, fold: DataFold) -> Iterable[str]:
        if fold not in self._fold_to_path:
            return []
        else:
            with open(self._fold_to_path[fold]) as f:
                if self._data_format == DataFormat.CSV:
                    for row in csv.DictReader(f):
                        if CSV_REACTION_SMILES_COLUMN_NAME not in row:
                            raise ValueError(
                                f"No {CSV_REACTION_SMILES_COLUMN_NAME} column found in the CSV data file"
                            )
                        yield row[CSV_REACTION_SMILES_COLUMN_NAME]
                else:
                    for line in f:
                        yield line.rstrip()

    def __getitem__(self, fold: DataFold) -> Iterable[SampleType]:
        if self._data_format == DataFormat.JSONL:
            parse_fn = partial(DiskReactionDataset.sample_from_json, sample_cls=self._sample_cls)
        else:
            parse_fn = partial(self._sample_cls.from_reaction_smiles_strict, mapped=True)

        yield from parallelize(parse_fn, self._get_lines(fold), num_processes=self._num_processes)

    def get_num_samples(self, fold: DataFold) -> int:
        if fold not in self._num_samples:
            self._num_samples[fold] = ilen(self._get_lines(fold))

        return self._num_samples[fold]

    @staticmethod
    def match_paths_to_folds(format: DataFormat, paths: List[Path]) -> Dict[DataFold, Path]:
        fold_to_path: Dict[DataFold, Path] = {}
        for fold in DataFold:
            suffix = DiskReactionDataset.get_filename_suffix(format, fold)
            matching_paths = [path for path in paths if path.name.endswith(suffix)]

            if len(matching_paths) > 1:
                raise ValueError(
                    f"Found more than one {format.value} file for fold {fold.name}: {matching_paths}"
                )

            if matching_paths:
                [path] = matching_paths
                fold_to_path[fold] = path

        return fold_to_path

    @staticmethod
    def get_filename_suffix(format: DataFormat, fold: DataFold) -> str:
        return f"{fold.value}.{format.value}"

    @staticmethod
    def sample_from_json(data: str, sample_cls: Type[SampleType]) -> SampleType:
        return sample_cls.from_dict(json.loads(data))

    @staticmethod
    def save_samples_to_file(
        data_dir: Union[str, Path], fold: DataFold, samples: Iterable[SampleType]
    ) -> None:
        filename = DiskReactionDataset.get_filename_suffix(format=DataFormat.JSONL, fold=fold)

        with open(Path(data_dir) / filename, "wt") as f:
            for sample in samples:
                f.write(json.dumps(asdict_extended(sample)) + "\n")
