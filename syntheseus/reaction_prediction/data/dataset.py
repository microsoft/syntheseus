import json
from abc import abstractmethod
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Dict, Generic, Iterable, Type, TypeVar, Union

from more_itertools import ilen

from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample
from syntheseus.reaction_prediction.utils.misc import asdict_extended, parallelize

SampleType = TypeVar("SampleType", bound=ReactionSample)


class DataFold(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class ReactionDataset(Generic[SampleType]):
    """Dataset holding raw reactions split into folds."""

    @abstractmethod
    def __getitem__(self, fold: DataFold) -> Iterable[SampleType]:
        pass

    @abstractmethod
    def get_num_samples(self, fold: DataFold) -> int:
        pass

    @classmethod
    def get_data_path(cls, data_dir: Union[str, Path], fold: DataFold) -> Path:
        return Path(data_dir) / f"{fold.value}.jsonl"

    @classmethod
    def sample_from_json(cls, data: str, sample_cls: Type[SampleType]) -> SampleType:
        return sample_cls.from_dict(json.loads(data))

    @classmethod
    def save_samples_to_file(
        cls, data_dir: Union[str, Path], fold: DataFold, samples: Iterable[SampleType]
    ) -> None:
        with open(ReactionDataset.get_data_path(data_dir=data_dir, fold=fold), "wt") as f:
            for sample in samples:
                f.write(json.dumps(asdict_extended(sample)) + "\n")


class DiskReactionDataset(ReactionDataset[SampleType]):
    def __init__(
        self,
        data_dir: Union[str, Path],
        sample_cls: Type[SampleType],
        num_processes: int = 0,
    ):
        self._data_dir = data_dir
        self._sample_cls = sample_cls
        self._num_processes = num_processes

        self._num_samples: Dict[DataFold, int] = {}

    def _get_lines(self, fold: DataFold) -> Iterable[str]:
        data_path = ReactionDataset.get_data_path(data_dir=self._data_dir, fold=fold)

        if not data_path.exists():
            return []
        else:
            with open(data_path) as f:
                yield from f

    def __getitem__(self, fold: DataFold) -> Iterable[SampleType]:
        yield from parallelize(
            partial(ReactionDataset.sample_from_json, sample_cls=self._sample_cls),
            self._get_lines(fold),
            num_processes=self._num_processes,
        )

    def get_num_samples(self, fold: DataFold) -> int:
        if fold not in self._num_samples:
            self._num_samples[fold] = ilen(self._get_lines(fold))

        return self._num_samples[fold]
