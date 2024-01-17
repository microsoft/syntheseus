import logging
import multiprocessing
import os
import random
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import fields, is_dataclass
from itertools import islice
from os import devnull
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule


def set_random_seed(seed: int = 0) -> None:
    """Set random seed for `Python`, `torch` and `numpy`."""
    random.seed(seed)
    np.random.seed(seed)

    # If `torch` is installed set its seed as well.
    try:
        import torch

        torch.manual_seed(seed)
    except ModuleNotFoundError:
        pass


@contextmanager
def suppress_outputs():
    """Suppress messages written to both stdout and stderr."""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull), redirect_stdout(fnull):
            # Save the root logger handlers in order to restore them afterwards.
            root_handlers = list(logging.root.handlers)
            logging.disable(logging.CRITICAL)

            yield

            logging.root.handlers = root_handlers
            logging.disable(logging.NOTSET)


def dictify(data: Any) -> Any:
    # Need to ensure we make return objects fully serializable
    if isinstance(data, (int, float, str)) or data is None:
        return data
    elif isinstance(data, Molecule):
        return {"smiles": data.smiles}
    elif isinstance(data, (List, tuple, Bag)):
        # Captures lists of `Prediction`s
        return [dictify(x) for x in data]
    elif isinstance(data, dict):
        return {k: dictify(v) for k, v in data.items()}
    elif is_dataclass(data):
        result = {}
        for f in fields(data):
            value = getattr(data, f.name)
            result[f.name] = dictify(value)
        return result
    else:
        raise TypeError(f"Type {type(data)} cannot be handled by `dictify`")


def asdict_extended(data) -> Dict[str, Any]:
    """Convert a dataclass containing various reaction-related objects into a dict."""
    if not is_dataclass(data):
        raise TypeError(f"asdict_extended only for use on dataclasses, input is type {type(data)}")

    return dictify(data)


def undictify_bag_of_molecules(data: List[Dict[str, str]]) -> Bag[Molecule]:
    """Recovers a bag of molecules serialized with `dictify`."""
    return Bag(Molecule(d["smiles"]) for d in data)


def parallelize(
    fn,
    inputs: Iterable,
    num_processes: int = 0,
    chunksize: int = 32,
    num_chunks_per_process_per_segment: Optional[int] = 64,
) -> Iterator:
    """Parallelize an appliation of an arbitrary function using a pool of processes."""
    if num_processes == 0:
        yield from map(fn, inputs)
    else:
        # Needed for the chunking code to work on repeatable iterables e.g. lists.
        inputs = iter(inputs)

        with multiprocessing.Pool(num_processes) as pool:
            if num_chunks_per_process_per_segment is None:
                yield from pool.imap(fn, inputs, chunksize=chunksize)
            else:
                # A new segment will only be started if the previous one was consumed; this avoids doing
                # all the work upfront and storing it in memory if the consumer of the output is slow.
                segmentsize = num_chunks_per_process_per_segment * num_processes * chunksize

                non_empty = True
                while non_empty:
                    non_empty = False

                    # Call `imap` segment-by-segment to make sure the consumer of the output keeps up.
                    for result in pool.imap(fn, islice(inputs, segmentsize), chunksize=chunksize):
                        yield result
                        non_empty = True


def cpu_count(default: int = 8) -> int:
    """Return the number of CPUs, fallback to `default` if it cannot be determined."""
    return os.cpu_count() or default
