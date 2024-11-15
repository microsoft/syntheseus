import sys
from collections.abc import Collection
from typing import Generic, Iterable, Iterator, TypeVar

# mypy-compatible import recommended here:
# https://mypy.readthedocs.io/en/stable/runtime_troubles.html#using-new-additions-to-the-typing-module
if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


class Comparable(Protocol):
    def __lt__(self, __other) -> bool:
        ...


ElementT = TypeVar("ElementT", bound=Comparable)


class Bag(Collection, Generic[ElementT]):
    """Class representing a frozen multi-set (i.e. set where elements are allowed to repeat).

    The bag elements are internally stored as a sorted tuple for simplicity, thus lookup time is
    linear in terms of the size of the bag.
    """

    def __init__(self, values: Iterable[ElementT]) -> None:
        self._values = tuple(sorted(values))

    def __iter__(self) -> Iterator:
        return iter(self._values)

    def __contains__(self, element) -> bool:
        return element in self._values

    def __eq__(self, other) -> bool:
        if isinstance(other, Bag):
            return self._values == other._values
        else:
            return False

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return repr(self._values)

    def __hash__(self) -> int:
        return hash(self._values)
