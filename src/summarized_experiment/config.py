"""Configuration for summarized experiment types."""

from typing import Any, List, MutableMapping, Optional, Sequence, Union

from numpy.typing import NDArray
from scipy.sparse import spmatrix  # type: ignore

IndexType = Union[List[bool], List[int], slice]
OptionalIndexType = Optional[IndexType]
OptionalIndicesType = Sequence[OptionalIndexType]
AssaysType = Union[
    MutableMapping[str, spmatrix],
    MutableMapping[str, NDArray[Any]],
    MutableMapping[str, Union[NDArray[Any], spmatrix]],
]
