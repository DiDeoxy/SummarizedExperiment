"""Configuration for summarized experiment types."""

from typing import Any, List, MutableMapping, Optional, Tuple, Union

from numpy.typing import NDArray
from scipy.sparse import spmatrix  # type: ignore

IndexType = Union[List[bool], List[int], slice]
OptionalIndexType = Optional[IndexType]
OptionalIndicesType = Union[
    List[IndexType], Tuple[IndexType], Tuple[IndexType, IndexType]
]
AssaysType = Union[
    MutableMapping[str, spmatrix],
    MutableMapping[str, NDArray[Any]],
    MutableMapping[str, Union[NDArray[Any], spmatrix]],
]
