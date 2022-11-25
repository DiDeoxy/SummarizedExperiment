"""A constructor for `SummarizedExperiment` types."""

from typing import Any, MutableMapping, Optional, Union, overload

from genomicranges import GenomicRanges  # type: ignore
from pandas import DataFrame

from .config import AssaysType
from .ranged_summarized_experiment import RangedSummarizedExperiment
from .summarized_experiment import SummarizedExperiment

__author__ = "jkanche, whargrea"
__copyright__ = "jkanche"
__license__ = "MIT"


@overload
def construct_summarized_experiment(
    assays: AssaysType,
    rows: Optional[DataFrame] = None,
    cols: Optional[DataFrame] = None,
    metadata: Optional[MutableMapping[str, Any]] = None,
) -> SummarizedExperiment:
    ...


@overload
def construct_summarized_experiment(
    assays: AssaysType,
    rows: GenomicRanges,
    cols: Optional[DataFrame] = None,
    metadata: Optional[MutableMapping[str, Any]] = None,
) -> RangedSummarizedExperiment:
    ...


def construct_summarized_experiment(
    assays: AssaysType,
    rows: Union[DataFrame, GenomicRanges, None] = None,
    cols: Optional[DataFrame] = None,
    metadata: Optional[MutableMapping[str, Any]] = None,
) -> Union[RangedSummarizedExperiment, SummarizedExperiment]:
    """Get an appropriate `SummarizedExperimentType` object.

    Parameters
    ----------
    assays : AssaysType
        A `MutableMapping` (e.g. `dict`) of with assay name as `key` with as
        `value` a dense (NDArray[Any]) or sparse matrix (spmatrix).
    rows : DataFrame | GenomicRanges | None
        Feature data if any. Default = `None`.
    cols : DataFrame | None
        Sample data is any, must have the same number of columns as the
        number of columns of each assay. Default = `None`.
    metadata : MutableMapping[str, Any] | None
        A `MutableMapping` of experiment metadata, usually describing the
        methods used, if any. Default = `None`.

    Returns
    -------
    summarized_experiment : RangedSummarizedExperiment | SummarizedExperiment
        The appropriate `SummarizedExperimentType` representation.
    """
    if rows is None or isinstance(rows, DataFrame):
        return SummarizedExperiment(assays, rows, cols, metadata)

    return RangedSummarizedExperiment(assays, rows, cols, metadata)
