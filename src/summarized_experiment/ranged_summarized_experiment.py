"""The RangedSummarizedExperiment class."""

from typing import Any, MutableMapping, Optional

from genomicranges import GenomicRanges  # type: ignore
from pandas import DataFrame

from .config import AssaysType, OptionalIndicesType
from .summarized_experiment import BaseSummarizedExperiment

__author__ = "jkanche, whargrea"
__copyright__ = "jkanche"
__license__ = "MIT"


class RangedSummarizedExperiment(BaseSummarizedExperiment):
    """Container for genomic experiment metadata and ranged data."""

    def __init__(
        self,
        assays: AssaysType,
        rows: Optional[GenomicRanges] = None,
        cols: Optional[DataFrame] = None,
        metadata: Optional[MutableMapping[str, Any]] = None,
    ) -> None:
        """Initialize an instance of `RangedSummarizedExperiment`.

        Parameters
        ----------
        assays : AssaysType
            A `MutableMapping` (e.g. `dict`) of with assay name as `key` with
            as `value` a dense (NDArray[Any]) or sparse matrix (spmatrix).
        rows : GenomicRanges | None
            Feature data if any. Default = `None`.
        cols : DataFrame | None
            Sample data is any, must have the same number of columns as the
            number of columns of each assay. Default = `None`.
        metadata : MutableMapping[str, Any] | None
            A `MutableMapping` of experiment metadata, usually describing the
            methods used, if any. Default = `None`.
        """
        super().__init__(assays, None, cols, metadata=metadata)
        self._rows = rows

    @property
    def rowRanges(self) -> Optional[GenomicRanges]:
        """Get or set the row data.

        Parameters
        ----------
        rows : GenomicRanges | None
            New row data.

        Returns
        -------
        rows : GenomicRanges | None
            The row data of the `RangedSummarizedExperiment`
        """
        return self._rows

    @rowRanges.setter
    def rowRanges(self, rows: Optional[GenomicRanges]) -> None:
        # self._validate_row_ranges()
        self._rows = rows

    def __getitem__(
        self, args: OptionalIndicesType
    ) -> "RangedSummarizedExperiment":
        """Subset a `RangedSummarizedExperiment`.

        Parameters
        ----------
        args : OptionalIndicesType
            Row and possibly column indices to subset by. If `None` all rows or
            columns will be retained. If only one value is provided all columns
            will be retained.

        Raises
        ------
        ValueError
            Too many indices.

        Returns
        -------
        summarized_experiment : SummarizedExperiment
            A new `SummarizedExperiment` with the subset of data.
        """  # noqa: E501
        num_args = len(args)
        if num_args < 1 or num_args > 2:
            raise ValueError(
                f"'{len(args)}' indices were given, at least '1' and a "
                "maximum of '2' are accepted."
            )

        row_indices = args[0]
        col_indices = None if len(args) == 1 else args[1]

        assay_subsets = self.subset_assays(row_indices, col_indices)
        rows_subset = (
            None
            if row_indices is None or self._rows is None
            # TODO: support boolean indexing
            else self._rows[row_indices]  # type: ignore
        )
        cols_subset = (
            None
            if col_indices is None or self._cols is None
            else self._cols.iloc[col_indices]
        )

        return RangedSummarizedExperiment(
            assay_subsets, rows_subset, cols_subset, self._metadata
        )

    def subsetByOverlaps(
        self, query: GenomicRanges
    ) -> "RangedSummarizedExperiment":
        """Subset a `RangedSummarizedExperiment` by feature overlaps.

        Parameters
        ----------
        query : GenomicRanges
            Query genomic intervals.

        Returns
        -------
        ranged_summarized_experiment : RangedSummarizedExperiment
            Subset of original `SummarizedExperiment` object

        Raises
        ------
        NotImplementedError
            Currently not implemented
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Represent the object as a `str`.

        Returns
        -------
        __str__ : str
            A description of the object.
        """
        sample_names = None if self._cols is None else self._cols.columns
        ranges_names = (
            None if self._rows is None else self._rows.ranges().columns
        )
        return f"""\
Class: 'SummarizedExperiment'
    Shape: {self.shape}
    Assay names: {list(self._assays.keys())}
    Sample names: {sample_names}
    Feature names: {ranges_names}
"""
