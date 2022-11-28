"""The SummarizedExperiment class."""

from typing import Any, MutableMapping, Optional, Sequence, Tuple, Union

from numpy.typing import NDArray
from pandas import DataFrame
from scipy.sparse import spmatrix  # type: ignore

from .config import AssaysType, OptionalIndexType, OptionalIndicesType

__author__ = "jkanche, whargrea"
__copyright__ = "jkanche"
__license__ = "MIT"


class BaseSummarizedExperiment:
    """Base class for `SummarizedExperiment`."""

    def __init__(
        self,
        assays: AssaysType,
        rows: Optional[DataFrame] = None,
        cols: Optional[DataFrame] = None,
        metadata: Optional[MutableMapping[str, Any]] = None,
    ) -> None:
        """Initialize an instance of `SummarizedExperiment`.

        Parameters
        ----------
        assays : MutableMapping[str, NDArray[Any] | spmatrix]
            A `MutableMapping` (e.g. `dict`) of with assay name as `key` with
            as `value` a dense (NDArray[Any]) or sparse matrix (spmatrix).
        rows : DataFrame | None
            Feature data if any, must have the same number of columns as the
            number of rows of each assay. Default = `None`.
        cols : DataFrame | None
            Sample data is any, must have the same number of columns as the
            number of columns of each assay. Default = `None`.
        metadata : MutableMapping[str, Any] | None
            A `MutableMapping` of experiment metadata, usually describing the
            methods used, if any. Default = `None`.
        """
        self._rows = rows
        self._assays = assays
        self._cols = cols
        self._metadata = metadata

        self._validate_data()

    def _validate_assays(self, assays: AssaysType) -> None:
        """Validate the assays attribute.

        Parameters
        ----------
        assays : AssaysType
            The assays to validate.
        """
        if len(assays) == 0:
            raise ValueError(
                "An empty `MutableMapping` is not a valid argument to the "
                "`assays` parameter."
            )

        # ensure all assays have the same shape
        first_name = ""
        shape: Optional[Sequence[int]] = None
        for name, matrix in assays.items():
            if len(matrix.shape) > 2:
                raise ValueError("Only 2D matrices are accepted.")
            if shape is None:
                first_name = name
                shape = matrix.shape
            else:
                for i, dim in enumerate(shape):
                    if matrix.shape[i] != dim:
                        raise ValueError(
                            f"The matrix: '{name}' does not have the same "
                            f"shape as '{first_name}'."
                        )

    def _validate_rows(self, rows: Optional[DataFrame]) -> None:
        """Validate the row data.

        Parameters
        ----------
        rows : DataFrame | None
            The rows to validate.
        """
        if rows is not None:
            rows_num_rows = rows.shape[0]
            assay_num_rows = next(iter(self._assays.values())).shape[0]
            if rows_num_rows != assay_num_rows:
                raise ValueError(
                    f"The row data has: '{rows_num_rows}' columns which does "
                    "not equal the number of rows in the assays: "
                    f"'{assay_num_rows}'."
                )

    def _validate_cols(self, cols: Optional[DataFrame]) -> None:
        """Validate the row data.

        Parameters
        ----------
        cols : DataFrame | None
            The cols to validate.
        """
        if cols is not None:
            cols_num_rows = cols.shape[0]
            assays_num_cols = next(iter(self._assays.values())).shape[1]
            if cols_num_rows != assays_num_cols:
                raise ValueError(
                    f"The column data has: '{cols_num_rows}' columns which "
                    "does not equal the number of columns in the assays: "
                    f"'{assays_num_cols}'."
                )

    def _validate_data(self) -> None:
        """Validate the assay, row, and column data."""
        # assays must be validated first
        self._validate_assays(self._assays)
        self._validate_rows(self._rows)
        self._validate_cols(self._cols)

    @property
    def assays(self) -> AssaysType:
        """Get or set the the assays.

        Parameters
        ----------
        assays : MutableMapping[str, NDArray[Any] | spmatrix]
            A `MutableMapping` of the assay matrices.

        Returns
        -------
        assays : MutableMapping[str, NDArray[Any] | spmatrix]
            A `MutableMapping` of the assay matrices.
        """
        return self._assays

    @assays.setter
    def assays(self, assays: AssaysType) -> None:
        self._validate_assays(assays)
        self._assays = assays

    @property
    def colData(self) -> Optional[DataFrame]:
        """Get or set the column data.

        Parameters
        ----------
        cols : DataFrame | None
            New column data.

        Returns
        -------
        cols : DataFrame | None
            The column data of the `SummarizedExperiment`
        """
        return self._cols

    @colData.setter
    def colData(self, cols: Optional[DataFrame]) -> None:
        self._validate_cols(cols)
        self._cols = cols

    @property
    def metadata(self) -> Any:
        """Get or set the metadata.

        Parameters
        ----------
        metadata : MutableMapping[str, Any] | None
            A `MutableMapping` of experiment metadata, usually describing the
            methods used, if any. Default = `None`.

        Returns
        -------
        metadata : MutableMapping[str, Any] | None
            The current metadata.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Optional[MutableMapping[str, Any]]) -> None:
        self._metadata = metadata

    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the `SummarizedExperiment`.

        Returns
        -------
        shape : Tuple[int, int]
            The shape of the `SummarizedExperiment`.
        """
        return tuple(next(iter(self._assays.values())).shape)

    def subset_assays(
        self,
        row_indices: OptionalIndexType = None,
        col_indices: OptionalIndexType = None,
    ) -> MutableMapping[str, Union[NDArray[Any], spmatrix]]:
        """Get identically sub-setted assays.

        Parameters
        ----------
        row_indices : List[bool] | List[int] | slice | None
            The indices to subset the assay rows with, if `None` all rows will
            be retained. Default = `None`.
        col_indices : List[bool] | List[int] | slice | None
            The indices to subset the assay columns with, if `None` all columns
            will be retained. Default = `None`.

        Returns
        -------
        assay_subsets : MutableMapping[str, Union[NDArray[Any], spmatrix]]
            The assays subset by the given row and/or column indices.
        """
        if row_indices is None and col_indices is None:
            raise TypeError(
                "At least one of 'rowIndices' and 'colIndices' must not be "
                "'None'"
            )

        assay_subsets: MutableMapping[str, Union[NDArray[Any], spmatrix]] = {}
        for name, assay in self._assays.items():
            if row_indices is not None:
                assay = assay[row_indices, ...]  # type: ignore
            if col_indices is not None:
                assay = assay[..., col_indices]  # type: ignore

            assay_subsets[name] = assay

        return assay_subsets

    def __str__(self) -> str:
        """Represent the object as a `str`.

        Returns
        -------
        __str__ : str
            A description of the object.
        """
        sample_names = None if self._cols is None else self._cols.columns
        feature_names = None if self._rows is None else self._rows.columns
        return f"""\
Class: 'SummarizedExperiment'
    Shape: {self.shape}
    Assay names: {list(self._assays.keys())}
    Sample names: {sample_names}
    Feature names: {feature_names}
"""


class SummarizedExperiment(BaseSummarizedExperiment):
    """Container for genomic experiment metadata and data."""

    @property
    def rowData(self) -> Optional[DataFrame]:
        """Get or set the row data.

        Parameters
        ----------
        rows : DataFrame | None
            New row data.

        Returns
        -------
        rows : DataFrame | None
            The row data of the `SummarizedExperiment`
        """
        return self._rows

    @rowData.setter
    def rowData(self, rows: Optional[DataFrame]) -> None:
        self._validate_rows(rows)
        self._rows = rows

    def __getitem__(
        self, indices: OptionalIndicesType
    ) -> "SummarizedExperiment":
        """Subset a `SummarizedExperiment`.

        Parameters
        ----------
        indices : Sequence[List[bool] | List[int] | slice | None]
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
        num_args = len(indices)
        if num_args < 1 or num_args > 2:
            raise ValueError(
                f"'{len(indices)}' indices were given, at least '1' and a "
                "maximum of '2' are accepted."
            )

        row_indices = indices[0]
        col_indices = None if len(indices) == 1 else indices[1]

        assay_subsets = self.subset_assays(row_indices, col_indices)
        rows_subset = (
            None
            if row_indices is None or self._rows is None
            else self._rows.iloc[row_indices]
        )
        cols_subset = (
            None
            if col_indices is None or self._cols is None
            else self._cols.iloc[col_indices]
        )

        return SummarizedExperiment(
            assay_subsets, rows_subset, cols_subset, self._metadata
        )
