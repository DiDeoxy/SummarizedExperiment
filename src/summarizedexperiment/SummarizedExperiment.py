from typing import (
    Any,
    Union,
    Optional,
    MutableMapping,
    Hashable,
    Sequence,
)
from scipy.sparse import spmatrix
from numpy.typing import NDArray
from pandas import DataFrame
import logging

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class SummarizedExperiment:
    """Container for genomic experiment metadata and data."""

    def __init__(
        self,
        assays: MutableMapping[Hashable, Union[NDArray[Any], spmatrix]],
        rows: Optional[DataFrame] = None,
        cols: Optional[DataFrame] = None,
        metadata: Optional[MutableMapping[Hashable, Any]] = None,
    ) -> None:
        """Initialize an instance of `SummarizedExperiment`

        Parameters
        ----------
        assays : MutableMapping[Hashable, Union[NDArray[Any], spmatrix]]
            A `MutableMapping` (e.g. `dict`) of with assay name as `key` with
            as `value` a dense (NDArray[Any]) or sparse matrix (spmatrix).
        rows : DataFrame | None
            Feature data if any, must have the same number of columns as the
            number of rows of each assay. Default = `None`.
        cols : DataFrame | None
            Sample data is any, must have the same number of columns as the
            number of columns of each assay. Default = `None`.
        metadata : MutableMapping[Hashable, Any] | None
            A `MutableMapping` of experiment metadata, usually describing the
            methods used, if any. Default = `None`.
        """

        self._rows = rows
        self._assays = assays
        self._cols = cols
        self._metadata = metadata

        self._validate_data()

    def _validate_assays(
        self, assays: MutableMapping[Hashable, Union[NDArray[Any], spmatrix]]
    ) -> None:
        """Validate the assays attribute.

        Parameters
        ----------
        assays : MutableMapping[Hashable, Union[NDArray[Any], spmatrix]]
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
            row_cols = rows.shape[1]
            assay_rows = self._assays[next(iter(self._assays))].shape[0]
            if row_cols != assay_rows:
                raise ValueError(
                    f"The row data has: '{row_cols}' columns which does not "
                    f"equal the number of rows in the assays: '{assay_rows}'."
                )

    def _validate_cols(self, cols: Optional[DataFrame]) -> None:
        """Validate the row data.

        Parameters
        ----------
        cols : DataFrame | None
            The cols to validate.
        """
        if cols is not None:
            cols_cols = cols.shape[1]
            assay_cols = self._assays[next(iter(self._assays))].shape[1]
            if cols_cols != assay_cols:
                raise ValueError(
                    f"The column data has: '{cols_cols}' columns which does "
                    "not equal the number of columns in the assays: "
                    f"'{assay_cols}'."
                )

    def _validate_data(self) -> None:
        """Validate the assay, row, and column data."""
        # assays must be validated first
        self._validate_assays(self._assays)
        self._validate_rows(self._rows)
        self._validate_cols(self._cols)

    @property
    def assays(
        self,
    ) -> MutableMapping[Hashable, Union[NDArray[Any], spmatrix]]:
        """Get or set the the assays.

        Parameters
        ----------
        assays : MutableMapping[Hashable, Union[NDArray[Any], spmatrix]]
            A `MutableMapping` of the assay matrices.

        Returns
        -------
        assays : MutableMapping[Hashable, Union[NDArray[Any], spmatrix]]
            A `MutableMapping` of the assay matrices.
        """
        return self._assays

    @assays.setter
    def assays(
        self, assays: MutableMapping[Hashable, Union[NDArray[Any], spmatrix]]
    ) -> None:
        self._validate_assays(assays)
        self._assays = assays

    # HERE
    @property
    def rowData(self) -> DataFrame:
        """Accessor to retrieve features

        Returns:
            pd.DataFrame: returns features in the container
        """
        return self._rows

    def colData(self) -> pd.DataFrame:
        """Accessor to retrieve sample metadata

        Returns:
            pd.DataFrame: returns features in the container
        """
        return self.cols

    def subsetAssays(
        self, rowIndices: List[int] = None, colIndices: List[int] = None
    ) -> Dict[str, Union[np.ndarray, sp.spmatrix]]:
        """Subset all assays to a specific row or col indices

        Args:
            rowIndices (List[int], optional): row indices to subset. Defaults to None.
            colIndices (List[int], optional): col indices to subset. Defaults to None.

        Returns:
            Dict[str, Union[np.ndarray, sp.spmatrix]]: assays with subset matrices
        """

        new_assays = {}
        for key in self._assays:
            mat = self._assays[key]
            if rowIndices is not None:
                mat = mat[rowIndices, :]

            if colIndices is not None:
                mat = mat[:, colIndices]

            new_assays[key] = mat

        return new_assays

    def __getitem__(self, args: tuple) -> "SummarizedExperiment":
        """Subset a SummarizedExperiment

        Args:
            args (tuple): indices to slice. tuple can
                contains slices along dimensions

        Raises:
            Exception: Too many slices

        Returns:
            SummarizedExperiment: new SummarizedExperiment object
        """
        rowIndices = args[0]
        colIndices = None

        if len(args) > 1:
            colIndices = args[1]
        elif len(args) > 2:
            logging.error(
                f"too many slices, args length must be 2 but provided {len(args)} "
            )
            raise Exception("contains too many slices")

        new_rows = None
        new_cols = None
        new_assays = None

        if rowIndices is not None and self.rows is not None:
            new_rows = self.rows.iloc[rowIndices]

        if colIndices is not None and self.cols is not None:
            new_cols = self.cols.iloc[colIndices]

        new_assays = self.subsetAssays(
            rowIndices=rowIndices, colIndices=colIndices
        )

        return SummarizedExperiment(
            new_assays, new_rows, new_cols, self.metadata
        )

    def metadata(self) -> Any:
        """Access Experiment metadata

        Returns:
            Any: metadata (could be anything)
        """
        return self.metadata

    @property
    def shape(self) -> tuple:
        """Get shape of the object

        Returns:
            tuple: dimensions of the experiment; (rows, cols)
        """
        a_mat = self._assays[list(self._assays.keys())[0]]
        return a_mat.shape

    def __str__(self) -> str:
        """string representation

        Returns:
            str: description of the class
        """
        return (
            "Class: SummarizedExperiment\n"
            f"\tshape: {self.shape}\n"
            f"\tcontains assays: {list(self._assays.keys())}\n"
            f"\tsample metadata: {self.cols.columns if self.cols is not None else None}\n"
            f"\tfeatures: {self.rows.columns if self.rows is not None else None}\n"
        )

    def assay(self, name: str) -> Union[np.ndarray, sp.spmatrix]:
        """Convenience function to access an assay by name

        Args:
            name (str): name of the assay

        Raises:
            Exception: if assay name does not exist

        Returns:
            Union[np.ndarray, sp.spmatrix]: the experiment data
        """
        if name not in self._assays:
            logging.error(f"Assay {name} does not exist")
            raise Exception(f"Assay {name} does not exist")

        return self._assays[name]
