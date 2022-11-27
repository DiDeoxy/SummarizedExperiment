"""Test `SummarizedExperimentTypes`."""

from random import random

from genomicranges import GenomicRanges  # type: ignore
from numpy.random import rand
from pandas import DataFrame

from summarized_experiment import construct_summarized_experiment
from summarized_experiment.ranged_summarized_experiment import (
    RangedSummarizedExperiment,
)
from summarized_experiment.summarized_experiment import SummarizedExperiment

__author__ = "jkanche, whargrea"
__copyright__ = "jkanche"
__license__ = "MIT"


row_data = DataFrame(
    {
        "seqnames": [
            "chr1",
            "chr2",
            "chr2",
            "chr2",
            "chr1",
            "chr1",
            "chr3",
            "chr3",
            "chr3",
            "chr3",
        ],
        "starts": range(100, 110),
        "ends": range(110, 120),
        "strand": ["-", "+", "+", "*", "*", "+", "+", "+", "-", "-"],
        "score": range(0, 10),
        "GC": [random() for _ in range(10)],
    }
)

ranges_data = GenomicRanges(row_data)

cols = DataFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)


num_rows = 200
num_cols = 6
counts = {"counts": rand(num_rows, num_cols)}


# nosec
# noqa: B101
def test_SE_creation():
    summarized_experiment = construct_summarized_experiment(
        counts, row_data, cols
    )

    assert summarized_experiment is not None
    assert isinstance(summarized_experiment, SummarizedExperiment)
    assert summarized_experiment.shape == (200, 6)

    _ = summarized_experiment[list(range(10)), list(range(3))]
    _ = summarized_experiment[
        [True] * 10 + [False] * 190, [True] * 3 + [False] * 3
    ]
    summarized_experiment_subset = summarized_experiment[0:10, 0:3]

    assert summarized_experiment_subset is not None
    assert isinstance(summarized_experiment_subset, SummarizedExperiment)
    assert summarized_experiment_subset.rowData is not None
    assert len(summarized_experiment_subset.rowData) == 10
    assert summarized_experiment_subset.colData is not None
    assert len(summarized_experiment_subset.colData) == 3
    assert summarized_experiment_subset.shape == (10, 3)


def test_RSE_creation():
    ranged_summarized_experiment = construct_summarized_experiment(
        counts, ranges_data, cols
    )

    assert ranged_summarized_experiment is not None
    assert isinstance(ranged_summarized_experiment, RangedSummarizedExperiment)

    # TODO: fix genomic ranges for this type of slicing
    # _ = ranged_summarized_experiment[list(range(10)), list(range(3))]
    # _ = ranged_summarized_experiment[
    #     [True] * 10 + [False] * 190, [True] * 3 + [False] * 3
    # ]
    ranged_summarized_experiment_subset = ranged_summarized_experiment[
        0:10, 0:3
    ]

    assert ranged_summarized_experiment_subset is not None
    assert isinstance(
        ranged_summarized_experiment_subset, RangedSummarizedExperiment
    )
    assert ranged_summarized_experiment_subset.rowRanges is not None
    assert len(ranged_summarized_experiment_subset.rowRanges.ranges) == 10
    assert ranged_summarized_experiment_subset.colData is not None
    assert len(ranged_summarized_experiment_subset.colData) == 3
    assert ranged_summarized_experiment_subset.shape == (10, 3)


def test_SE_none():
    summarized_experiment = construct_summarized_experiment(counts)

    assert summarized_experiment is not None
    assert isinstance(summarized_experiment, SummarizedExperiment)
