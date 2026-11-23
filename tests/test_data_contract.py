"""Data-contract tests. These guard every downstream result: if they fail, nothing is reportable."""

import pytest

from satelliteguard import load_config
from satelliteguard.data.validate import load_raw, run_checks

cfg = load_config()
RAW = cfg["paths"]["raw"]

pytestmark = pytest.mark.skipif(
    not (RAW / "dataset.csv").exists(),
    reason="raw data not downloaded; run `python -m satelliteguard.data.download`",
)


@pytest.fixture(scope="module")
def audit():
    segments, dataset = load_raw(cfg)
    return run_checks(segments, dataset)


@pytest.mark.parametrize(
    "check",
    [
        "segments_columns",
        "dataset_required_columns",
        "dataset_unique_segment_ids",
        "one_channel_target_split_per_segment",
        "segment_ids_match_between_files",
        "target_split_channel_agree_between_files",
        "split_counts_match_publisher",
    ],
)
def test_contract(audit, check):
    assert audit["checks"][check]["passed"], audit["checks"][check]["detail"]


def test_no_train_test_overlap():
    _, dataset = load_raw(cfg)
    train_ids = set(dataset.loc[dataset["train"] == 1, "segment"])
    test_ids = set(dataset.loc[dataset["train"] == 0, "segment"])
    assert not train_ids & test_ids
    assert len(train_ids) + len(test_ids) == dataset["segment"].nunique()
