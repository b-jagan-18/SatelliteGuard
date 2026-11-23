"""Episode grouping: unit tests on synthetic data + contract checks on the real index."""

import pandas as pd
import pytest

from satelliteguard import load_config
from satelliteguard.data.episodes import build_segment_index


def _toy_segments():
    """Channel A: seg 1,2 contiguous (1 s apart), seg 3 after a 1 h gap. Channel B: seg 4."""
    rows = []
    t0 = pd.Timestamp("2022-01-01", tz="UTC")
    spec = [  # segment, channel, start offset (s), n samples, train, anomaly
        (1, "A", 0, 5, 1, 0),
        (2, "A", 5, 5, 0, 1),
        (3, "A", 3600, 5, 1, 0),
        (4, "B", 0, 5, 1, 0),
    ]
    for seg, ch, off, n, tr, an in spec:
        for k in range(n):
            rows.append({"channel": ch, "timestamp": t0 + pd.Timedelta(seconds=off + k),
                         "value": 0.0, "label": "anomaly", "sampling": 1,
                         "anomaly": an, "segment": seg, "train": tr})
    return pd.DataFrame(rows)


def test_episode_boundaries():
    idx = build_segment_index(_toy_segments(), max_gap_s=10).set_index("segment")
    assert idx.loc[1, "episode"] == idx.loc[2, "episode"]          # contiguous -> same episode
    assert idx.loc[3, "episode"] != idx.loc[2, "episode"]          # 1 h gap -> new episode
    assert idx.loc[4, "episode"] not in idx.loc[[1, 2, 3], "episode"].values  # other channel


def test_touches_other_split():
    idx = build_segment_index(_toy_segments(), max_gap_s=10).set_index("segment")
    assert idx.loc[1, "touches_other_split"] and idx.loc[2, "touches_other_split"]
    assert not idx.loc[3, "touches_other_split"]
    assert not idx.loc[4, "touches_other_split"]


cfg = load_config()
INDEX = cfg["paths"]["processed"] / "segment_index.csv"


@pytest.mark.skipif(not INDEX.exists(), reason="run `python -m satelliteguard.data.episodes`")
def test_real_index_contract():
    idx = pd.read_csv(INDEX)
    dataset = pd.read_csv(cfg["paths"]["raw"] / "dataset.csv")
    assert idx["segment"].is_unique
    assert set(idx["segment"]) == set(dataset["segment"])
    # every episode lives on exactly one channel
    assert idx.groupby("episode")["channel"].nunique().eq(1).all()
