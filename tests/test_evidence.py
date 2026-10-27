"""Tests for ground-truth evidence handling (NExT-GQA style grounding).

The loader is tested against a synthetic annotation fixture, so these run
without the 6.6 GB video download.
"""

import json

import numpy as np
import pytest

from afs.data.base import QASample, covers_evidence, evidence_groups, spans_to_groups


def sample_with(meta) -> QASample:
    return QASample(
        qid="q", video_id="v", video_path="", question="?",
        options=["a", "b"], answer_idx=0, meta=meta,
    )


# ---------------- spans -> frame indices ----------------

def test_spans_map_to_the_decoded_grid():
    times = np.arange(0, 30, 1.0)          # 1 fps
    groups = spans_to_groups([[3.0, 5.0], [10.5, 12.5]], times)
    assert groups[0] == [3, 4, 5]
    assert groups[1] == [11, 12]


def test_spans_are_resolved_against_whatever_fps_we_decoded_at():
    """The same annotation must land on the same seconds at any sampling rate."""
    span = [[8.0, 9.0]]
    g1 = spans_to_groups(span, np.arange(0, 30, 1.0))     # 1 fps
    g2 = spans_to_groups(span, np.arange(0, 30, 0.5))     # 2 fps
    assert {float(i * 1.0) for i in g1[0]} <= {8.0, 9.0}
    assert {float(i * 0.5) for i in g2[0]} <= {8.0, 8.5, 9.0}


def test_short_span_snaps_to_nearest_frame_rather_than_vanishing():
    times = np.arange(0, 30, 1.0)
    groups = spans_to_groups([[7.2, 7.4]], times)   # falls between samples
    assert groups == [[7]]


def test_span_beyond_the_decoded_range_still_yields_a_frame():
    times = np.arange(0, 10, 1.0)
    groups = spans_to_groups([[50.0, 60.0]], times)
    assert len(groups) == 1 and len(groups[0]) == 1


def test_empty_times_is_handled():
    assert spans_to_groups([[1.0, 2.0]], np.array([])) == []


# ---------------- any vs all ----------------

def test_evidence_mode_any_vs_all():
    s = sample_with({"evidence_groups": [[1, 2], [8, 9]]})
    assert covers_evidence(s, [1], mode="any") is True
    assert covers_evidence(s, [1], mode="all") is False
    assert covers_evidence(s, [1, 8], mode="all") is True
    assert covers_evidence(s, [5], mode="any") is False


def test_mode_defaults_to_the_dataset_convention():
    any_s = sample_with({"evidence_groups": [[1], [9]], "evidence_mode": "any"})
    all_s = sample_with({"evidence_groups": [[1], [9]], "evidence_mode": "all"})
    assert covers_evidence(any_s, [1]) is True
    assert covers_evidence(all_s, [1]) is False


def test_no_annotation_returns_none():
    assert covers_evidence(sample_with({}), [1, 2]) is None


def test_flat_needle_frames_are_treated_as_one_group():
    s = sample_with({"needle_frames": [4, 5]})
    assert evidence_groups(s) == [[4, 5]]
    assert covers_evidence(s, [5]) is True


# ---------------- the NExT-GQA loader ----------------

@pytest.fixture
def fake_nextgqa(tmp_path):
    """A two-question NExT-GQA directory, real schema, fake content."""
    root = tmp_path / "nextgqa"
    (root / "annotations").mkdir(parents=True)
    (root / "videos").mkdir()

    import pandas as pd

    rows = []
    for vid, qid, ans in [("1111", "1", 2), ("2222", "3", 0)]:
        (root / "videos" / f"{vid}.mp4").write_bytes(b"not-a-real-video")
        rows.append({
            "video_id": vid, "frame_count": 900, "width": 640, "height": 480,
            "question": "why did the dog bark", "answer": ans, "qid": qid,
            "type": "CW",
            **{f"a{i}": f"option {i}" for i in range(5)},
        })
    # a third question whose video is absent, to exercise the skip path
    rows.append({
        "video_id": "9999", "frame_count": 900, "width": 640, "height": 480,
        "question": "unused", "answer": 0, "qid": "1", "type": "TN",
        **{f"a{i}": f"option {i}" for i in range(5)},
    })
    pd.DataFrame(rows).to_csv(root / "annotations" / "val.csv", index=False)

    gsub = {
        "1111": {"duration": 30, "fps": 30.0, "location": {"1": [[2.0, 4.0]]}},
        "2222": {"duration": 40, "fps": 25.0,
                 "location": {"3": [[1.0, 3.0], [20.0, 22.0]]}},
        "9999": {"duration": 20, "fps": 30.0, "location": {"1": [[1.0, 2.0]]}},
    }
    (root / "annotations" / "gsub_val.json").write_text(json.dumps(gsub))
    return root


def test_nextgqa_loader_reads_grounding(fake_nextgqa):
    from afs.data.nextgqa import NextGQADataset

    ds = NextGQADataset(root=fake_nextgqa, split="val")
    assert len(ds) == 2                      # the missing-video row is skipped
    assert ds.n_skipped_no_video == 1
    assert ds.name == "nextgqa-val"

    by_qid = {s.qid: s for s in ds}
    a = by_qid["1111_1"]
    assert a.answer_idx == 2 and a.answer_text == "option 2"
    assert a.qtype == "causal-why"           # code expanded for readable tables
    assert a.meta["evidence_spans"] == [[2.0, 4.0]]
    assert a.meta["evidence_mode"] == "any"
    assert a.duration == 30

    b = by_qid["2222_3"]
    assert b.meta["evidence_spans"] == [[1.0, 3.0], [20.0, 22.0]]


def test_nextgqa_spans_become_groups_on_a_real_grid(fake_nextgqa):
    from afs.data.nextgqa import NextGQADataset

    ds = NextGQADataset(root=fake_nextgqa, split="val")
    s = {x.qid: x for x in ds}["2222_3"]
    times = np.arange(0, 40, 1.0)
    groups = spans_to_groups(s.meta["evidence_spans"], times)
    assert groups == [[1, 2, 3], [20, 21, 22]]

    s.meta["evidence_groups"] = groups
    assert covers_evidence(s, [2]) is True             # "any" is the default
    assert covers_evidence(s, [2], mode="all") is False
    assert covers_evidence(s, [2, 21], mode="all") is True


def test_nextgqa_raises_clearly_when_annotations_are_missing(tmp_path):
    from afs.data.nextgqa import NextGQADataset

    with pytest.raises(FileNotFoundError, match="val.csv"):
        NextGQADataset(root=tmp_path / "nope", split="val")


def test_nextgqa_keeps_ungrounded_questions_when_asked(fake_nextgqa):
    from afs.data.nextgqa import NextGQADataset

    gsub = fake_nextgqa / "annotations" / "gsub_val.json"
    data = json.loads(gsub.read_text())
    data["1111"]["location"] = {}            # drop the grounding
    gsub.write_text(json.dumps(data))

    strict = NextGQADataset(root=fake_nextgqa, split="val")
    assert len(strict) == 1
    loose = NextGQADataset(root=fake_nextgqa, split="val", require_grounding=False)
    assert len(loose) == 2
