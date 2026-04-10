from hologram.api import Hologram


def test_build_kg_batch_snapshot():
    hg = Hologram.init(use_gravity=False, encoder_mode="hash", use_clip=False)

    items = [
        {"id": "q1", "item_type": "query", "payload": "What is gravitational time dilation?"},
        {"id": "r1", "item_type": "response", "payload": "Time dilation increases near massive bodies."},
        {"id": "r2", "item_type": "response", "payload": "Massive bodies curve spacetime and affect clocks."},
    ]

    snapshot = hg.build_kg_batch("batch:alpha", items)

    assert snapshot["batch_id"] == "batch:alpha"
    assert snapshot["metadata"]["item_count"] == 3
    assert len(snapshot["nodes"]) > 0


def test_compare_drift_returns_dimensions():
    hg = Hologram.init(use_gravity=False, encoder_mode="hash", use_clip=False)

    baseline_items = [
        {"id": "bq", "payload": "Explain gravity in classical mechanics."},
        {"id": "br", "payload": "Gravity is a force between masses."},
    ]
    target_items = [
        {"id": "tq", "payload": "Explain gravity in general relativity."},
        {"id": "tr", "payload": "Gravity emerges from spacetime curvature."},
    ]

    report = hg.compare_drift(
        baseline_id="baseline:v1",
        target_id="target:v2",
        baseline_items=baseline_items,
        target_items=target_items,
    )

    assert report["baseline_id"] == "baseline:v1"
    assert report["target_id"] == "target:v2"
    assert "dimensions" in report and len(report["dimensions"]) >= 2
    dim_names = {d["name"] for d in report["dimensions"]}
    assert "embedding_centroid" in dim_names
    assert "kg_structure" in dim_names
    assert 0.0 <= report["drift_score"] <= 1.0
    assert 0.0 <= report["confidence"] <= 1.0
