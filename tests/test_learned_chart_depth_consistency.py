from learned_chart_depth_consistency_experiment import train_consistent
from learned_cyclic_chart_experiment import LearnedChartConfig, evaluate_chart


def test_multidepth_successor_consistency_stabilizes_long_recurrence():
    cfg = LearnedChartConfig()
    model, _ = train_consistent(cfg, seed=1, depths=(1, 2, 3), steps=600, lr=0.05)
    ev = evaluate_chart(
        model,
        arm="closed_cycle",
        sequence_examples=256,
        sequence_depths=(32, 64),
        seed=1,
    )
    assert ev["heldout_binary_pair_accuracy_b_ne_1"] == 1.0
    assert all(row["final_accuracy"] == 1.0 for row in ev["recurrence"])
