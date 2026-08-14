from learned_cyclic_chart_experiment import (
    LearnedChartConfig,
    evaluate_chart,
    train_chart,
)


def test_closed_cycle_successor_law_learns_full_addition_chart():
    cfg = LearnedChartConfig()
    model, _ = train_chart(cfg, arm="closed_cycle", seed=0, steps=600, lr=0.05)
    ev = evaluate_chart(
        model,
        arm="closed_cycle",
        sequence_examples=256,
        sequence_depths=(8,),
        seed=0,
    )
    assert ev["successor_accuracy_all_edges"] == 1.0
    assert ev["heldout_binary_pair_accuracy_b_ne_1"] == 1.0
    assert ev["recurrence"][0]["final_accuracy"] == 1.0
