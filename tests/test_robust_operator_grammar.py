from generator_orbit_chart_experiment import Config, CyclicChart
from robust_operator_grammar_experiment import full_group_accuracy


def test_full_group_accuracy_is_bounded():
    model=CyclicChart(Config(order=7,harmonics=2),seed=0,arm='generator_orbit_closure')
    value=full_group_accuracy(model)
    assert 0.0 <= value <= 1.0
