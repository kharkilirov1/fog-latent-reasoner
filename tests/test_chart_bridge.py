from chart_bridge_experiment import identity_split


def test_bridge_identity_split_is_disjoint_and_complete():
    train, test, _ = identity_split(30, 101)
    assert set(train).isdisjoint(set(test))
    assert set(train) | set(test) == set(range(30))
