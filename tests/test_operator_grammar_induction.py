from operator_grammar_induction_experiment import discrete_log_table, sample_nonzero_add_pair, sample_mul_pair
import torch


def test_field_pair_generators_and_logs_are_consistent():
    exp_to_value, value_to_exp = discrete_log_table()
    assert sorted(exp_to_value) == list(range(1, 31))
    for e, v in enumerate(exp_to_value):
        assert value_to_exp[v] == e
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        a, b, c = sample_nonzero_add_pair(g)
        assert c == (a + b) % 31 and c != 0
        a, b, c = sample_mul_pair(g)
        assert c == (a * b) % 31 and c != 0
