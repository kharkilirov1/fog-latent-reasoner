import torch

from joint_commutant_block_compiler_experiment import (
    compile_repeated_irrep,
    make_problem,
    max_word_spectral_distinct,
)


def test_commutant_compiler_recovers_repeated_irrep_without_simple_spectrum():
    p = make_problem(multiplicity=3, seed=5, broken=False)
    actions = [p["A"], p["B"]]
    spec = max_word_spectral_distinct(actions, max_depth=6)
    assert spec["distinct"] <= 2
    comp = compile_repeated_irrep(actions, seed=5)
    assert comp["accepted"]
    assert comp["commutant_dimension"] == 9
    assert comp["inferred_multiplicity"] == 3
    assert comp["inferred_block_dim"] == 2
    assert comp["max_aligned_block_relative_difference"] < 1e-8
    assert max(comp["reconstruction_relative_errors"]) < 1e-8
    assert comp["operator_storage_compression_factor"] == 9.0


def test_broken_shared_irrep_is_rejected_by_commutant_dimension():
    p = make_problem(multiplicity=3, seed=7, broken=True)
    comp = compile_repeated_irrep([p["A"], p["B"]], seed=7)
    assert not comp["accepted"]
    assert comp["commutant_dimension"] != 9
