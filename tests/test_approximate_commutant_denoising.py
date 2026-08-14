from approximate_commutant_denoising_experiment import (
    compile_approximate_repeated_irrep,
    infer_approximate_commutant,
    perturb_orthogonal,
)
from joint_commutant_block_compiler_experiment import _orthogonal, make_problem


def test_approximate_commutant_infers_multiplicity_from_singular_gap():
    p = make_problem(3, seed=13)
    actions = [
        perturb_orthogonal(p["A"], 0.05, 101),
        perturb_orthogonal(p["B"], 0.05, 102),
    ]
    comp = compile_approximate_repeated_irrep(actions, seed=13, gap_threshold=5.0)
    assert comp["accepted"]
    assert comp["best"]["multiplicity"] == 3
    assert comp["best"]["block_dim"] == 2
    assert comp["best"]["gap_ratio"] > 10


def test_generic_random_operator_pair_has_no_repeated_irrep_gap():
    actions = [_orthogonal(1, 8), _orthogonal(2, 8)]
    inferred = infer_approximate_commutant(actions, gap_threshold=5.0)
    assert not inferred["accepted"]
