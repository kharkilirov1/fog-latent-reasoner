from transition_stability_diagnostic import diagnose
from normed_operator_parameterization_experiment import Config, StructuredNormedAlgebra


def test_stability_diagnostic_returns_finite_metrics():
    model = StructuredNormedAlgebra(Config(), seed=0).eval()
    result = diagnose(model, seed=0, perturb_eps=0.01, samples=32, max_depth=4)
    assert result["closure_angle_mean_rad"] >= 0
    assert result["local_gain_mean"] >= 0
    assert "4" in result["trajectory"]
