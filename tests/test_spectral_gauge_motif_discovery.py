import torch

from spectral_gauge_motif_discovery_experiment import spectral_sparsify


def _zero_sum_basis(p: int) -> torch.Tensor:
    # Span {e_i - e_{p-1}} and orthonormalize it.
    x = torch.zeros(p, p - 1, dtype=torch.float64)
    x[: p - 1] = torch.eye(p - 1, dtype=torch.float64)
    x[p - 1] = -1.0
    q, _ = torch.linalg.qr(x, mode="reduced")
    return q


def _perm_matrix(p: int, fn) -> torch.Tensor:
    P = torch.zeros(p, p, dtype=torch.float64)
    for x in range(p):
        P[fn(x), x] = 1.0
    return P


def test_spectral_gauge_recovers_sparse_affine_actions():
    p = 31
    q = _zero_sum_basis(p)
    add = _perm_matrix(p, lambda x: (x + 1) % p)
    mul = _perm_matrix(p, lambda x: (3 * x) % p)
    A = q.T @ add @ q
    M = q.T @ mul @ q

    recovered = spectral_sparsify(A, M)
    metrics = recovered["metrics"]
    assert metrics["A_diagonal_energy_fraction"] > 1 - 1e-10
    assert metrics["M_row_top1_energy_fraction_mean"] > 1 - 1e-10
    assert metrics["M_col_top1_energy_fraction_mean"] > 1 - 1e-10
    assert metrics["M_row_effective_support_mean"] < 1 + 1e-9
    assert metrics["A_reconstruction_relative_error"] < 1e-10
    assert metrics["M_reconstruction_relative_error"] < 1e-10
