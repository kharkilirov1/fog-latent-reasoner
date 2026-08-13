from __future__ import annotations

from .config import FOGReasonerConfig


FOG_10M_VOCAB_SIZE = 8192
FOG_10M_PARAMETER_COUNT = 10_035_848
FOG_BINDING_V2_10M_PARAMETER_COUNT = 10_000_039


def fog_10m_config(
    *,
    vocab_size: int = FOG_10M_VOCAB_SIZE,
    max_seq_len: int = 512,
    reasoning_steps: int = 4,
    dropout: float = 0.1,
) -> FOGReasonerConfig:
    """Balanced ~10M preset used by the real-data training pipeline.

    The exact reference count is 10,035,848 only for vocab=8192 and seq=512.
    Four slots are the only toy geometry with a positive mechanistic run so
    far.  An eight-slot memory cap makes compression active at steps 3+.
    """
    return FOGReasonerConfig(
        vocab_size=vocab_size,
        d_model=320,
        n_heads=5,
        n_layers=4,
        d_ff=1344,
        max_seq_len=max_seq_len,
        dropout=dropout,
        initializer_range=0.02,
        latent_slots=4,
        reasoning_steps=reasoning_steps,
        compare_rank=80,
        planner_ff=1344,
        memory_slots=8,
        n_reasoning_modes=8,
        diversity_weight=1e-3,
        route_entropy_weight=0.0,
    )


def fog_binding_v2_10m_config(
    *,
    vocab_size: int = FOG_10M_VOCAB_SIZE,
    max_seq_len: int = 512,
    reasoning_steps: int = 4,
    dropout: float = 0.1,
) -> FOGReasonerConfig:
    """10M query-bound revision with an exact protected payload channel.

    This is a distinct architecture, not a silent reinterpretation of the
    released legacy weights.  It keeps the lexical 10M geometry, expands only
    the auxiliary workspace FFN, and uses one protected binding carrier plus
    direct first-token readout.
    """

    return FOGReasonerConfig(
        vocab_size=vocab_size,
        d_model=320,
        n_heads=5,
        n_layers=4,
        d_ff=1344,
        max_seq_len=max_seq_len,
        dropout=dropout,
        initializer_range=0.02,
        latent_slots=4,
        reasoning_steps=reasoning_steps,
        compare_rank=80,
        planner_ff=2330,
        memory_slots=4,
        n_reasoning_modes=8,
        diversity_weight=1e-3,
        route_entropy_weight=0.0,
        architecture_version="query_bound_v2",
        binding_mode="query_conditioned",
        readout_mode="direct_latent",
        protected_binding_slots=1,
        binding_offsets=(2,),
    )
