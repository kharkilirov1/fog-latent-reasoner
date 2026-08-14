from __future__ import annotations

import torch
from torch import nn

from train_real import finite_eval


class _SpySFTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.seen_reasoning_steps: list[int | None] = []

    def forward(self, prompt_ids, answer_ids_with_bos, *, reasoning_steps=None, **kwargs):
        self.seen_reasoning_steps.append(reasoning_steps)
        loss = self.anchor * 0 + torch.tensor(1.0, device=self.anchor.device)
        return loss, {
            "token_accuracy": torch.tensor(0.5, device=self.anchor.device),
        }


def test_finite_eval_forwards_requested_reasoning_depth():
    model = _SpySFTModel()
    batch = {
        "prompt_ids": torch.tensor([[4, 5, 6]]),
        "prompt_attention_mask": torch.tensor([[True, True, True]]),
        "answer_ids_with_bos": torch.tensor([[1, 7]]),
        "answer_attention_mask": torch.tensor([[True, True]]),
        "answer_labels": torch.tensor([[7]]),
    }
    metrics = finite_eval(
        model,
        [batch],
        device=torch.device("cpu"),
        precision="fp32",
        stage="sft",
        decoder_mode="full",
        bos_token_id=1,
        max_batches=1,
        reasoning_steps=2,
    )
    assert model.seen_reasoning_steps == [2]
    assert metrics["target_tokens"] == 1.0
