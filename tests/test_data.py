from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import fog_lmw.data as data
from fog_lmw.data import (
    BPETokenizer,
    IGNORE_INDEX,
    PromptResponseCollator,
    SpecialTokens,
    TextBlockCollator,
    extract_gsm8k_final,
    iter_jsonl_records,
    iter_local_records,
    iter_text_records,
    load_hf_dataset,
    parse_prompt_response,
)


class ToyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        ids = [4 + (ord(char) % 64) for char in text if not char.isspace()]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            if add_eos:
                ids[-1] = self.eos_token_id
        return ids


def test_local_txt_and_jsonl_iterators(tmp_path: Path):
    text_path = tmp_path / "stories.txt"
    text_path.write_text("first story\n\n second story \n", encoding="utf-8")
    assert list(iter_text_records(text_path)) == [
        {"text": "first story"},
        {"text": " second story "},
    ]

    jsonl_path = tmp_path / "qa.jsonl"
    rows = [
        {"question": "one?", "answer": "1"},
        {"question": "two?", "answer": "2"},
    ]
    jsonl_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    assert list(
        iter_jsonl_records(
            jsonl_path, required_fields=("question", "answer")
        )
    ) == rows
    assert list(iter_local_records([text_path, jsonl_path])) == [
        {"text": "first story"},
        {"text": " second story "},
        *rows,
    ]


def test_jsonl_errors_include_source_line(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text": "ok"}\nnot-json\n', encoding="utf-8")
    records = iter_jsonl_records(path)
    assert next(records) == {"text": "ok"}
    with pytest.raises(ValueError, match=r"bad\.jsonl at line 2"):
        next(records)


def test_hf_backend_forwards_revision_and_streaming(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_load_dataset(dataset_id, **kwargs):
        calls.append((dataset_id, kwargs))
        return {"sentinel": True}

    fake_module = SimpleNamespace(load_dataset=fake_load_dataset)
    real_import = data.import_module

    def fake_import(name: str):
        if name == "datasets":
            return fake_module
        return real_import(name)

    monkeypatch.setattr(data, "import_module", fake_import)
    result = load_hf_dataset(
        "roneneldan/TinyStories",
        config="default",
        split="validation",
        revision="abc123",
        streaming=True,
        token="secret",
    )
    assert result == {"sentinel": True}
    assert calls == [
        (
            "roneneldan/TinyStories",
            {
                "name": "default",
                "split": "validation",
                "streaming": True,
                "revision": "abc123",
                "token": "secret",
            },
        )
    ]


def test_optional_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch):
    def missing(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(data, "import_module", missing)
    with pytest.raises(ImportError, match=r"pip install datasets"):
        load_hf_dataset("example/data")
    with pytest.raises(ImportError, match=r"pip install tokenizers"):
        BPETokenizer.load("unused.json")


def test_gsm8k_full_and_final_parsing():
    record = {
        "question": "Natalia sold 48 clips and then 24. How many?",
        "answer": (
            "She sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72"
        ),
    }
    assert extract_gsm8k_final(record["answer"]) == "72"
    full = parse_prompt_response(
        record, prompt_template="Question: {prompt}\nAnswer:"
    )
    final = parse_prompt_response(record, target_mode="final")
    assert full.prompt.startswith("Question: Natalia")
    assert full.response.endswith("#### 72")
    assert final.prompt == record["question"]
    assert final.response == "72"

    with pytest.raises(ValueError, match="missing final-answer marker"):
        extract_gsm8k_final("The result is 72")
    with pytest.raises(ValueError, match="target_mode"):
        parse_prompt_response(record, target_mode="unsupported")  # type: ignore[arg-type]


def test_text_block_collator_uses_hf_labels_and_masks_padding():
    tokenizer = ToyTokenizer()
    collator = TextBlockCollator(tokenizer, block_size=4)
    batch = collator([{"text": "ab"}, {"text": "c"}])

    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"].shape == (2, 4)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    # HF convention: labels equal input IDs; causal_lm_loss applies the one
    # and only shift (logits[:-1] versus labels[1:]).
    assert torch.equal(batch["labels"][0], batch["input_ids"][0])
    shifted_targets = batch["labels"][:, 1:]
    expected_next_tokens = batch["input_ids"][:, 1:]
    active = batch["attention_mask"][:, 1:].bool()
    assert torch.equal(shifted_targets[active], expected_next_tokens[active])
    assert batch["labels"][1].tolist()[-1] == IGNORE_INDEX
    assert batch["input_ids"][1, -1].item() == tokenizer.pad_token_id


def test_prompt_response_collator_masks_prompt_bos_and_padding():
    tokenizer = ToyTokenizer()
    collator = PromptResponseCollator(
        tokenizer,
        target_mode="final",
        max_prompt_length=32,
        max_answer_length=8,
    )
    examples = [
        {"question": "1+1?", "answer": "work\n#### 2"},
        {"question": "10+5?", "answer": "longer work\n#### 15"},
    ]
    batch = collator(examples)

    expected_keys = {
        "input_ids",
        "attention_mask",
        "labels",
        "prompt_ids",
        "prompt_attention_mask",
        "answer_ids_with_bos",
        "answer_attention_mask",
        "answer_labels",
    }
    assert set(batch) == expected_keys
    assert all(isinstance(value, torch.Tensor) for value in batch.values())
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["prompt_ids"].shape == batch["prompt_attention_mask"].shape
    assert batch["answer_ids_with_bos"].shape == batch["answer_labels"].shape

    for row in range(len(examples)):
        prompt_length = int(batch["prompt_attention_mask"][row].sum())
        answer_length = int(batch["answer_attention_mask"][row].sum())
        assert batch["labels"][row, : prompt_length + 1].eq(IGNORE_INDEX).all()
        first_response = batch["answer_ids_with_bos"][row, 1]
        assert batch["labels"][row, prompt_length + 1].eq(first_response)
        assert batch["answer_labels"][row, 0].item() == IGNORE_INDEX
        assert batch["answer_ids_with_bos"][row, 0].item() == tokenizer.bos_token_id
        assert (
            batch["answer_ids_with_bos"][row, answer_length - 1].item()
            == tokenizer.eos_token_id
        )
        assert batch["labels"][row, batch["attention_mask"][row].eq(0)].eq(
            IGNORE_INDEX
        ).all()


def test_bpe_train_save_load_roundtrip(tmp_path: Path):
    pytest.importorskip("tokenizers")
    texts = [
        "Once upon a time there was a small fox.",
        "The small fox found a bright red ball.",
        "Once upon a time the fox shared the ball.",
    ] * 4
    tokenizer = BPETokenizer.train(
        texts,
        vocab_size=300,
        min_frequency=1,
        show_progress=False,
        length=len(texts),
    )
    assert tokenizer.pad_token_id != tokenizer.bos_token_id
    assert tokenizer.eos_token_id != tokenizer.unk_token_id
    encoded = tokenizer.encode("small fox", add_bos=True, add_eos=True)
    assert encoded[0] == tokenizer.bos_token_id
    assert encoded[-1] == tokenizer.eos_token_id

    path = tokenizer.save(tmp_path / "tokenizer.json")
    loaded = BPETokenizer.load(path)
    assert loaded.encode("small fox", add_bos=True, add_eos=True) == encoded
    assert "small" in loaded.decode(encoded)


def test_offline_byte_fallback_is_lossless_and_exact_size(tmp_path: Path):
    pytest.importorskip("tokenizers")
    tokenizer = BPETokenizer.byte_fallback(vocab_size=8192)
    text = "Hello, мир — 你好!"
    encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert tokenizer.vocab_size == 8192
    assert encoded[0] == tokenizer.bos_token_id
    assert encoded[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(encoded) == text
    loaded = BPETokenizer.load(tokenizer.save(tmp_path / "fallback.json"))
    assert loaded.decode(loaded.encode(text)) == text


def test_special_tokens_must_be_distinct():
    with pytest.raises(ValueError, match="must be distinct"):
        SpecialTokens(pad="<x>", bos="<x>").as_list()
