"""Integrity fixes for the immutable Logic-v7 protocol; no model downloads on import."""
from __future__ import annotations
import ast
import hashlib
import math
from pathlib import Path
import random
import re
import sys
import types
from typing import Sequence

REFERENCE_SHA256 = '25df3a2ce826e92fde977c7fde9a78b870865a47e556ed6acfd4ac7fa36a22ac'
REFERENCE_RELATIVE = 'artifacts/logic-v7-copy-bound-entity-roles/fog_logic_v7_final.py'

def load_reference(root: Path | None = None):
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    path = root / REFERENCE_RELATIVE
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != REFERENCE_SHA256:
        raise ValueError(f'Unexpected v7 reference hash: {digest}; refusing protocol drift')
    tree = ast.parse(source, filename=str(path))
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Try, ast.If))]
    mod = types.ModuleType('_fog_v7_reference')
    mod.__file__ = str(path)
    sys.modules[mod.__name__] = mod
    exec(compile(tree, str(path), 'exec'), mod.__dict__)
    return mod

def active_mask(instructions, halt_opcode: int) -> list[bool]:
    """Include the first HALT target, exclude every subsequent target."""
    out, active = [], True
    for ins in instructions:
        out.append(active)
        if ins.op == halt_opcode:
            active = False
    return out

def forbidden_holdout(ins, ref) -> bool:
    return (
        (ins.op == ref.OID['FOLLOW'] and ins.rel == ref.RID[ref.HELD_FOLLOW_REL])
        or (ins.op == ref.OID['COMPARE'] and ins.e1 == ref.EID[ref.HELD_COMPARE_ENTITY])
        or (ins.op == ref.OID['SELECT'] and (ins.e1, ins.e2) == tuple(ref.EID[x] for x in ref.HELD_SELECT_PAIR))
    )

def audit_training(programs, ref) -> dict:
    all_leaks, active_leaks, post_halt = [], [], 0
    for pi, p in enumerate(programs):
        mask = active_mask(p.instructions, ref.OID['HALT'])
        for si, (ins, use) in enumerate(zip(p.instructions, mask)):
            post_halt += not use
            if forbidden_holdout(ins, ref):
                rec = {'program': pi, 'step': si, 'opcode': ref.OPCODES[ins.op], 'after_halt': not use}
                all_leaks.append(rec)
                if use:
                    active_leaks.append(rec)
    return {'unmasked_holdout_targets': len(all_leaks), 'masked_holdout_targets': len(active_leaks),
            'post_halt_targets_excluded': post_halt, 'unmasked_details': all_leaks}

def derangement(items: Sequence, seed: int) -> dict:
    """A shuffled single cycle has no fixed points for every n > 1."""
    items = list(items)
    if len(set(items)) != len(items):
        raise ValueError('Derangement keys must be unique')
    if len(items) < 2:
        return {x: x for x in items}
    shuffled = items.copy()
    random.Random(seed).shuffle(shuffled)
    return {x: shuffled[(i+1) % len(shuffled)] for i, x in enumerate(shuffled)}

class LexicalSlots:
    """Copy identities; canonicalize lexical names, never semantic roles.

    Slot assignment is first-mention order. No opcode, relation, answer, or
    source/target label is accepted. Repeated occurrences share a payload.
    """
    def __init__(self, entities: Sequence[str], canonical_names: Sequence[str] | None = None):
        if not entities or any(not isinstance(x, str) or not x.strip() for x in entities) or len(set(entities)) != len(entities):
            raise ValueError('Entity names must be unique and nonempty')
        self.entities = tuple(entities)
        self.canonical_names = tuple(canonical_names) if canonical_names is not None else self.entities
        if not self.canonical_names or len(set(self.canonical_names)) != len(self.canonical_names):
            raise ValueError('Canonical aliases must be nonempty and unique')
        self.ids = {x: i for i, x in enumerate(entities)}
        self.pattern = re.compile(r'(?<!\w)(?:' + '|'.join(map(re.escape, sorted(entities, key=len, reverse=True))) + r')(?!\w)')
    def encode(self, text: str) -> tuple[str, tuple[int, ...], tuple[tuple[tuple[int, int], ...], ...]]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Expected nonempty instruction text')
        mapping, payloads, spans, parts = {}, [], [], []
        oldpos = newpos = 0
        for match in self.pattern.finditer(text):
            prefix = text[oldpos:match.start()]
            parts.append(prefix); newpos += len(prefix)
            name = match.group()
            if name not in mapping:
                mapping[name] = len(mapping)
                payloads.append(self.ids[name]); spans.append([])
            slot = mapping[name]
            if slot >= len(self.canonical_names):
                raise ValueError('Too many distinct mentions for the available canonical aliases')
            alias = self.canonical_names[slot]
            parts.append(alias)
            spans[slot].append((newpos, newpos+len(alias)))
            newpos += len(alias); oldpos = match.end()
        parts.append(text[oldpos:])
        return ''.join(parts), tuple(payloads), tuple(tuple(s) for s in spans)

def select_device(requested: str):
    import torch
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev = torch.device(requested)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; refusing silent CPU fallback')
    if dev.type not in ('cpu', 'cuda'):
        raise ValueError('This experiment supports cpu, cuda, cuda:N, or auto')
    return dev

def strict_gates(train, locked, phrases, shuffled, oracle, baseline) -> dict:
    """Every threshold comes from LOGIC_V7_PROTOCOL.md, not the old verdict."""
    def gate(value, threshold, op='ge'):
        ok = value is not None and math.isfinite(float(value))
        passed = ok and ((value >= threshold-1e-12) if op == 'ge' else abs(value-threshold) <= 1e-12)
        return {'value': value, 'required': threshold, 'comparison': op, 'passed': bool(passed)}
    held = phrases.get('heldout', {})
    result = {
        'balanced_baseline': gate(baseline, 1/12, 'eq'),
        'train_program': gate(train.get('accuracy'), .99),
        'locked_bind': gate(locked.get('bind_instruction_joint'), .90),
        'locked_instruction': gate(locked.get('instruction_joint'), .90),
        'locked_program': gate(locked.get('accuracy'), .75),
        'heldout_program': gate(locked.get('heldout_accuracy'), .70),
        'compare_kai': gate(held.get('compare_kai', {}).get('joint'), .90),
        'follow_calls': gate(held.get('follow_calls', {}).get('joint'), .95),
        'select_pair': gate(held.get('select_pair', {}).get('joint'), .80),
        'shuffle_drop': gate(locked.get('accuracy', 0)-shuffled.get('accuracy', 0), .30),
        'full_oracle': gate(oracle.get('accuracy'), 1., 'eq'),
    }
    return {'passed': all(x['passed'] for x in result.values()), 'criteria': result}
