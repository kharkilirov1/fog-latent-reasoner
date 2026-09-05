"""Copy-equivariant semantic reader and an exact hard-ST latent register machine."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence
import torch
from torch import nn
import torch.nn.functional as F
from integrity import LexicalSlots, forbidden_holdout

LAYERS = (0, 1, 2, 22, 23)
MAX_SLOTS = 4

@dataclass(frozen=True)
class TextRef:
    index: int
    payload: tuple[int, ...]

class TextBank:
    def __init__(self, entities: Sequence[str], canonical_names: Sequence[str] | None = None):
        self.lexical = LexicalSlots(entities, canonical_names=canonical_names)
        self.texts, self.spans, self.index = [], [], {}
        self.features = self.mask = self.mentions = self.valid = None
    def add(self, text: str) -> TextRef:
        canonical, payload, spans = self.lexical.encode(text)
        if len(payload) > MAX_SLOTS:
            raise ValueError(f'More than {MAX_SLOTS} distinct entity mentions: {text}')
        if canonical not in self.index:
            if self.features is not None:
                raise RuntimeError('Cannot add texts to a featurized bank')
            self.index[canonical] = len(self.texts)
            self.texts.append(canonical); self.spans.append(spans)
        return TextRef(self.index[canonical], payload)
    @torch.no_grad()
    def featurize(self, backbone, tokenizer, device, batch=12, max_tokens=64):
        if not getattr(tokenizer, 'is_fast', False):
            raise ValueError('A fast tokenizer with character offsets is required')
        if backbone.config.num_hidden_layers < max(LAYERS):
            raise ValueError('Backbone does not have the protocol-selected layers')
        n, d = len(self.texts), backbone.config.hidden_size
        feat = torch.zeros(n, len(LAYERS), max_tokens, d, dtype=torch.float16)
        mask = torch.zeros(n, max_tokens, dtype=torch.bool)
        mentions = torch.zeros(n, MAX_SLOTS, max_tokens, dtype=torch.bool)
        valid = torch.zeros(n, MAX_SLOTS, dtype=torch.bool)
        for start in range(0, n, batch):
            texts = self.texts[start:start+batch]
            enc = tokenizer(texts, padding=True, truncation=False, add_special_tokens=False,
                            return_tensors='pt', return_offsets_mapping=True)
            offsets = enc.pop('offset_mapping')
            if enc['input_ids'].size(1) > max_tokens:
                raise ValueError('Instruction exceeds the token limit; no silent truncation is allowed')
            enc = {k: v.to(device) for k, v in enc.items()}
            result = backbone.model(**enc, use_cache=False, output_hidden_states=True, return_dict=True)
            for i in range(len(texts)):
                length = int(enc['attention_mask'][i].sum())
                if length == 0:
                    raise ValueError('Tokenizer produced an empty instruction')
                mask[start+i, :length] = True
                for k, layer in enumerate(LAYERS):
                    feat[start+i, k, :length] = result.hidden_states[layer][i, :length].cpu().half()
                for slot, occurrences in enumerate(self.spans[start+i]):
                    for a, b in occurrences:
                        overlap = (offsets[i, :length, 1] > a) & (offsets[i, :length, 0] < b)
                        if not overlap.any():
                            raise ValueError('An entity mention is missing from tokenizer offsets')
                        mentions[start+i, slot, :length] |= overlap
                    valid[start+i, slot] = True
        self.features = feat.to(device)
        self.mask = mask.to(device)
        self.mentions = mentions.to(device)
        self.valid = valid.to(device)
        return self
    def get(self, ids: torch.Tensor):
        f, m = self.features[ids], self.mask[ids]
        length = int(m.sum(-1).max())
        return f[:, :, :length], m[:, :length], self.mentions[ids, :, :length], self.valid[ids]

class JointRoleReader(nn.Module):
    """Both roles attend to the whole instruction and share a joint assignment."""
    def __init__(self, ref, d: int, width: int = 96):
        super().__init__()
        self.mix = ref.HeadLayerMixer(list(LAYERS), 22)
        self.project = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, width), nn.GELU())
        self.context = nn.TransformerEncoderLayer(width, 4, 2*width, dropout=0.0,
                                                  batch_first=True, norm_first=True)
        self.role = nn.Sequential(nn.Linear(3*width, width), nn.GELU(), nn.Linear(width, 2))
        self.pair_q = nn.Linear(width, 32, bias=False)
        self.pair_k = nn.Linear(width, 32, bias=False)
        self.register_buffer('position', self._position(64, width), persistent=False)
    @staticmethod
    def _position(n, d):
        x = torch.zeros(n, d)
        phase = torch.arange(n).float()[:, None] * torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.)/d))
        x[:, 0::2] = phase.sin(); x[:, 1::2] = phase.cos()
        return x * .1
    def forward(self, f, mask, mentions, valid):
        x = self.project(self.mix.ctx(f.float()))
        x = x + self.position[:x.size(1)]
        x = self.context(x, src_key_padding_mask=~mask)
        mm = mentions.float()
        slots = torch.einsum('bmt,btd->bmd', mm, x)/mm.sum(-1, keepdim=True).clamp_min(1)
        sentence = (x*mask[..., None]).sum(1)/mask.sum(1, keepdim=True).clamp_min(1)
        sent = sentence[:, None].expand_as(slots)
        role = self.role(torch.cat((slots, sent, slots*sent), -1))
        joint = role[:, :, 0, None] + role[:, None, :, 1]
        joint = joint + torch.einsum('bmd,bnd->bmn', self.pair_q(slots), self.pair_k(slots))/math.sqrt(32)
        permitted = valid.clone()
        permitted[~valid.any(1), 0] = True
        return joint.masked_fill(~(permitted[:, :, None] & permitted[:, None, :]), -1e4)

class SemanticReader(nn.Module):
    def __init__(self, ref, d, op_proto, relation_proto):
        super().__init__()
        self.op_mix = ref.HeadLayerMixer(list(LAYERS), 1)
        self.rel_mix = ref.HeadLayerMixer(list(LAYERS), 0)
        self.op = ref.SentencePrototypeBinder(2*d, op_proto, 128)
        self.rel = ref.PrototypeBinder(d, relation_proto, 96)
        self.roles = JointRoleReader(ref, d)
    def forward(self, f, mask, mentions, valid):
        return (self.op(f, mask, self.op_mix),
                self.rel(self.rel_mix.ctx(f.float()), mask, self.rel_mix),
                self.roles(f, mask, mentions, valid))
    def mix_reg(self):
        return sum((x.w()-x.prior).square().sum() for x in (self.op_mix, self.rel_mix, self.roles.mix))

def st_onehot(logits: torch.Tensor) -> torch.Tensor:
    soft = torch.softmax(logits.float(), -1)
    hard = F.one_hot(soft.argmax(-1), soft.size(-1)).float()
    return hard + (soft-soft.detach())

def slot_payload(refs: Sequence[TextRef], entities: int, device) -> torch.Tensor:
    out = torch.zeros(len(refs), MAX_SLOTS, entities, device=device)
    out[:, 0, 0] = 1.0
    for i, r in enumerate(refs):
        if r.payload:
            out[i].zero_()
            for j, entity in enumerate(r.payload):
                out[i, j, entity] = 1.0
    return out

def hard_arguments(op_logits, relation_logits, pair_logits, payload):
    op, rel = st_onehot(op_logits), st_onehot(relation_logits)
    pair = st_onehot(pair_logits.flatten(-2)).reshape_as(pair_logits)
    e1 = torch.einsum('...m,...me->...e', pair.sum(-1), payload)
    e2 = torch.einsum('...m,...me->...e', pair.sum(-2), payload)
    return op, e1, rel, e2

def execute_tensor(op, e1, relation, e2, *, return_trace=False):
    """Shared recurrence; no labels or oracle program enter the machine.

    Missing reads preserve the old state, matching v7, but are also reported.
    """
    batch, steps, _ = op.shape
    entities, relations = e1.size(-1), relation.size(-1)
    cur = F.one_hot(torch.zeros(batch, dtype=torch.long, device=op.device), entities).float()
    memory = torch.zeros(batch, relations, entities, entities, device=op.device)
    pred = torch.zeros(batch, device=op.device)
    halted = torch.zeros(batch, device=op.device)
    invalid = torch.zeros(batch, device=op.device)
    trace = []
    for t in range(steps):
        o, a, r, b = op[:, t], e1[:, t], relation[:, t], e2[:, t]
        active = 1-halted
        follow_raw = torch.einsum('br,bs,brsu->bu', r, cur, memory)
        mass = follow_raw.sum(-1)
        exists = (mass.detach() > .5).float() + (mass-mass.detach())
        follow = follow_raw + (1-exists[:, None])*cur
        selected = pred[:, None]*a + (1-pred[:, None])*b
        pl, pf, ps = (active*o[:, k] for k in (1, 2, 4))
        next_cur = cur*(1-pl-pf-ps)[:, None] + pl[:, None]*a + pf[:, None]*follow + ps[:, None]*selected
        pc = active*o[:, 3]
        next_pred = pred*(1-pc)+pc*(cur*a).sum(-1)
        key = r[:, :, None]*a[:, None, :]
        pb = (active*o[:, 0])[:, None, None, None]
        memory = memory*(1-pb*key[..., None]) + pb*key[..., None]*b[:, None, None, :]
        invalid = invalid + pf*(1-exists)
        halted = halted+active*o[:, 5]
        cur, pred = next_cur, next_pred
        if return_trace:
            trace.append((cur.clone(), memory.clone(), pred.clone(), halted.clone()))
    return {'current': cur, 'halted': halted, 'predicate': pred, 'memory': memory,
            'invalid_reads': invalid, 'trace': trace}

def final_loss(state, targets):
    onehot = F.one_hot(targets, state['current'].size(-1)).float()
    return ((state['current']-onehot).square().sum(-1)+.25*(state['halted']-1).square()).mean()

def forward_refs(model, bank, refs):
    device = next(model.parameters()).device
    ids = torch.tensor([r.index for r in refs], device=device)
    unique, inverse = ids.unique(return_inverse=True)
    logits = model(*bank.get(unique))
    return tuple(x[inverse] for x in logits)

def auxiliary_loss(logits, rows, refs, use, ref):
    op, rel, pair = logits
    device = op.device
    keep = [i for i, active in enumerate(use) if active]
    if not keep:
        return op.sum()*0
    if any(forbidden_holdout(rows[i], ref) for i in keep):
        raise AssertionError('Held-out supervised target reached the training loss')
    loss = F.cross_entropy(op[keep], torch.tensor([rows[i].op for i in keep], device=device))
    rid = [i for i in keep if rows[i].rel >= 0]
    if rid:
        loss = loss+F.cross_entropy(rel[rid], torch.tensor([rows[i].rel for i in rid], device=device))
    lp = F.log_softmax(pair.flatten(1), -1).reshape_as(pair)
    terms = []
    for i in keep:
        ins = rows[i]
        if ins.e1 >= 0:
            try:
                a = refs[i].payload.index(ins.e1)
                if ins.e2 >= 0:
                    b = refs[i].payload.index(ins.e2)
                    terms.append(-2*lp[i, a, b])
                else:
                    terms.append(-torch.logsumexp(lp[i, a], -1))
            except ValueError as exc:
                raise ValueError('Supervised identity is absent from lexical payload') from exc
    if terms:
        loss = loss+torch.stack(terms).mean()
    return loss
