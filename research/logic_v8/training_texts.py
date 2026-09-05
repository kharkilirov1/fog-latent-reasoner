"""Versioned reachability repair of the already-declared TRAIN phrase inventory.

No SCAN/TEST template is read by augmentation. Test rendering is unchanged.
"""
from integrity import active_mask, forbidden_holdout

def declared_training_texts(ins, ref):
    if forbidden_holdout(ins, ref):
        raise ValueError('Cannot construct training text for a held-out operation')
    op = ref.OPCODES[ins.op]
    a = ref.ENTITIES[ins.e1] if ins.e1 >= 0 else None
    b = ref.ENTITIES[ins.e2] if ins.e2 >= 0 else None
    r = ref.RELATIONS[ins.rel] if ins.rel >= 0 else None
    templates = list(ref.TRAIN_TEMPLATES[op])
    if op == 'BIND': templates.extend(ref.REL_BIND_TRAIN[r])
    if op == 'FOLLOW': templates.extend(ref.REL_FOLLOW_TRAIN[r])
    return tuple(dict.fromkeys(t.format(a=a, b=b, r=r) for t in templates))

def augment_declared(spec, ref, train_programs):
    """Restore intended paraphrases for labels already present in training.

    This CHANGES training data; it is not an isolated architecture comparison.
    """
    labels = {(r[1], r[2], r[3], r[4]) for r in spec.train_examples}
    for program in train_programs:
        for ins, used in zip(program.instructions, active_mask(program.instructions, ref.OID['HALT'])):
            if used: labels.add((ins.op, ins.e1, ins.rel, ins.e2))
    original_manifest = spec.manifest()
    rows = []
    for fields in sorted(labels):
        ins = ref.Instr(*fields)
        for text in declared_training_texts(ins, ref):
            rows.append((spec.add(text), *fields))
    spec.train_examples = rows
    return {'variant': 'all_declared_train_templates', 'original_manifest': original_manifest,
            'supervised_instruction_types': len(labels), 'supervised_phrase_examples': len(rows),
            'added_test_templates': 0, 'description': 'Repair parity-unreachable TRAIN templates; active training labels only.'}
