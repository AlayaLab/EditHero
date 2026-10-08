"""Step 11. Write the annotated per-turn instructions into the chain recipes: every record of a turn gets the turn's
instruction / instruction_zh, and instruction_original keeps the engine's template sentence (the same convention as
every other family). The source is one instruction file per chain (default instructions_relational.json); turns
flagged by qwen_check.py and decided by a person can be given in a decisions file
{"<oid>": {"<n>": {"text": ..., "zh": ...}}}, which overrides the source for those turns.

    python apply_instructions.py [--source instructions_relational.json] [--decisions decisions.json]
"""
import argparse, json, os
from paths import hosts, chain_dir

ap = argparse.ArgumentParser(); ap.add_argument('--source', default='instructions_relational.json'); ap.add_argument('--decisions')
a = ap.parse_args()
dec = json.load(open(a.decisions)) if a.decisions else {}
n_ok = n_missing = 0
for oid in hosts():
    d = chain_dir(oid); src = f'{d}/{a.source}'
    if not os.path.exists(src):
        n_missing += 1; continue
    by_n = {int(t['n']): dict(text=t['text'], zh=t['zh']) for t in json.load(open(src))['turns']}
    for n, t in dec.get(oid, {}).items():
        by_n[int(n)] = t
    rep = json.load(open(f'{d}/place_report.json')); changed = False
    for r in rep:
        t = by_n.get(int(r['turn']))
        if not t:
            print('no instruction for', oid[:8], 'turn', r['turn']); continue
        if r.get('instruction') != t['text'] or r.get('instruction_zh') != t['zh']:
            r.setdefault('instruction_original', r['instruction']); r['instruction'] = t['text']; r['instruction_zh'] = t['zh']; changed = True
    if changed:
        json.dump(rep, open(f'{d}/place_report.json', 'w'), indent=1)
    n_ok += 1
print(f'instructions applied to {n_ok} chains; {n_missing} chains without {a.source}')
