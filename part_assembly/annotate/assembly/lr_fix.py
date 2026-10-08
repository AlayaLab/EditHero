#!/usr/bin/env python3
"""Step 8b. Deterministic left/right correction on top of the agent revision. Only chains where the Qwen draft and the
revision agree on the front axis are touched. For a turn whose added parts all lie on one lateral side and whose
English text uses exactly one of left/right, the side word is set from the coordinates (front +z -> object's left is
+x, front -z -> -x, front +x -> -z, front -x -> +z); 左/右 in the Chinese sentence are swapped accordingly.
Writes <chain>/instructions_checked.json (text_pre_lr/zh_pre_lr keep the revised sentence). Chains with a front
disagreement or an unparsable front are copied unchanged and listed in <work>/lr_pending.json.

    python lr_fix.py
"""
import os, json, re, sys
from lr_audit import front_axis, LEFT
from paths import WORK, hosts as host_list, chain_dir
hosts = host_list()
def swap_en(t): return re.sub(r'\b(left|right|Left|Right)\b', lambda m: {'left': 'right', 'right': 'left', 'Left': 'Right', 'Right': 'Left'}[m.group(1)], t)
def swap_zh(t): return t.replace('左', '\0').replace('右', '左').replace('\0', '右')
pending = []; n_fix = n_chain = 0
for oid in hosts:
    d = chain_dir(oid); b = json.load(open(f'{d}/brief.json')); q = json.load(open(f'{d}/instructions_qwen.json')); f = json.load(open(f'{d}/instructions_final.json'))
    fq, ff = front_axis(q['front']), front_axis(f['front']); out = dict(f); out['lr_checked'] = False
    if fq and ff and fq == ff:
        ax, sgn = LEFT[ff]; ai = 0 if ax == 'x' else 2; out['lr_checked'] = True; out['front_axis'] = ff; out['left_axis'] = ('+' if sgn > 0 else '-') + ax; fixed = 0
        for k, t in enumerate(out['turns']):
            en = t['text']; l = bool(re.search(r'\bleft\b', en, re.I)); r = bool(re.search(r'\bright\b', en, re.I))
            if not (l ^ r): continue
            cs = [p['centre_xyz'][ai] * sgn for p in b['turns'][k]['adds']]
            if not (all(c > 0.04 for c in cs) or all(c < -0.04 for c in cs)): continue
            expect_left = cs[0] > 0
            if (l and expect_left) or (r and not expect_left): continue
            t['text_pre_lr'] = t['text']; t['zh_pre_lr'] = t['zh']; t['text'] = swap_en(t['text']); t['zh'] = swap_zh(t['zh'])
            t['lr_fixed'] = True; t['reason'] = (t.get('reason') or '') + f' | left/right set from coordinates ({out["left_axis"]} is the object\'s left)'; fixed += 1
        n_fix += fixed; n_chain += 1
    else:
        pending.append(dict(oid=oid, front_qwen=q['front'], front_revision=f['front'], parsed=[fq, ff]))
    json.dump(out, open(f'{d}/instructions_checked.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
json.dump(pending, open(WORK + '/lr_pending.json', 'w'), ensure_ascii=False, indent=1)
print(f'{n_chain} chains checked, {n_fix} turns had their side word corrected; {len(pending)} chains pending (front unclear): ' + ' '.join(p['oid'][:8] for p in pending))
