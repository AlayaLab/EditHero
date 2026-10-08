#!/usr/bin/env python3
"""Step 8a. Left/right audit of the instructions. Given the declared front axis (+z/-z/+x/-x, parsed from the
'front' statement), the object's own left axis follows (front +z -> left +x, front -z -> left -x, front +x -> left -z,
front -x -> left +z). For every turn whose text says left or right and adds parts on one lateral side, check the word
against the parts' coordinates. Reports per version (Qwen draft / agent revision) and writes <work>/lr_audit.json.

    python lr_audit.py
"""
import os, json, re
from paths import WORK, hosts as host_list, chain_dir
LEFT = {'+z': ('x', +1), '-z': ('x', -1), '+x': ('z', -1), '-x': ('z', +1)}
def front_axis(s):
    s = s.lower()
    if 'no intrinsic' in s: return None
    m = re.search(r'(?:faces?|facing|toward|towards|points?|pointing)\s*(?:the\s+)?(?:direction\s+)?(?:of\s+)?(?:the\s+)?([+-]|positive|negative|minus|plus)?\s*([xz])(?:[\s-]*axis)?', s)
    if not m: m = re.search(r'([+-]|positive|negative)\s*([xz])\b', s)
    if not m: return None
    sign, ax = m.group(1) or '+', m.group(2); sign = '-' if sign in ('-', 'negative', 'minus') else '+'
    return sign + ax
def words(t):
    t = t.lower(); l = bool(re.search(r'\bleft\b', t)); r = bool(re.search(r'\bright\b', t)); return l, r


def main():
    hosts = host_list()
    report = {}; tot = {'qwen': [0, 0], 'final': [0, 0]}; disagree = []
    for oid in hosts:
        d = chain_dir(oid); b = json.load(open(f'{d}/brief.json')); q = json.load(open(f'{d}/instructions_qwen.json')); f = json.load(open(f'{d}/instructions_final.json'))
        fq, ff = front_axis(q['front']), front_axis(f['front'])
        if fq and ff and fq != ff: disagree.append((oid[:8], fq, ff))
        rows = []
        for ver, ins, fr in (('qwen', q, fq), ('final', f, ff)):
            if fr is None: continue
            ax, sgn = LEFT[fr]; ai = 0 if ax == 'x' else 2
            for k, t in enumerate(ins['turns']):
                l, r = words(t['text'])
                if not (l ^ r): continue          # no side word, or both sides (a pair): nothing to check
                cs = [p['centre_xyz'][ai] * sgn for p in b['turns'][k]['adds']]   # +: on the object's left
                if max(abs(c) for c in cs) < 0.04: continue
                if all(c > 0.04 for c in cs) or all(c < -0.04 for c in cs):
                    expect_left = cs[0] > 0; ok = (l and expect_left) or (r and not expect_left)
                    tot[ver][0] += 1; tot[ver][1] += (not ok)
                    if not ok: rows.append(dict(version=ver, turn=k + 1, says='left' if l else 'right', parts_on='left' if expect_left else 'right', text=t['text']))
        report[oid] = dict(front_qwen=fq, front_final=ff, problems=rows)
    json.dump(dict(totals=tot, front_disagree=disagree, chains=report), open(WORK + '/lr_audit.json', 'w'), ensure_ascii=False, indent=1)
    print('checked turns / wrong side word:  qwen %d/%d   revision %d/%d' % (tot['qwen'][0], tot['qwen'][1], tot['final'][0], tot['final'][1]))
    print('front axis unparsed: qwen', sum(1 for r in report.values() if r['front_qwen'] is None), 'revision', sum(1 for r in report.values() if r['front_final'] is None))
    print('front axis disagree (qwen vs revision):', len(disagree), disagree[:12])
    bad_h = [(o[:8], len([p for p in r['problems'] if p['version'] == 'final'])) for o, r in report.items() if any(p['version'] == 'final' for p in r['problems'])]
    bad_q = [(o[:8], len([p for p in r['problems'] if p['version'] == 'qwen'])) for o, r in report.items() if any(p['version'] == 'qwen' for p in r['problems'])]
    print('chains with wrong-side words: revision', len(bad_h), bad_h[:20]); print('chains with wrong-side words: qwen ', len(bad_q), bad_q[:20])


if __name__ == '__main__':
    main()
