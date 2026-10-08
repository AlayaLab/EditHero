"""Step 4. Assembly chain in the data engine's recipe format, from an assembly plan.

Input: <work>/plans/<oid>.json {"steps": [{"pids": [...], "name": "..."}, ...]} (step 0 = start state, every later step
= one turn that adds the host's own parts back at their original placement). Output in <work>/chains/<oid>/:
  turn00_manifest.json  start state = one anchor node holding the step-0 pids
  place_report.json     one 'add' record per part and turn (identity transform, source = the host itself), with the
                        engine's template instruction ("Add <step name>."), replaced later by the annotated one
  plan.json             host, object caption, steps
  provenance.json, checks.json   written by the engine's own writers (chain_run.write_provenance, assemble.sig)
  turnNN.glb            built and verified by tools/assemble.py
  img/turnNN/f000V.png  rendered by chain_run.stage_render (blender_kit, 4 views), unless --no-render

    python build_recipe.py <oid> [--no-render]
"""
import argparse, json, os, re, subprocess, sys
from paths import ENGINE, TEX_PART, CAPTIONS, WORK, chain_dir

I4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
ASSEMBLE = os.path.join(ENGINE, 'tools', 'assemble.py')


def slug(s, n=18):
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')[:n]


def assemble(out, verify):
    r = subprocess.run([sys.executable, ASSEMBLE, '--pv-root', TEX_PART, '--chain', out, '--out', out] + ([] if verify else ['--no-verify']),
                       capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-300:], flush=True)
    if r.returncode:
        sys.exit(1)


def package(out, host, rep):
    """provenance.json and checks.json with the engine's own writers."""
    import trimesh
    sys.path.insert(0, os.path.join(ENGINE, 'tools'))
    import chain_run, assemble as asm
    chain_run.write_provenance(host, rep, out)
    checks = {}
    for t in range(max([r['turn'] for r in rep] + [0]) + 1):
        sc = trimesh.load(f'{out}/turn{t:02d}.glb', process=False, file_type='glb')
        checks[f'turn{t:02d}'] = {n: asm.sig(g) for n, g in sc.geometry.items()}
    json.dump(checks, open(f'{out}/checks.json', 'w'), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('oid'); ap.add_argument('--no-render', action='store_true'); a = ap.parse_args()
    caps = json.load(open(CAPTIONS)).get(a.oid, {}); pc = caps.get('parts', {})
    steps = json.load(open(f'{WORK}/plans/{a.oid}.json'))['steps']; out = chain_dir(a.oid); os.makedirs(out, exist_ok=True)
    man = dict(host_oid=a.oid, dataset='PartVerseXL textured_part_glbs', concat='trimesh.util.concatenate in pid order',
               nodes={'slot_anchor_' + slug(steps[0].get('name', 'body'), 12): dict(pids=[str(p) for p in steps[0]['pids']], verified=True)})
    json.dump(man, open(f'{out}/turn00_manifest.json', 'w'), indent=1)
    rep = []
    for t, st in enumerate(steps[1:], 1):
        name = st.get('name', '') or ' and '.join(pc.get(str(p), ['part'])[0] for p in st['pids'])
        for p in st['pids']:
            p = str(p); cap = pc.get(p, [''])[0]
            rep.append(dict(turn=t, op='add', slot=f'add{t:02d}_{slug(cap or name)}_{p}', ds='pv', src=f'{a.oid[:8]}/{p}',
                            added_oid=a.oid, added_pid=p, transform=I4, scale=1.0, gap_after=0.0, size_ratio=1.0, added=cap, removed='',
                            instruction=f'Add {name}.' if len(st['pids']) > 1 else f'Add {cap.rstrip(".")}.'))
    json.dump(rep, open(f'{out}/place_report.json', 'w'), indent=1)
    json.dump(dict(host=a.oid, object=caps.get('object', [''])[0], steps=steps), open(f'{out}/plan.json', 'w'), indent=1)
    assemble(out, verify=False)
    package(out, a.oid, rep)
    assemble(out, verify=True)
    if a.no_render:
        return
    os.chdir(ENGINE); sys.argv = ['x']
    import chain_run
    chain_run.stage_render(out, f'{out}/img')


if __name__ == '__main__':
    main()
