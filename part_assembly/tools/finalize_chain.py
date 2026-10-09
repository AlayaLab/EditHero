"""Turn a chain produced by chain_run.py into a self-contained recipe, in place, and verify it.

chain_run.py leaves place_report.json, provenance.json and turnNN.glb in the chain folder. This adds what the rebuild
needs:
  turn00_manifest.json   host and, per start-state node, the ordered library part ids it is made of (checked against
                         turn00.glb)
  checks.json            per turn and node: vertex count, sum of |v|, texture hashes (what assemble.py verifies)
  retex/turnNN_<node>.glb  the baked part of every retexture turn
and then rebuilds the chain from the recipe alone with assemble.py and compares it with the engine's own GLBs.

    python tools/finalize_chain.py <chain folder> [<chain folder> ...] [--hy3d-root DIR]
"""
import argparse, json, os, subprocess, sys
import numpy as np, trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from local_paths import DATA_ROOT
import slot_registry as SR
from assemble import sig

PV = os.path.join(DATA_ROOT, 'pv_textured', 'textured_part_glbs')


def host_oid(chain):
    for a in json.load(open(f'{chain}/provenance.json'))['assets']:
        if a['role'] == 'host':
            return a['objaverse_uid']
    raise RuntimeError('no host in provenance.json')


def manifest(chain, oid):
    """Every start-state node 'slot_<name>' is mapped to its library parts through the host's slot registry entries
    (any granularity), or to the part of the same id; the concatenation must match turn00.glb."""
    nodes = dict(trimesh.load(f'{chain}/turn00.glb', process=False, file_type='glb').geometry)
    groups = [e.get('groups', {}) for e in SR.get(oid, verified_only=False)]
    out, bad = {}, []
    for name, g in nodes.items():
        if not name.startswith('slot_'):
            continue
        s = name[len('slot_'):]
        pids = next(([str(p) for p in gr[s]] for gr in groups if s in gr), None)
        if pids is None and os.path.isfile(f'{PV}/{oid}/{s}.glb'):
            pids = [s]
        ok = False
        if pids:
            ms = [trimesh.load(f'{PV}/{oid}/{p}.glb', process=False, force='mesh') for p in pids if os.path.isfile(f'{PV}/{oid}/{p}.glb')]
            if ms:
                mm = trimesh.util.concatenate(ms) if len(ms) > 1 else ms[0]
                ok = (len(mm.vertices) == len(g.vertices)
                      and abs(float(np.abs(mm.vertices).sum()) - float(np.abs(np.asarray(g.vertices)).sum())) < 1e-2)
        out[name] = dict(pids=pids or [], verified=bool(ok))
        if not ok:
            bad.append(s)
    return dict(host_oid=oid, dataset='PartVerseXL textured_part_glbs', concat='trimesh.util.concatenate in pid order', nodes=out), bad


def finalize(chain, hy3d_root=None):
    rep = json.load(open(f'{chain}/place_report.json'))
    oid = host_oid(chain)
    man, bad = manifest(chain, oid)
    json.dump(man, open(f'{chain}/turn00_manifest.json', 'w'), indent=1, ensure_ascii=False)
    n_turns = max([r['turn'] for r in rep] + [0])
    checks = {}
    for t in range(n_turns + 1):
        sc = trimesh.load(f'{chain}/turn{t:02d}.glb', process=False, file_type='glb')
        checks[f'turn{t:02d}'] = {n: sig(g) for n, g in sc.geometry.items()}
    json.dump(checks, open(f'{chain}/checks.json', 'w'), indent=1, ensure_ascii=False)
    for r in rep:
        if (r.get('op') or 'replace') == 'retexture':
            t = r['turn']; key = r.get('glb_node') or ('slot_' + r['slot'])
            g = trimesh.load(f'{chain}/turn{t:02d}.glb', process=False, file_type='glb').geometry.get(key)
            if g is None:
                raise RuntimeError(f'turn {t}: retextured node {key} not in turn{t:02d}.glb')
            os.makedirs(f'{chain}/retex', exist_ok=True)
            s = trimesh.Scene(); s.add_geometry(g, node_name=key, geom_name=key); s.export(f'{chain}/retex/turn{t:02d}_{key}.glb')
    if bad:
        print(f'{chain}: start-state nodes without a verified part mapping: {bad} (add them to the slot registry)')
        return False
    out = f'{chain}/.rebuild_check'
    cmd = [sys.executable, os.path.join(HERE, 'assemble.py'), '--pv-root', PV, '--chain', chain, '--out', out]
    if hy3d_root:
        cmd += ['--hy3d-root', hy3d_root]
    r = subprocess.run(cmd, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or [r.stderr.strip()[-300:]])[-1]
    print(f'{chain}: {n_turns} turns; rebuild from the recipe: {last}')
    subprocess.run(['rm', '-rf', out])
    return r.returncode == 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('chains', nargs='+'); ap.add_argument('--hy3d-root'); a = ap.parse_args()
    ok = all([finalize(c.rstrip('/'), a.hy3d_root) for c in a.chains])
    sys.exit(0 if ok else 1)
