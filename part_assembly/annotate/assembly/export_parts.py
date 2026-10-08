"""Step 2. Planner input of every host: <work>/hosts/<oid>/parts.json (pid, caption, centre, extent, area share,
articulation, contact neighbours) and a compact one-line-per-part table on stdout. Also reports the number of
contact-graph components (1 = every part touches something).

    python export_parts.py [--quiet]
"""
import argparse, json, os
import numpy as np, trimesh
import contact_graph as cg
from paths import TEX_PART, CAPTIONS, WORK, hosts


def export(oid, caps):
    pids = sorted([f[:-4] for f in os.listdir(f'{TEX_PART}/{oid}') if f.endswith('.glb')], key=int)
    pieces = {p: trimesh.load(f'{TEX_PART}/{oid}/{p}.glb', force='mesh', process=False) for p in pids}
    allv = np.vstack([g.vertices for g in pieces.values()]); lo, hi = allv.min(0), allv.max(0)
    diag = float(np.linalg.norm(hi - lo)); ctr = (lo + hi) / 2
    keys, adj = cg.build(pieces, diag)[:2]; A = np.asarray(adj.todense() if hasattr(adj, 'todense') else adj) > 0
    idx = {k: i for i, k in enumerate(keys)}; arts = set(cg.articulation_points(keys, adj)[0])
    tot = sum(float(g.area) for g in pieces.values()); ncomp = int(cg.n_components(adj))
    pc = caps.get(oid, {}).get('parts', {}); parts = []
    for p in pids:
        b = pieces[p].bounds; c = ((b[0] + b[1]) / 2 - ctr) / diag; e = (b[1] - b[0]) / diag
        parts.append(dict(pid=p, caption=pc.get(p, ['?'])[0], centre_xyz=[round(float(x), 2) for x in c], extent_xyz=[round(float(x), 2) for x in e],
                          area_share=round(float(pieces[p].area) / tot, 3), articulation=p in arts, touches=[keys[j] for j in np.where(A[idx[p]])[0]]))
    os.makedirs(f'{WORK}/hosts/{oid}', exist_ok=True)
    obj = caps.get(oid, {}).get('object', [''])[0]
    json.dump(dict(host=oid, object_caption=obj, components=ncomp, axes='y up, x left-right, z front-back; object-normalised, centre 0',
                   parts=parts), open(f'{WORK}/hosts/{oid}/parts.json', 'w'), indent=1)
    return parts, ncomp, obj


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--quiet', action='store_true'); a = ap.parse_args()
    caps = json.load(open(CAPTIONS))
    for oid in hosts():
        try:
            parts, ncomp, obj = export(oid, caps)
        except Exception as e:
            print(f'## {oid[:8]} FAILED {type(e).__name__}: {e}'); continue
        print(f'## {oid[:8]}  {obj[:80]}  | parts {len(parts)}, components {ncomp}')
        if a.quiet:
            continue
        for p in parts:
            c = p['centre_xyz']; e = p['extent_xyz']
            print(f"{p['pid']:>3} | {p['caption'][:60]:60s} | c({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}) e({e[0]:.2f},{e[1]:.2f},{e[2]:.2f}) "
                  f"a{p['area_share']:.2f}{' *' if p['articulation'] else '  '} | {','.join(p['touches'])}")
