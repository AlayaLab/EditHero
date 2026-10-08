"""Step 1a. Screen PartVerse-XL objects as hosts of assembly chains: 8-25 textured parts, one connected contact
component, largest part < 60% of the surface, no giant meshes, not already a host of another chain.
Writes <work>/candidates.json (oid -> stats).

    python screen_hosts.py [--sample 900] [--exclude-chains EditHero/data/chains.json]
"""
import argparse, json, os, random
from multiprocessing import Pool
from paths import TEX_PART, CAPTIONS, WORK


def stats(oid):
    import numpy as np, trimesh, contact_graph as cg
    try:
        pids = sorted([f[:-4] for f in os.listdir(f'{TEX_PART}/{oid}') if f.endswith('.glb')], key=int)
        if sum(os.path.getsize(f'{TEX_PART}/{oid}/{p}.glb') for p in pids) > 60e6:
            return oid, None
        pieces = {p: trimesh.load(f'{TEX_PART}/{oid}/{p}.glb', force='mesh', process=False) for p in pids}
        if sum(len(g.faces) for g in pieces.values()) > 600000:
            return oid, None
        allv = np.vstack([g.vertices for g in pieces.values()]); diag = float(np.linalg.norm(allv.max(0) - allv.min(0)))
        keys, adj = cg.build(pieces, diag)[:2]; tot = sum(float(g.area) for g in pieces.values())
        return oid, dict(n=len(pids), components=int(cg.n_components(adj)), largest=max(float(g.area) for g in pieces.values()) / tot,
                         faces=sum(len(g.faces) for g in pieces.values()))
    except Exception:
        return oid, None


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--sample', type=int, default=900)
    ap.add_argument('--exclude-chains', help="chains.json whose host_uid's are not used again"); ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    caps = json.load(open(CAPTIONS))
    used = {c['host_uid'] for c in json.load(open(a.exclude_chains))} if a.exclude_chains else set()
    pool = [oid for oid in os.listdir(TEX_PART) if oid not in used and oid in caps
            and 8 <= len([f for f in os.listdir(f'{TEX_PART}/{oid}') if f.endswith('.glb')]) <= 25]
    random.seed(0); random.shuffle(pool); sample = pool[:a.sample]
    print('unused hosts with 8-25 parts:', len(pool), 'screening', len(sample), flush=True)
    out = {}
    with Pool(a.workers) as p:
        for i, (oid, st) in enumerate(p.imap_unordered(stats, sample, chunksize=4)):
            if st and st['components'] == 1 and st['largest'] < 0.6:
                out[oid] = dict(st, caption=caps[oid]['object'][0])
            if i % 100 == 0:
                print(i, 'done,', len(out), 'kept', flush=True)
    os.makedirs(WORK, exist_ok=True)
    json.dump(out, open(os.path.join(WORK, 'candidates.json'), 'w'), indent=1); print('kept', len(out), 'of', len(sample))
