"""Step 1b. Colour statistics of candidate hosts (all part textures pooled): mean HSV saturation and brightness of the
texture pixels and the number of textured parts. Hosts with no texture (textured == 0) or colourless grey textures
(saturation < 0.02) are dropped by select_hosts.py.

    python host_colour_stats.py [--hosts <work>/candidates.json] [--out <work>/host_colour_stats.json]
"""
import argparse, json, os, numpy as np
from multiprocessing import Pool
from paths import TEX_PART, CAPTIONS, WORK


def stats(oid):
    import trimesh
    sats, vals, textured, parts = [], [], 0, 0
    for f in sorted(os.listdir(f'{TEX_PART}/{oid}')):
        if not f.endswith('.glb'):
            continue
        parts += 1
        try:
            g = trimesh.load(f'{TEX_PART}/{oid}/{f}', force='mesh', process=False); vis = g.visual
            img = getattr(getattr(vis, 'material', None), 'baseColorTexture', None)
            if img is None and hasattr(vis, 'material') and getattr(vis.material, 'image', None) is not None:
                img = vis.material.image
            if img is None:
                continue
            textured += 1
            hsv = np.asarray(img.convert('RGB').resize((64, 64)).convert('HSV'), dtype=np.float32) / 255.
            sats.append(hsv[..., 1].mean()); vals.append(hsv[..., 2].mean())
        except Exception:
            pass
    return oid, dict(parts=parts, textured=textured, saturation=float(np.mean(sats)) if sats else 0.0,
                     brightness=float(np.mean(vals)) if vals else 1.0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--hosts', default=os.path.join(WORK, 'candidates.json'))
    ap.add_argument('--out', default=os.path.join(WORK, 'host_colour_stats.json')); a = ap.parse_args()
    oids = list(json.load(open(a.hosts))) if a.hosts.endswith('.json') else [l.strip() for l in open(a.hosts) if l.strip()]
    caps = json.load(open(CAPTIONS)); out = {}
    with Pool(16) as p:
        for i, (oid, st) in enumerate(p.imap_unordered(stats, oids)):
            out[oid] = dict(st, caption=caps.get(oid, {}).get('object', [''])[0])
            if i % 50 == 0:
                print(i, flush=True)
    json.dump(out, open(a.out, 'w'), indent=1); print('done', len(out))
