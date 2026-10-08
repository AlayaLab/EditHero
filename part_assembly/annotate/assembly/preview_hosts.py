"""Step 1d. Preview renders of the hosts for screening by eye and for the planner. --concat (CPU) puts every textured
part of a host into <work>/preview/glb/<oid>.glb; --render (GPU) renders it with blender_kit (file-embedded materials,
4 views at azimuth 35/125/215/305, elevation 25) into <work>/preview/img/<oid>/f000V.png; --rank/--world shard the list.

    python preview_hosts.py --concat
    python preview_hosts.py --render [--rank R --world W]
"""
import argparse, json, os, subprocess, sys
from paths import TEX_PART, WORK, BLENDER, RENDER_SCRIPT, hosts

ap = argparse.ArgumentParser(); ap.add_argument('--concat', action='store_true'); ap.add_argument('--render', action='store_true')
ap.add_argument('--rank', type=int, default=0); ap.add_argument('--world', type=int, default=1); a = ap.parse_args()
out = os.path.join(WORK, 'preview'); H = hosts()
if a.concat:
    import trimesh
    os.makedirs(f'{out}/glb', exist_ok=True)
    for oid in H:
        dst = f'{out}/glb/{oid}.glb'
        if os.path.exists(dst):
            continue
        try:
            files = sorted([f for f in os.listdir(f'{TEX_PART}/{oid}') if f.endswith('.glb')], key=lambda s: int(s[:-4]) if s[:-4].isdigit() else 0)
            sc = trimesh.Scene()
            for i, f in enumerate(files):
                sc.add_geometry(trimesh.load(f'{TEX_PART}/{oid}/{f}', force='mesh', process=False), node_name=f'p{i}', geom_name=f'p{i}')
            sc.export(dst)
        except Exception as e:
            print('FAIL', oid, e, flush=True)
    print('concat done', flush=True)
if a.render:
    jobs = [dict(scene='mesh', mesh=f'{out}/glb/{oid}.glb', material='file_embedded', out_dir=f'{out}/img/{oid}')
            for oid in H[a.rank::a.world] if not os.path.exists(f'{out}/img/{oid}/f0003.png') and os.path.exists(f'{out}/glb/{oid}.glb')]
    if not jobs:
        print('nothing to render', flush=True); sys.exit(0)
    man = f'{out}/jobs_rank{a.rank}.jsonl'; open(man, 'w').write('\n'.join(json.dumps(j) for j in jobs) + '\n')
    subprocess.run([BLENDER, '-b', '--python', RENDER_SCRIPT, '--', '--manifest', man, '--trajectory', 'static', '--frames', '4',
                    '--start_az', '35', '--elevation', '25', '--distance', '1.7', '--res', '320', '--samples', '16', '--outputs', 'rgb',
                    '--normalize', 'whole'], check=False)
    print('render done rank', a.rank, flush=True)
