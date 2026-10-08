"""Render one state of a chain with the chain's fixed four-view rig, the way the renders in the data package were made.

    python scripts/render_state.py --data /path/to/EditHero/data --chain A/1a3267df_s0 --turn 1 --out out/A_1a3267df_s0_turn01

Needs Blender 4.2 (PXFORM_BLENDER, default `blender`) and blender_kit (PXFORM_RENDER_SCRIPT = path to its scripts/render.py,
https://github.com/AuroraRyan0301/Blender-Visualization-Skill). Writes f0000.png ... f0003.png: f0000 is the conditioning view
(view0_cond in the data package), f0001-f0003 are the held-out views."""
import argparse, json, os, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True, help='data/ directory of the data package')
ap.add_argument('--chain', required=True)
ap.add_argument('--turn', type=int, required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--glb', default=None, help='render this GLB instead of the package state (e.g. an edited result)')
a = ap.parse_args()
fr = json.load(open(os.path.join(a.data, 'cameras.json')))[a.chain]['frame']
glb = a.glb or os.path.join(a.data, 'states', a.chain, f'turn{a.turn:02d}.glb')
blender = os.environ.get('PXFORM_BLENDER', 'blender')
kit = os.environ.get('PXFORM_RENDER_SCRIPT', './third_party/blender_kit/scripts/render.py')
cmd = [blender, '-b', '--python', kit, '--', '--hdri', 'studio.exr', '--hdri_strength', '1.6',
       '--scene', 'mesh', '--mesh', glb, '--material', 'file_embedded', '--normalize', fr['normalize'],
       '--trajectory', 'circle', '--frames', '4', '--start_az', str(fr['start_az']), '--elevation', str(fr['elevation']),
       '--distance', str(fr['distance']), '--frame_center', *map(str, fr['frame_center']), '--frame_diag', str(fr['frame_diag']),
       '--res', '420', '--samples', '24', '--outputs', 'rgb', '--out_dir', a.out]
sys.exit(subprocess.call(cmd))
