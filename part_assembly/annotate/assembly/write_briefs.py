"""Step 5. <work>/chains/<oid>/brief.json for every built chain: what the annotators (the Qwen draft, the revising and
rewriting agents) need: host caption, the plan's turns with the parts each adds (caption, centre, extent, contacts),
the parts already present, and the renders to look at.

    python write_briefs.py
"""
import json, os
from paths import WORK, hosts, chain_dir, chain_id

n = 0
for oid in hosts():
    d = chain_dir(oid)
    if not os.path.exists(f'{d}/plan.json'):
        continue
    pj = json.load(open(f'{WORK}/hosts/{oid}/parts.json')); steps = json.load(open(f'{d}/plan.json'))['steps']
    parts = {p['pid']: p for p in pj['parts']}; last = len(steps) - 1
    def part(pid):
        p = parts[pid]; return dict(pid=pid, caption=p['caption'], centre_xyz=p['centre_xyz'], extent_xyz=p['extent_xyz'], touches=p['touches'])
    turns = []; present = list(steps[0]['pids'])
    for t, st in enumerate(steps[1:], 1):
        turns.append(dict(n=t, step_name=st['name'], adds=[part(p) for p in st['pids']],
                          already_present=[dict(pid=p, caption=parts[p]['caption'][:60]) for p in present], image_after=f'{d}/img/turn{t:02d}/f0000.png'))
        present += st['pids']
    brief = dict(chain=chain_id(oid), host_oid=oid, host_caption=pj['object_caption'],
                 axes='y up; x and z horizontal; coordinates are in the object frame, centre 0, units = fraction of the object size. Which horizontal direction is the object FRONT is not given: decide it from the images.',
                 camera='Each image is a Blender render of the state after that turn, auto-framed. f0000..f0003 orbit the object at azimuth 35, 125, 215, 305 degrees, elevation 25. Use the four final-state views to find the object front and its own left/right before writing anything.',
                 start_state=dict(name=steps[0]['name'], parts=[part(p) for p in steps[0]['pids']], image=f'{d}/img/turn00/f0000.png'),
                 final_state_views=[f'{d}/img/turn{last:02d}/f{v:04d}.png' for v in range(4)], turns=turns)
    json.dump(brief, open(f'{d}/brief.json', 'w'), indent=1); n += 1
print('briefs', n)
