"""Step 3 (check). Validate the assembly plan written by the planning agent (prompts/plan_assembly.md):
every pid of the host appears exactly once, no unknown pid, step 0 first.

    python check_plan.py <oid> [<oid> ...]        # prints "ok <steps> steps <parts> parts" or the problem
"""
import json, os, sys
from paths import WORK

bad = 0
for oid in sys.argv[1:]:
    plan = json.load(open(f'{WORK}/plans/{oid}.json'))
    parts = [q['pid'] for q in json.load(open(f'{WORK}/hosts/{oid}/parts.json'))['parts']]
    used = [str(p) for s in plan['steps'] for p in s['pids']]
    missing, extra = sorted(set(parts) - set(used), key=int), sorted(set(used) - set(parts))
    dup = sorted({p for p in used if used.count(p) > 1})
    if missing or extra or dup or len(plan['steps']) < 2:
        bad += 1; print(f'{oid}: missing {missing} unknown {extra} duplicated {dup} steps {len(plan["steps"])}')
    else:
        print(f'ok {len(plan["steps"])} steps {len(used)} parts')
sys.exit(1 if bad else 0)
