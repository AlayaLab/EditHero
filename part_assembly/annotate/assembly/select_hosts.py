"""Step 1c. Pick N diverse hosts from candidates.json: drop untextured and colourless hosts (host_colour_stats.json),
bucket the rest by a coarse category keyword in the object caption, then go round-robin over the buckets (more parts
first) so no category dominates. Writes the host list (PXFORM_ASSEMBLY_HOSTS, default <work>/hosts.txt).
The hosts are then looked at in preview_hosts.py renders; hosts that do not read as appealing game-style assets are
removed from the list by hand before planning.

    python select_hosts.py [N=150]
"""
import os, json, re, sys, collections
from paths import WORK, HOSTS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
CATS = [('vehicle', r'\b(car|truck|bus|van|jeep|tractor|tank|vehicle|motorcycle|bike|bicycle|scooter|cart|wagon|train|locomotive|plane|aircraft|airplane|helicopter|drone|boat|ship|submarine|rocket|spaceship|spacecraft|forklift|excavator)\b'),
        ('character', r'\b(character|man|woman|girl|boy|knight|warrior|soldier|figure|figurine|humanoid|person|doll|mascot|wizard|pirate|ninja|astronaut|elf|dwarf|zombie|hero|king|queen|princess)\b'),
        ('animal', r'\b(animal|dog|cat|bird|fish|rabbit|bunny|dragon|dinosaur|horse|cow|pig|bear|fox|wolf|lion|tiger|elephant|frog|turtle|owl|duck|chicken|penguin|monkey|deer|sheep|goat|crab|octopus|whale|shark|snake|insect|bee|butterfly|creature|monster)\b'),
        ('robot', r'\b(robot|mech|mecha|droid|android|cyborg|machine)\b'),
        ('furniture', r'\b(chair|table|desk|sofa|couch|bed|shelf|cabinet|drawer|stool|bench|wardrobe|dresser|bookcase|lamp|furniture)\b'),
        ('building', r'\b(house|building|cabin|hut|tower|castle|temple|shop|store|barn|windmill|lighthouse|shrine|tent|bridge|room|kitchen|scene)\b'),
        ('weapon', r'\b(sword|gun|rifle|pistol|cannon|bow|axe|hammer|spear|shield|weapon|blaster|turret)\b'),
        ('plant', r'\b(tree|plant|flower|cactus|mushroom|bush|bonsai|palm|grass|pot)\b'),
        ('food', r'\b(food|burger|cake|pizza|fruit|apple|ice cream|sushi|bread|drink|bottle|cup|mug|coffee)\b'),
        ('appliance', r'\b(camera|phone|computer|monitor|keyboard|microphone|speaker|radio|television|tv|console|printer|clock|watch|fan|oven|stove|fridge|washing|toaster|kettle|blender|drill)\b'),
        ('instrument', r'\b(guitar|piano|drum|violin|trumpet|instrument|keyboard)\b'),
        ('toy', r'\b(toy|lego|block|puzzle|game|chess)\b')]
c = json.load(open(os.path.join(WORK, 'candidates.json')))
cs_path = os.path.join(WORK, 'host_colour_stats.json')
if os.path.exists(cs_path):
    cs = json.load(open(cs_path))
    c = {o: s for o, s in c.items() if o not in cs or (cs[o]['textured'] > 0 and cs[o]['saturation'] >= 0.02)}
buckets = collections.defaultdict(list)
for oid, st in c.items():
    cap = st['caption'].lower(); cat = next((n for n, rx in CATS if re.search(rx, cap)), 'other'); buckets[cat].append((st['n'], oid))
for b in buckets.values():
    b.sort(reverse=True)
print({k: len(v) for k, v in sorted(buckets.items(), key=lambda x: -len(x[1]))})
order = sorted(buckets, key=lambda k: -len(buckets[k])); picked = []; seen = set()
while len(picked) < N and any(buckets.values()):
    for k in order:
        if buckets[k] and len(picked) < N:
            n, oid = buckets[k].pop(0); key = c[oid]['caption'].lower()[:40]
            if key in seen:
                continue
            seen.add(key); picked.append((k, oid))
open(HOSTS, 'w').write('\n'.join(o for _, o in picked) + '\n')
print(len(picked), 'picked:', dict(collections.Counter(k for k, _ in picked)))
