"""Register a host's slots from a released chain, so new chains can be built on the same host.

A chain's turn00_manifest.json lists the start-state nodes ('slot_<name>') and the library parts each is made of; nodes
named 'slot_anchor_*' are the fixed core. This writes them to the slot registry as one verified division
(granularity 'coarse'), optionally with per-slot roles ({slot: {"queries": [...], "rejects": [...]}}, see slot_registry.py).

    python tools/registry_from_manifest.py EditHero/data/recipes/G/02593609_s1/turn00_manifest.json [--roles roles.json]
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import slot_registry as SR

ap = argparse.ArgumentParser(); ap.add_argument('manifest'); ap.add_argument('--roles'); ap.add_argument('--granularity', default='coarse')
a = ap.parse_args()
m = json.load(open(a.manifest))
groups = {n[len('slot_'):]: v['pids'] for n, v in m['nodes'].items() if n.startswith('slot_')}
anchors = [g for g in groups if g.startswith('anchor')]
roles = json.load(open(a.roles)) if a.roles else {}
d = SR.put(m['host_oid'], a.granularity, groups, anchors=anchors, verified=True, roles=roles, method='manifest',
           notes=f'from {os.path.basename(os.path.dirname(a.manifest))}')
print(f"{m['host_oid']}: {d['n_slots']} slots {[g for g in groups if g not in anchors]}, anchors {anchors}")
