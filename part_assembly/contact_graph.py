"""Contact graph for components: determines which slots can be swapped and whether
the structure remains assembled after each swap.

This is the only place in the entire pipeline where object structure is determined,
and it is generic, not specific to any object class:

  Node = component, Edge = surface distance between two components < tau
  (using the Connectivity Accuracy threshold from the DGL paper;
  tau_c = 0.01, in normalized coordinates).

  **Articulation points = anchors that cannot be swapped.** Removing them
  disconnects the graph --- flower pot base, car chassis, lamp base,
  human torso. All are articulation points. The precise definition of
  "swapping it disconnects the structure" is exactly an articulation point,
  no need to write rules for each object class.

  **Non-articulation points = replaceable slots.** Removing them leaves
  the remainder connected, so swapping them won't leave other parts floating.
  Head, hands, feet, wheels, flowers, lamp shades all fall into this category.

  **After each swap, rebuilding the graph must still form one connected
  component** --- this is the decidable form of Articraft's NO FLOATING PARTS
  constraint.

Articulation points are found by "remove each and test connectivity", not using
Tarjan: we have only tens of components, this way is clearer and harder to
get wrong. Connectivity is checked directly using scipy.
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

CONTACT_TAU = 0.01          # Connectivity Accuracy threshold from DGL (NeurIPS 2020)
N_SAMPLE = 4000


def _samples(m, n=N_SAMPLE):
    if m is None or len(getattr(m, "faces", [])) == 0:
        return np.zeros((0, 3))
    # seed must be fixed: this feeds the "disassembly" gate (is_connected), without fixing
    # edge cases the test result flips each time --- chain_run.samples fixed the same issue,
    # this was a missed case.
    p, _ = trimesh.sample.sample_surface(m, n, seed=0)
    return np.asarray(p)


def build(pieces, diag, tau=CONTACT_TAU, n_sample=N_SAMPLE):
    """pieces: {slot_id: mesh}. Returns (key order, adjacency matrix bool, contact point count per edge)."""
    keys = list(pieces)
    pts = {k: _samples(pieces[k], n_sample) for k in keys}
    trees = {k: (cKDTree(pts[k]) if len(pts[k]) else None) for k in keys}
    n = len(keys)
    adj = np.zeros((n, n), bool)
    weight = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            if trees[keys[i]] is None or len(pts[keys[j]]) == 0:
                continue
            d, _ = trees[keys[i]].query(pts[keys[j]])
            c = int((d < tau * diag).sum())
            if c > 0:
                adj[i, j] = adj[j, i] = True
                weight[i, j] = weight[j, i] = c
    return keys, adj, weight


def n_components(adj, mask=None):
    """When mask is None, compute for the whole graph; otherwise only for nodes where mask is True."""
    idx = np.arange(len(adj)) if mask is None else np.where(mask)[0]
    if len(idx) == 0:
        return 0
    sub = adj[np.ix_(idx, idx)]
    return int(connected_components(coo_matrix(sub), directed=False)[0])


def articulation_points(keys, adj):
    """Articulation points: nodes whose removal increases the number of connected components.

    Isolated nodes (degree 0) are not articulation points, but their presence indicates
    the original object was not properly connected; they are returned separately.
    """
    n = len(keys)
    base = n_components(adj)
    cut, isolated = [], []
    for i in range(n):
        if adj[i].sum() == 0:
            isolated.append(keys[i])
            continue
        mask = np.ones(n, bool); mask[i] = False
        if n_components(adj, mask) > base:
            cut.append(keys[i])
    return cut, isolated


def replaceable_slots(pieces, diag, tau=CONTACT_TAU):
    """Returns (replaceable slots, anchor points = articulation points, isolated parts, number of connected components in the graph)."""
    keys, adj, _ = build(pieces, diag, tau)
    cut, isolated = articulation_points(keys, adj)
    bad = set(cut) | set(isolated)
    return [k for k in keys if k not in bad], cut, isolated, n_components(adj)


def bridges(adj):
    """Bridges: edges whose removal increases the number of connected components.

    The smaller side of a bridge is a **sub-assembly that can be removed as a whole** ---
    a head with eyes and ears, the entire head attached via the neck edge.
    Swapping the head should replace the whole assembly, not just the shell:
    if you only swap leaf nodes, each swap moves only a small part, rendering shows no change.
    """
    n = len(adj)
    base = n_components(adj)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i, j]:
                continue
            a2 = adj.copy(); a2[i, j] = a2[j, i] = False
            if n_components(a2) > base:
                out.append((i, j))
    return out


def _side_of(adj, i, j, keep):
    """The side of keep after removing edge (i, j) (set of node indices)."""
    a2 = adj.copy(); a2[i, j] = a2[j, i] = False
    lab = connected_components(coo_matrix(a2), directed=False)[1]
    return set(np.where(lab == lab[keep])[0])


def subassemblies(keys, adj, min_frac=0.05, max_frac=0.45, size=None):
    """Extract sub-assemblies that can be replaced as a whole, cut along bridges.

    size: {slot_id: weight} (by surface area or visible pixels), used to judge
    "is this part large enough". Returns [(mount point keys[i], sub-assembly keys list, fraction)],
    sorted by fraction descending.
    """
    n = len(adj)
    w = np.array([1.0 if size is None else float(size.get(k, 0.0)) for k in keys])
    tot = float(w.sum()) or 1.0
    out, seen = [], set()
    for i, j in bridges(adj):
        for a, b in ((i, j), (j, i)):
            side = _side_of(adj, i, j, b)          # side of b
            if len(side) == n:
                continue
            frac = float(w[list(side)].sum()) / tot
            if not (min_frac <= frac <= max_frac):
                continue
            key = frozenset(side)
            if key in seen:
                continue
            seen.add(key)
            out.append((keys[a], [keys[x] for x in sorted(side)], frac))
    out.sort(key=lambda r: -r[2])
    return out


def is_connected(pieces, diag, tau=CONTACT_TAU):
    """Whether the structure remains as one piece after one swap."""
    _, adj, _ = build(pieces, diag, tau)
    return n_components(adj) == 1
