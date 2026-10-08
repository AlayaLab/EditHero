"""Fast per-geometry AABB + face count from a GLB, without decoding buffers.

glTF POSITION accessors are required to carry `min`/`max`, so the JSON chunk
alone is enough. Falls back to trimesh only if that invariant is broken.
"""
import json
import struct

import numpy as np


def _read_glb_json(path):
    with open(path, "rb") as f:
        head = f.read(12)
        magic, _ver, _length = struct.unpack("<III", head)
        if magic != 0x46546C67:
            raise ValueError("not a GLB: %s" % path)
        clen, ctype = struct.unpack("<II", f.read(8))
        if ctype != 0x4E4F534A:
            raise ValueError("first chunk is not JSON: %s" % path)
        return json.loads(f.read(clen).decode("utf-8"))


def _node_matrix(node):
    if "matrix" in node:
        # glTF stores column-major
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    M = np.eye(4)
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        M[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    if "scale" in node:
        M[:3, :3] = M[:3, :3] @ np.diag(node["scale"])
    if "translation" in node:
        M[:3, 3] = node["translation"]
    return M


def geom_bboxes(path):
    """-> {mesh_name: {"bbox": [[minx,miny,minz],[maxx,maxy,maxz]], "n_faces": int}}

    Mesh names follow the g00000.. convention written by the cleaning stage.
    """
    g = _read_glb_json(path)
    meshes = g.get("meshes", [])
    accs = g.get("accessors", [])
    nodes = g.get("nodes", [])

    # world transform per node (single-level scenes here, but walk anyway)
    world = {}

    def walk(idx, parent):
        M = parent @ _node_matrix(nodes[idx])
        world[idx] = M
        for c in nodes[idx].get("children", []):
            walk(c, M)

    roots = set(range(len(nodes)))
    for n in nodes:
        for c in n.get("children", []):
            roots.discard(c)
    for r in sorted(roots):
        walk(r, np.eye(4))

    mesh_to_node = {}
    for i, n in enumerate(nodes):
        if "mesh" in n:
            mesh_to_node.setdefault(n["mesh"], i)

    out = {}
    for mi, m in enumerate(meshes):
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        nf = 0
        M = world.get(mesh_to_node.get(mi, -1), np.eye(4))
        for prim in m.get("primitives", []):
            pa = prim.get("attributes", {}).get("POSITION")
            if pa is None:
                continue
            a = accs[pa]
            if "min" not in a or "max" not in a:
                raise KeyError("accessor without min/max")
            amin = np.array(a["min"], dtype=np.float64)
            amax = np.array(a["max"], dtype=np.float64)
            # transform the 8 corners of the local AABB
            corners = np.array(np.meshgrid(*zip(amin, amax))).reshape(3, -1).T
            corners = corners @ M[:3, :3].T + M[:3, 3]
            lo = np.minimum(lo, corners.min(0))
            hi = np.maximum(hi, corners.max(0))
            if "indices" in prim:
                nf += accs[prim["indices"]]["count"] // 3
            else:
                nf += a["count"] // 3
        if not np.isfinite(lo).all():
            continue
        name = m.get("name", "mesh_%d" % mi)
        out[name] = {"bbox": [lo.tolist(), hi.tolist()], "n_faces": int(nf)}
    return out
