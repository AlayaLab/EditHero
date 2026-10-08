"""Run one assembly chain. Split into three stages, each in its own environment (see PIPELINE.md).

  --stage place   articraft environment: compute initial pose → snap_to finalization → export GLB for each round
  --stage check   trellis2 environment: dual grid surface voxel intersection, check for penetration
  --stage render  any environment: invoke blender_kit to produce 4 viewpoints + contact close-up

Initial pose is a **deterministic single computation by rule**, not a search:

  1. Measure the contact surface between original part and host: contact centroid c, outward direction d, install section radius r.
  2. Rotate new part's principal axis to align with original part's axis (choose the direction that points the "install end" toward host).
  3. Scale uniformly so new part's install-end cross-section radius equals r—"cap fits over head", rim width irrelevant.
  4. Translate to place new part's install-end centroid at c.
  5. `snap_to` closes remaining gap. Throw `SnapRefused` if initial value is wrong; report and I will fix it,
     **do not enlarge max_move**.

Fix initial values by inspecting render, not by expanding search range—Articraft's snap_to comment is clear:
"a large required move means the piece is constrained or misplaced".
"""
import os, sys, json, random, argparse, math
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contact_graph as CG


def phrase(c):
    """Compress a description into a noun phrase that fits into an instruction.

    The removed part originally used the complete description **including the trailing period**, resulting in
      "Replace A gray helmet …, serving as head protection for the warrior figure.
       with the side shield of …"
    The period is stuck in the middle and reads as a run-on. Qwen3's descriptions are long sentences, exposing this—
    Qwen2.5's were short and acceptable by comparison.
    """
    from build_replace_chains import head_clause
    t = head_clause(str(c)).strip().rstrip(". ").strip()
    return (t[0].lower() + t[1:]) if t else t

from local_paths import DATA_ROOT as D
ANNO = f"{D}/partverse_anno/anno_infos"
CONTACT_TAU = 0.01          # DGL's Connectivity Accuracy threshold
# Maximum "fragment gap" allowed within a candidate part (as fraction of whole object diagonal). See max_component_gap.
DEBRIS_TAU = 0.02
N_REST, N_PART = 20000, 3000
from local_paths import HY3D_MESH_ROOT, HY3D_PART_ROOTS
_HY3D_SCENES = {}


def part_glb_size(cand):
    """Get candidate GLB file size (bytes) without loading. Huge library parts (600k faces, ~1GB unpacked) load alone
    strains serve to 95G and glibc doesn't return memory (W3-25 hit twice); screen by file size before load.
    HY3D per-object scenes are scored by object; single parts typically small, use conservative limit."""
    ds = cand.get("ds", "pv")
    try:
        if ds == "hy3d":
            oid = cand["oid"]
            for root in HY3D_PART_ROOTS:
                baked = f"{root}/{oid}/{int(cand['pid'])}.glb"
                if os.path.isfile(baked):
                    return os.path.getsize(baked)
            return os.path.getsize(f"{HY3D_MESH_ROOT}/{oid}/mesh.glb") // 8
        return os.path.getsize(os.path.join(D, "pv_textured",
                                            f"textured_part_glbs/{cand['oid']}/{cand['pid']}.glb"))
    except OSError:
        return 0


def part_mesh(cand):
    """Load part mesh based on candidate source library. pv = PartVerseXL per-part GLB; hy3d = geometry named g%05d
    from mesh_clean multi-geometry GLB (one geometry per part, with material). HY3D scenes cached by oid."""
    ds = cand.get("ds", "pv")
    if ds == "hy3d":
        oid = cand["oid"]
        # Prefer projection-baked textured parts (full bake writes to OSS, pilot wrote to CPFS); fall back to pure geometry if unavailable
        for root in HY3D_PART_ROOTS:
            baked = f"{root}/{oid}/{int(cand['pid'])}.glb"
            if os.path.isfile(baked):
                return trimesh.load(baked, process=False, force="mesh")
        if oid not in _HY3D_SCENES:
            if len(_HY3D_SCENES) > 8:
                _HY3D_SCENES.clear()
            _HY3D_SCENES[oid] = trimesh.load(f"{HY3D_MESH_ROOT}/{oid}/mesh.glb", process=False)
        sc = _HY3D_SCENES[oid]
        name = f"g{int(cand['pid']):05d}"
        g = sc.geometry.get(name)
        if g is None:
            raise FileNotFoundError(f"hy3d {oid} has no geometry {name}")
        m = g.copy()
        try:
            T, _ = sc.graph.get(name)
            m.apply_transform(T)
        except Exception:
            pass
        return m
    return trimesh.load(os.path.join(D, "pv_textured",
                                     f"textured_part_glbs/{cand['oid']}/{cand['pid']}.glb"),
                        process=False, force="mesh")

# Camera framing: each round uses minimum bounding sphere of current state, view cone exactly tangent
# (r = R / sin(half_fov)), no longer normalized by bounding box. Visibility gate compares two images from same viewpoint:
# after turn k's viewpoint is set, render turn k-1's state with turn k's viewpoint (img/turnKK_prev),
# else viewpoint change causes whole object to scale/pan in frame, every pixel counts as "changed".
FRAME_FOCAL, FRAME_SENSOR = 50.0, 36.0     # blender_kit default focal / Blender default sensor width
_HALF_FOV = math.atan(FRAME_SENSOR / 2.0 / FRAME_FOCAL)
# kit camera distance: r = frame_diag × distance, frame_diag passes 2R
FRAME_DIST = 1.0 / math.sin(_HALF_FOV) / 2.0


# Panorama camera settings shared across all renders (suite passed from RenderServer / stage_render to blender_kit),
# recorded in frames.json with each round's framing; read directly when re-rendering chain at uniform distance later.
CAM_BASE = dict(trajectory="circle", frames=4, start_az=35.0, elevation=25.0,
                focal_mm=FRAME_FOCAL, sensor_mm=FRAME_SENSOR, res=420, samples=24,
                hdri="studio.exr", hdri_strength=1.6, material="file_embedded")


def record_frame(chain_dir, key, fr, mesh):
    """Append panorama render viewpoint to <chain_dir>/frames.json (key = turnNN / turnNN_prev)."""
    p = os.path.join(chain_dir, "frames.json")
    d = json.load(open(p)) if os.path.isfile(p) else {"camera": CAM_BASE, "frames": {}}
    d["camera"] = CAM_BASE
    d["frames"][key] = dict(mesh=os.path.basename(mesh),
                            center_zup=fr["frame_center"], sphere_radius=fr["frame_diag"] / 2.0,
                            frame_diag=fr["frame_diag"], distance_factor=fr["distance"],
                            camera_distance=fr["frame_diag"] * fr["distance"],
                            normalize=fr["normalize"])
    tmp = f"{p}.{os.getpid()}.tmp"          # Include pid to avoid collisions in multi-process rendering
    try:
        json.dump(d, open(tmp, "w"), indent=1)
        os.replace(tmp, p)
    except OSError:
        pass                                  # Frame recording is convenience data; not worth crashing render for


def _frames_ok(dd, n=4, min_bytes=1024):
    """Blender worker occasionally dies mid-run: missing 1 of 4 frames or PNG half-written. Checking only final frame
    lets incomplete batches through; _grid_labeled later crashes on missing frames (W3-5 hit 4 times)."""
    for fi in range(n):
        fp = os.path.join(dd, f"f{fi:04d}.png")
        if not os.path.isfile(fp) or os.path.getsize(fp) < min_bytes:
            return False
    return True


def _jdump(obj, path, **kw):
    """Atomic JSON write: temp file in same directory + fsync + os.replace (machine auto-restarts; truncated files don't heal; review #4/#7)."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, **kw)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _jload(path, default=None):
    """Resilient JSON read: move truncated/corrupted files to .corrupt and return default, don't crash serve."""
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, OSError):
        try:
            os.replace(path, path + ".corrupt")
        except OSError:
            pass
        return default


def _used_persist(out_dir, key, pairs):
    """Persist failed candidates to disk: when stage_run throws, in-memory used set is lost,
    retry pulls same batch again unchanged (review #6)."""
    if not pairs:
        return
    pp = os.path.join(out_dir, "ckpt", "used_extra.json")
    os.makedirs(os.path.dirname(pp), exist_ok=True)
    d = _jload(pp, {}) or {}
    cur = {tuple(x) for x in d.get(key, [])}
    cur.update(pairs)
    d[key] = sorted([list(x) for x in cur])
    _jdump(d, pp)


def sphere_frame(mesh):
    """Framing parameters for a state GLB (job keys for blender_kit).

    GLB is y-up; blender_kit uses bpy's glTF importer which converts to z-up, so sphere center
    transforms (x, y, z) -> (x, -z, y).
    """
    # Only geometry determines the camera. force="mesh" also packs every
    # material into a texture atlas, which can take minutes for a single frame.
    scene = trimesh.load(mesh, process=False)
    if isinstance(scene, trimesh.Scene):
        V = np.vstack([
            trimesh.transform_points(scene.geometry[geom].vertices, transform)
            for node in scene.graph.nodes_geometry
            for transform, geom in [scene.graph[node]]
        ]).astype(np.float64, copy=False)
    else:
        V = np.asarray(scene.vertices, np.float64)
    try:
        c, r = trimesh.nsphere.minimum_nsphere(V)
        if not (np.all(np.isfinite(c)) and np.isfinite(r) and r > 0):
            raise RuntimeError("nsphere degenerate")
    except Exception:
        # qhull crashes on coplanar/sparse vertex meshes (QhullError is RuntimeError subclass, caught by serve's broad
        # except and kills whole op; review #2 hit 14 chains): fall back to bbox's circumsphere, lose only minor framing margin
        lo, hi = V.min(0), V.max(0)
        c = (lo + hi) / 2.0
        r = float(np.linalg.norm(hi - lo) / 2.0) or 1e-3
    return dict(normalize="none",
                frame_center=[float(c[0]), float(-c[2]), float(c[1])],
                frame_diag=float(2.0 * r), distance=FRAME_DIST)


# ------------------------------------------------------------------ Shared

def grouped_pieces(oid, groups_path, anchor_prefix="anchor"):
    """Merge parts by group into "slots"—a slot is one whole piece (head, left arm, left leg), not one face.

    Grouping is manual (see group_tool.py), not algorithmically cut: pure geometry bridges miss arms connecting
    to torso via two edges, pure labels contradict. After merge each slot is large enough on-screen to see
    when swapped—before, swapping single parts per-slot, 9 of 20 rounds showed <1% screen change.

    Return ({slot_name: mesh}, diagonal, anchor_slot_names).
    """
    pieces, diag, seg = object_pieces(oid)
    groups = groups_path if isinstance(groups_path, dict) else json.load(open(groups_path))
    out, anchors = {}, set()
    for g, members in groups.items():
        ms = [pieces[m] for m in members if m in pieces]
        if not ms:
            continue
        out[g] = trimesh.util.concatenate(ms) if len(ms) > 1 else ms[0]
        if g.startswith(anchor_prefix):
            anchors.add(g)
    return out, diag, anchors


def object_pieces(oid):
    """Return {slot_id: original mesh(with texture)} and whole-object diagonal.

    Assemble host from per-part GLBs in textured_part_glbs, not by cutting _segmented.glb—
    testing shows **face-identical** (same face count, bounds), but only first has materials.
    Cutting with segmented results in all-gray render across the chain.
    """
    seg = trimesh.load(f"{ANNO}/{oid}/{oid}_segmented.glb", process=False, force="mesh")
    info = json.load(open(f"{ANNO}/{oid}/{oid}_info.json"))
    faceids, labels = info["ordered_faceid"], info["ordered_face_label"]
    pieces = {}
    for r, p in enumerate(labels):
        if r >= len(faceids) or not faceids[r]:
            continue
        tp = f"{D}/pv_textured/textured_part_glbs/{oid}/{p}.glb"
        pieces[str(p)] = (trimesh.load(tp, process=False, force="mesh")
                          if os.path.isfile(tp) else seg.submesh([faceids[r]], append=True))
    diag = float(np.linalg.norm(seg.bounds[1] - seg.bounds[0]))
    return pieces, diag, seg


def max_component_gap(mesh):
    """The most isolated piece of this part, distance from rest (in world units, as fraction of object diagonal).

    Retrieved "parts" may carry floating debris. Bicycle round 3 swapped "basketball hoop ring"
    came with detached bracket, hanging by front fork—**passed all five geometric gates** because contact
    uses nearest-point distance; nearest was ring, bracket unmanaged.

    Criterion is **spatial gap** not component count. Multi-shell (gun from unsoldered shells, "five-claw set")
    is topologically several, visually one thing, should not block. First version checked component count, reported 5 of 20 turns,
    measuring gaps all 0.0000, all false positives.

    Also **must merge vertices by distance first**: GLB vertices don't share; splitting by shared vertices gives
    "block count ≈ face count" (measured 2546).
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    m = mesh.copy()
    m.merge_vertices()
    f = np.asarray(m.faces)
    n = len(m.vertices)
    if n == 0 or len(f) == 0:
        return 0.0
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    k, lab = connected_components(g, directed=False)
    if k <= 1:
        return 0.0
    V = np.asarray(m.vertices)
    # Building complement KD-tree per component is O(k*n log n): sprinkle covered in doughnuts has hundreds of components,
    # hangs serve 35-58 minutes solid (W3-8 / W3-13 reported). Switch to single tree + cross-label nearest-neighbor:
    # sample first, then for each point find first different-label point in 32-neighbors; only components where all 32-neighbors
    # are same-label check precisely against complement (truly isolated fragments, usually 1-2).
    if len(V) > 20000:
        idx = np.random.default_rng(0).choice(len(V), 20000, replace=False)
        V, lab = V[idx], lab[idx]
    tree = cKDTree(V)
    kq = min(32, len(V))
    dist, nbr = tree.query(V, k=kq, workers=-1)
    diff = lab[nbr] != lab[:, None]
    first = np.where(diff.any(1), diff.argmax(1), -1)
    cross = np.where(first >= 0, dist[np.arange(len(V)), np.clip(first, 0, kq - 1)], np.inf)
    out = 0.0
    for i in np.unique(lab):
        sel = lab == i
        gi = float(cross[sel].min())
        if not np.isfinite(gi):        # 32 neighbors all own: check complement precisely for this block
            gi = float(cKDTree(V[~sel]).query(V[sel])[0].min())
        out = max(out, gi)
    return out


def samples(m, n):
    if m is None or len(getattr(m, "faces", [])) == 0:
        return np.zeros((0, 3))
    # Must pass seed explicitly: new trimesh sample_surface uses its own Generator,
    # np.random.seed doesn't reach it. Without fixed seed, contact surface/install radius jitter
    # round-to-round, edge gates flip—tested same seed twice, different candidates chosen at round 12.
    # veto / yaw_fix's "deterministic replay of rest of chain" rests on this line.
    p, _ = trimesh.sample.sample_surface(m, n, seed=0)
    return np.asarray(p)


def principal_axis(pts):
    if len(pts) < 3:
        return np.array([0.0, 1.0, 0.0])
    c = pts - pts.mean(0)
    _, v = np.linalg.eigh(c.T @ c)
    return v[:, -1] / (np.linalg.norm(v[:, -1]) + 1e-12)


def rot_between(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def contact_frame(part, rest, diag):
    """How the original part contacts the host. Returns (c, d, r, contact point count)."""
    from scipy.spatial import cKDTree
    rp, pp = samples(rest, N_REST), samples(part, N_PART)
    if len(rp) == 0 or len(pp) == 0:
        return None, None, None, 0
    dist, _ = cKDTree(rp).query(pp)
    hit = pp[dist < CONTACT_TAU * diag]
    if len(hit) < 10:
        return None, None, None, 0
    c = hit.mean(0)
    d = pp.mean(0) - c
    n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
    rel = hit - c
    perp = rel - np.outer(rel @ d, d)
    return c, d, max(float(np.percentile(np.linalg.norm(perp, axis=1), 90)), 1e-4), len(hit)


def mount_end(pts, d, frac=0.25):
    """The innermost 25% of points along d—that's the 'install end'. Returns (centroid, cross-section radius)."""
    t = pts @ d
    sel = pts[t <= np.quantile(t, frac)]
    if len(sel) < 5:
        sel = pts
    c = sel.mean(0)
    rel = sel - c
    perp = rel - np.outer(rel @ d, d)
    return c, max(float(np.percentile(np.linalg.norm(perp, axis=1), 90)), 1e-4)


UP = np.array([0.0, 1.0, 0.0])      # These GLBs have Y pointing up (Objaverse convention)

# Replaced part's overall size bounds relative to **original slot**. See explanation in initial_pose.
SIZE_LO, SIZE_HI = 0.75, 2.0
# plane fit uses separate upper bound: fitting footprint into support footprint expands narrow-crown hanging-brim hats
# to fill old brim circle (C4: 10 rejected hats all at 1.7–2.0x, swallowing adjacent bottle); all passing canopies on fish-cart ≤ 1.0x.
PLANE_SIZE_HI = 1.25


def rot_about(axis, ang):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def initial_pose(q, part, rest, diag, frame=None, slot_diag=None, yaw_extra=0.0, side=None):
    """Compute initial value by rule, no search.

    **Rotate around vertical axis only, no arbitrary rotation.** Before, rotating new part's principal axis to match slot's principal axis
    flipped cups, helmet fronts faced arms—these pieces have inherent "which side up, which side front"; arbitrary rotation ruins it.

      * Vertical: preserve new part's orientation from its original object (all GLBs are Y-up, so keep as-is),
        cup with mouth up stays mouth-up when swapped.
      * Horizontal: rotate around vertical axis so new part's "front" faces same direction as swapped part—
        use horizontal component of slot's outward direction d as reference, helmet front faces body front not arm.
    """
    c, d, r_slot, n_hit = frame if frame and frame[0] is not None else contact_frame(part, rest, diag)
    if c is None:
        return None, dict(note="Original part has no contact with host, cannot get contact surface")

    qp0 = samples(q, N_PART)
    q2 = q.copy()
    q2.apply_translation(-q2.vertices.mean(0))

    mdir = -d
    if side:
        # Install-end hint available: do true "coaxial" fit—rotate install-end direction m fully to point toward host (-d),
        # not just around vertical axis (C1 reported: hand-type candidates in shoulder crease inserted backward every time, each needs extra yaw 180 turn).
        # Self-rotation around axis: keep piece's original "up" pointing up as much as possible; reviewer's yaw angle rotates around this install axis.
        m = _side_dir(side, q2, d)
        R = rot_between(m, -d)
        up1 = R @ UP
        pa = up1 - float(up1 @ d) * d
        pb = UP - float(UP @ d) * d
        if np.linalg.norm(pa) > 1e-6 and np.linalg.norm(pb) > 1e-6:
            pa, pb = pa / np.linalg.norm(pa), pb / np.linalg.norm(pb)
            R = rot_about(d, float(np.arctan2(float(np.cross(pa, pb) @ d), float(pa @ pb)))) @ R
        q2.apply_transform(_T(R))
        if yaw_extra:
            q2.apply_transform(_T(rot_about(d, np.deg2rad(float(yaw_extra)))))
    else:
        # No install-end hint (legacy path): rotate only around vertical axis, align new part's most-protruding horizontal direction to slot's outward direction
        dh = d - float(d @ UP) * UP
        if np.linalg.norm(dh) > 1e-6:
            dh = dh / np.linalg.norm(dh)
            p = samples(q2, N_PART)
            ph = p - np.outer(p @ UP, UP)
            far = ph[int(np.argmax(np.linalg.norm(ph, axis=1)))]
            if np.linalg.norm(far) > 1e-6:
                far = far / np.linalg.norm(far)
                ang = np.arctan2(float(np.cross(far, dh) @ UP), float(far @ dh))
                R = rot_about(UP, ang)
                q2.apply_transform(np.vstack([np.hstack([R, np.zeros((3, 1))]), [0, 0, 0, 1]]))
        if yaw_extra:
            R = rot_about(UP, np.deg2rad(float(yaw_extra)))
            q2.apply_transform(np.vstack([np.hstack([R, np.zeros((3, 1))]), [0, 0, 0, 1]]))
    # Scaling: align install cross-sections ("cap fits over head", rim width irrelevant)
    p = samples(q2, N_PART)
    m_c, m_r = mount_end(p - p.mean(0), -mdir)
    s = float(np.clip(r_slot / m_r, 0.25, 4.0))

    # But **aligning contact cross-section alone shrinks whole part**. Install-end-thick body-thin candidates (e.g. foot,
    # ankle cross-section wider than whole original leg) compress to 0.3x, robot leg becomes foot stuck to crotch.
    # Worse, compounds round-to-round: baseline is previous occupant; shrink each time, 20 rounds later
    # arm becomes small bump, head becomes white pimple—swap equals no swap.
    # So add overall size band, **anchored to original slot**, not floating with last round.
    # Upper bound loose (2.0): a baseball cap replacing a cowboy hat naturally exceeds the original box, which is fine.
    # Lower bound 0.75 is hard requirement—invisible arm worse than imperfect-seam arm.
    if slot_diag:
        pd = float(np.linalg.norm(q2.extents))
        if pd > 1e-9:
            s = float(np.clip(s, SIZE_LO * slot_diag / pd, SIZE_HI * slot_diag / pd))
    q2.apply_scale(s)
    p = samples(q2, N_PART)
    m_c2, _ = mount_end(p - p.mean(0), -mdir)
    q2.apply_translation(c - (q2.vertices.mean(0) + m_c2))
    return q2, dict(scale=s, contact_points=int(n_hit), slot_radius=float(r_slot),
                    mount_radius=float(m_r), contact_dir=[float(x) for x in d],
                    yaw_only=(not side), side=side)



MOUNT_TYPES_FILE = "mount_types.json"     # {slot: plane|axis|point}, written by agent after reading probe image
CAND_VERDICTS_FILE = "cand_verdicts.json"  # {"oid/pid": ["fit","bottom"] | ["skip","reason"]}


def _T(R, t=None):
    M = np.eye(4); M[:3, :3] = R
    if t is not None:
        M[:3, 3] = t
    return M


def _obb2(P):
    """2D point set principal-axis bounding box: (center, major axis a, minor axis b, length L1, width L2)."""
    m = P.mean(0)
    C = P - m
    _, v = np.linalg.eigh(C.T @ C)
    a, b = v[:, 1], v[:, 0]
    u, t = C @ a, C @ b
    ctr = m + a * (u.min() + u.max()) / 2 + b * (t.min() + t.max()) / 2
    return ctr, a, b, float(u.max() - u.min()), float(t.max() - t.min())


MOUNT_AXES = {
    "xpos": (1., 0., 0.), "xneg": (-1., 0., 0.),
    "ypos": (0., 1., 0.), "yneg": (0., -1., 0.),
    "zpos": (0., 0., 1.), "zneg": (0., 0., -1.),
}
MOUNT_SIDES = ("bottom", "top", "side", "end", "rim", *MOUNT_AXES)


def _side_dir(side, q, d):
    """Agent's "install end on which side of candidate" converts to unit vector pointing to install end in candidate's own coordinates."""
    if side in MOUNT_AXES:
        # Explicit canonical axes avoid ambiguous PCA on disks with thin stems.
        # q is centered here but has not been rotated out of canonical coordinates.
        return np.asarray(MOUNT_AXES[side], dtype=float)
    if side in ("bottom", "rim"):
        return -UP
    if side == "top":
        return UP
    pts = samples(q, N_PART)
    if side == "end":
        ax = principal_axis(pts)
        t = pts @ ax
        lo, hi = pts[t <= np.quantile(t, 0.2)], pts[t >= np.quantile(t, 0.8)]
        def rad(sel):
            rel = sel - sel.mean(0)
            perp = rel - np.outer(rel @ ax, ax)
            return float(np.percentile(np.linalg.norm(perp, axis=1), 90))
        return -ax if rad(lo) <= rad(hi) else ax       # install end is the thinner cross-section end
    if side == "side":
        ph = pts - np.outer(pts @ UP, UP)
        far = ph[int(np.argmax(np.linalg.norm(ph, axis=1)))]
        return -far / (np.linalg.norm(far) + 1e-12)      # reverse side of most protruding side adheres to host
    return -UP


def _support_normal(part, rest, diag, c, d):
    """plane slot install-surface normal = host's **surface normal** at contact region, not "contact points toward old part centroid".
    Old canopy's off-center centroid, that direction 22° off vertical, four wheels install cockeyed (C2 fish-cart). Take host surface normals
    of nearest triangles to old part's contact points, area-weighted average, flip to same side as d; degenerate case returns d."""
    d = d / (np.linalg.norm(d) + 1e-12)
    try:
        from scipy.spatial import cKDTree
        pp = samples(part, N_PART)
        rp, rf = trimesh.sample.sample_surface(rest, N_REST, seed=0)   # sample points + containing triangles
        dist, j = cKDTree(np.asarray(rp)).query(pp)
        sel = dist < CONTACT_TAU * diag * 1.5
        if sel.sum() < 10:
            return d
        fid = np.asarray(rf)[j[sel]]
        fn = rest.face_normals[fid]
        fn = fn * np.sign(fn @ d)[:, None]            # flip all to d side
        w = rest.area_faces[fid]
        m = (fn * w[:, None]).sum(0) / (w.sum() + 1e-12)
        if np.linalg.norm(m) < 0.5:                   # normals scattered heavily (contact surface is not a plane)
            return d
        return m / np.linalg.norm(m)
    except Exception:
        return d


def initial_pose_plane(q, part, rest, diag, frame=None, slot_diag=None, side="bottom",
                       yaw_extra=0.0):
    """Planar fit (plane-type slot, CAD mate idea):

      1. Host install surface = plane at contact centroid c with normal n = outward direction d; support footprint = 2D principal OBB
         of sampled points on rest near this plane and close to c (cart rim edge → rectangle).
      2. New part's install end (side agent points to on candidate image) rotate to face -n, take closest 8% sampled points
         as "landing footprint", also 2D principal OBB.
      3. Rotate around n to align landing principal axis with footprint principal axis (symmetry plane of symmetric part aligns with host).
      4. Uniform scale: landing footprint fits inside support footprint, use smaller of length/width ratio—tent legs fit cart corners;
         apply original slot size band (SIZE_LO..SIZE_HI) as fallback.
      5. Translate: landing footprint center to support center (center), lowest point touches install surface (slight embed, for snap_to).
    """
    c, d, r_slot, n_hit = frame if frame and frame[0] is not None else contact_frame(part, rest, diag)
    if c is None:
        return None, dict(note="Original part has no contact with host, cannot get contact surface")
    n = _support_normal(part, rest, diag, c, d)
    e1 = np.cross(n, UP)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(n, np.array([1.0, 0.0, 0.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    B = np.stack([e1, e2])                       # orthonormal basis in plane, n = e1 × e2
    # Support contours found only within **old part's own footprint** (15% margin): when cabin opening and fuselage top are at same height,
    # distance-based selection pulls entire fuselage top surface, all candidates pushed to upper limit (B4 reported)
    pp = samples(part, N_PART)
    hpp = (pp - c) @ n
    old_foot = pp[hpp <= np.quantile(hpp, 0.15)]
    oO, aO, bO, LO1, LO2 = _obb2((old_foot - c) @ B.T)
    rp = samples(rest, N_REST)
    h = (rp - c) @ n
    P2 = (rp - c) @ B.T
    u, t = (P2 - oO) @ aO, (P2 - oO) @ bO
    mrg = 0.1 * float(slot_diag or diag * 0.3)     # old part footprint margin: 15% + one-tenth of slot size
    inside = (np.abs(u) <= 0.575 * LO1 + mrg) & (np.abs(t) <= 0.575 * LO2 + mrg)
    band = rp[(np.abs(h) < 0.03 * diag) & inside]
    if len(band) < 20:
        reach = 1.5 * max(float(slot_diag or 0.0), 2.0 * float(r_slot))
        band = rp[(np.abs(h) < 0.03 * diag) & (np.linalg.norm(rp - c, axis=1) < reach)]
    if len(band) < 20:
        return None, dict(note="Cannot find support contour near install surface")
    oH, aH, bH, LH1, LH2 = _obb2((band - c) @ B.T)

    q2 = q.copy()
    q2.apply_translation(-q2.vertices.mean(0))
    md = _side_dir(side, q2, d)
    q2.apply_transform(_T(rot_between(md, -n)))          # install end faces host

    def footprint(m, q=0.25):
        """Landing contour takes lowest quarter nearest install surface (not lowest ring): round-bottom container's base rim only has small circle,
        using it as contour template would push entire part to upper limit (B2 reported); fitting height still uses lowest point."""
        p = samples(m, N_PART)
        hp = p @ n
        foot = p[hp <= np.quantile(hp, q)]
        return _obb2((foot - c) @ B.T), float(hp.min())
    (oF, aF, bF, LF1, LF2), _ = footprint(q2)
    ang = float(np.arctan2(aF[0] * aH[1] - aF[1] * aH[0], aF @ aH))
    if ang > np.pi / 2:
        ang -= np.pi
    elif ang < -np.pi / 2:
        ang += np.pi
    R2 = rot_about(n, ang + np.deg2rad(float(yaw_extra or 0.0)))
    q2.apply_transform(_T(R2))
    (oF, aF, bF, LF1, LF2), _ = footprint(q2)
    s = min(LH1 / max(LF1, 1e-6), LH2 / max(LF2, 1e-6)) * 0.97
    pd = float(np.linalg.norm(q2.extents))
    if slot_diag and pd > 1e-9:
        s = float(np.clip(s, SIZE_LO * slot_diag / pd, PLANE_SIZE_HI * slot_diag / pd))
    s = float(np.clip(s, 0.25, 4.0))
    q2.apply_scale(s)
    (oF, aF, bF, LF1, LF2), lowest = footprint(q2)
    shift = (oH - oF) @ B + n * (float(c @ n) - 0.002 * diag - lowest)
    q2.apply_translation(shift)
    return q2, dict(scale=s, contact_points=int(n_hit), slot_radius=float(r_slot),
                    mount_radius=float(max(LF1, LF2) / 2.0), contact_dir=[float(x) for x in d],
                    yaw_only=False, mount_type="plane", side=side,
                    outline=[LH1, LH2], footprint=[LF1, LF2],
                    plane_normal=[float(x) for x in n])


def _grid(pngs, out, title="", cols=4, size=420):
    """Compose render images into titled review grid (text English-only; machine lacks mixed CJK font)."""
    from PIL import Image, ImageDraw
    ims = []
    for p in pngs:
        im = Image.open(p).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        ims.append(Image.alpha_composite(bg, im).convert("RGB").resize((size, size)))
    rows = (len(ims) + cols - 1) // cols
    top = 22 * (title.count("\n") + 1) + 8
    sh = Image.new("RGB", (size * min(cols, len(ims)), top + size * rows), (255, 255, 255))
    d = ImageDraw.Draw(sh)
    for i, line in enumerate(title.split("\n")):
        d.text((8, 4 + 22 * i), line, fill=(0, 0, 0))
    for i, im in enumerate(ims):
        sh.paste(im, ((i % cols) * size, top + (i // cols) * size))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sh.save(out)
    return out



# ------------------------------------------------------------------ Add / Remove

FACE_MIN_AREA = 0.004      # minimum area of usable install face, fraction of diag²
FACE_FREE_MIN = 0.25       # if fraction of empty sampled points on face is below this, not usable


def _slug(text, n=18):
    import re
    t = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return t[:n] or "part"


def _rank_faces(faces, max_faces=8):
    """faces display ranking: pass vacancy threshold → sort → number. Decoupled from cache (cache stores pre-threshold raw,
    blocked faces can revive after blockers move)."""
    faces = [dict(f) for f in faces if f["free_frac"] >= FACE_FREE_MIN]
    # Sorting: total area × (0.5 + 0.5 × vacancy fraction). Sorting by free area alone puts box top (half occupied by hats/bottles)
    # behind small-but-empty shoulder-edge face (tested: rank 9 gets cut off)
    faces.sort(key=lambda f: (-(f["kind"] == "top"), -f["area"] * (0.5 + 0.5 * f["free_frac"])))
    for i, f in enumerate(faces[:max_faces], 1):
        f["id"] = i
    return faces[:max_faces]


def _faces_update(cache, pieces, diag):
    """Incremental update of install-face interference edges (Wilson–Latombe 1994 blocking-graph unidirectional case +
    broad-phase AABB collision filter; audited ASAP/Assemble-Them-All source: rebuilds SDF each check, no geometry cache,
    portable reusable pieces track "blocking relations per edge").
    cache=dict(fp, raw, slots); return (raw_faces, note). note: hit=geometry bitwise unchanged (retexture/re-ask same turn),
    local:N/M=re-probe N parts, full=all. Invalidation (any one triggers re-probe): ①part changed ②its face's current blocker changed ③edit AABB hits face's probe volume."""
    fp = {k: (len(m.vertices), tuple(np.round(np.asarray(m.bounds, float).ravel(), 5).tolist()))
          for k, m in pieces.items()}
    if cache.get("fp") == fp and cache.get("raw") is not None:
        return cache["raw"], "hit"
    if cache.get("fp") and cache.get("raw") is not None:
        old_fp = cache["fp"]
        changed = {k for k in set(fp) | set(old_fp) if fp.get(k) != old_fp.get(k)}
        boxes = [np.asarray(pieces[k].bounds, float) for k in changed if k in pieces]
        boxes += [np.asarray(old_fp[k][1], float).reshape(2, 3) for k in changed if k in old_fp]
        if boxes and changed:
            rlo = np.min([b[0] for b in boxes], 0)
            rhi = np.max([b[1] for b in boxes], 0)

            def _hit(f):
                if f["piece"] in changed or (changed & set(f.get("blockers", ()))):
                    return True
                a = f.get("probe_aabb")
                if a is None:
                    return True                     # old cache missing probe geometry, conservatively recompute
                return bool(np.all(np.asarray(a[0]) <= rhi) and np.all(rlo <= np.asarray(a[1])))
            redo = ({f["piece"] for f in cache["raw"] if _hit(f)} | changed) & set(pieces)
            kept = [f for f in cache["raw"] if f["piece"] not in redo and f["piece"] in pieces]
            fresh = free_faces(pieces, diag, only=redo, raw=True) if redo else []
            raw = kept + fresh
            note = f"local:{len(redo)}/{len(pieces)}"
        else:
            raw = free_faces(pieces, diag, raw=True); note = "full"
    else:
        raw = free_faces(pieces, diag, raw=True); note = "full"
    cache["fp"] = fp
    cache["raw"] = raw
    return raw, note


GROUND_MARGIN = 0.45        # ground expansion ratio toward footprint (× diag)


def ground_face(pieces, diag):
    """Virtual ground as install surface (grounded parts are part of assembly, but ground has zero materialization—
    not in GLB/render/bounding sphere, only exists in placement calculation; literature-wise corresponds to ASAP's <ground/> and scene-composed
    support plane). Returns face dict with fixed id 0; spots around host footprint perimeter."""
    if not pieces:
        return None
    bs = np.array([[m.bounds[0], m.bounds[1]] for m in pieces.values()], float)
    lo = bs[:, 0].min(0); hi = bs[:, 1].max(0)
    y0 = float(lo[1])
    m = GROUND_MARGIN * diag
    cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
    hx, hz = (hi[0] - lo[0]) / 2 + m, (hi[2] - lo[2]) / 2 + m
    c = np.array([cx, y0, cz])
    e1 = np.array([1.0, 0.0, 0.0]); n = UP.copy(); e2 = np.cross(n, e1)
    fx, fz = (hi[0] - lo[0]) / 2, (hi[2] - lo[2]) / 2      # footprint half-width
    spots = []
    ring = [(fx + m / 2, 0), (-(fx + m / 2), 0), (0, fz + m / 2), (0, -(fz + m / 2)),
            (fx + m / 2, fz + m / 2), (-(fx + m / 2), -(fz + m / 2))]
    for du, dv in ring[:5]:
        p3 = c + du * e1 + dv * e2
        spots.append(dict(uv=[round(du / hx, 3), round(dv / hz, 3)], p=p3, clear=float(m / 2)))
    return dict(id=0, piece="__ground__", kind="ground", n=n, c=c, e1=e1, e2=e2,
                L1=2 * hx, L2=2 * hz, area=4 * hx * hz, free_frac=1.0,
                free_c=c, free_L1=2 * hx, free_L2=2 * hz, free_area=4 * hx * hz, spots=spots)


def free_faces(pieces, diag, max_faces=8, n_probe=1500, only=None, raw=False):
    """Enumerate mountable faces on current assembly: adjacent triangles with similar normals cluster into patches, top-facing (top) and
    side-facing (side) types, subtract parts already occupying space above. Return sorted by free area, each entry:
    id, piece, kind, n(normal), c(face centroid), e1/e2(in-face basis, e1 along long edge), L1/L2(whole outline),
    free_c / free_L1 / free_L2(free space outline)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    all_pts = {k: samples(m, N_REST // 4) for k, m in pieces.items()}
    faces = []
    for k, m0 in pieces.items():
        if only is not None and k not in only:
            continue                    # local update: only re-probe failed parts, occlusion check still views all
        if len(m0.faces) == 0:
            continue
        if len(m0.faces) > 150000:
            # High face-count parts skip mount-face enumeration: query_pairs explodes on ~million vertices
            # (329k faces gold bell crashed free_faces to 141G anon-rss OOM kill, W3-122 tested;
            # 400k ingest gate doesn't stop this). Serves as blocker for other face probes (all_pts unaffected).
            continue
        m = m0
        fn, fa = m.face_normals, m.area_faces
        fc = m.triangles.mean(1)
        up = fn @ UP
        # Grouping ignores normal difference between adjacent triangles (canopy folds, box dents split one face to strips),
        # only orientation class + spatial proximity: upward triangles spatially connected form one "top face"
        lab = np.full(len(m.faces), -1)
        nxt = 0
        for kind_sel, need_normal in ((up > 0.7, False), (np.abs(up) < 0.3, True)):
            idx = np.where(kind_sel)[0]
            if len(idx) == 0:
                continue
            # Connectivity: two triangles with vertices close enough (1.5% diag) count as connected—fine gaps between boards,
            # large non-vertex-sharing triangles connect; sides also require similar normals (front/back don't merge)
            V = m.vertices[m.faces[idx]].reshape(-1, 3)          # 3 vertices per face
            owner = np.repeat(np.arange(len(idx)), 3)
            vt = cKDTree(V)
            vp = np.array(list(vt.query_pairs(0.015 * diag)), dtype=int).reshape(-1, 2)
            pairs = np.stack([owner[vp[:, 0]], owner[vp[:, 1]]], 1) if len(vp) else np.zeros((0, 2), int)
            pairs = pairs[pairs[:, 0] != pairs[:, 1]] if len(pairs) else pairs
            if need_normal and len(pairs):
                cos = np.einsum("ij,ij->i", fn[idx[pairs[:, 0]]], fn[idx[pairs[:, 1]]])
                pairs = pairs[cos > np.cos(np.deg2rad(30.0))]
            nn = len(idx)
            if len(pairs):
                comp = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                                                       shape=(nn, nn)), directed=False)[1]
            else:
                comp = np.arange(nn)
            lab[idx] = comp + nxt
            nxt += comp.max() + 1
        # Coplanar merge: in same part, patches with similar normals (10°), consistent height (1.5% diag), gap <5% diag
        # merge—box top is three planks, plank gap wider than vertex neighbor radius
        labs = [int(x) for x in np.unique(lab) if x >= 0]
        info = {}
        for pid in labs:
            sel = lab == pid
            nrm = (fn[sel] * fa[sel, None]).sum(0)
            if np.linalg.norm(nrm) < 1e-9 or fa[sel].sum() < 0.05 * FACE_MIN_AREA * diag * diag:
                continue
            nrm /= np.linalg.norm(nrm)
            info[pid] = dict(n=nrm, c=(fc[sel] * fa[sel, None]).sum(0) / fa[sel].sum(),
                             tree=cKDTree(fc[sel]), pts=fc[sel])
        parent = {pid: pid for pid in info}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        keys = list(info)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a_, b_ = info[keys[i]], info[keys[j]]
                if float(a_["n"] @ b_["n"]) < np.cos(np.deg2rad(10.0)):
                    continue
                if abs(float((a_["c"] - b_["c"]) @ a_["n"])) > 0.015 * diag:
                    continue
                if a_["tree"].query(b_["pts"])[0].min() > 0.05 * diag:
                    continue
                parent[find(keys[i])] = find(keys[j])
        for pid in keys:
            lab[lab == pid] = find(pid)
        for pid in np.unique(lab):
            if pid < 0:
                continue
            sel = lab == pid
            area = float(fa[sel].sum())
            if area < FACE_MIN_AREA * diag * diag:
                continue
            nrm = (fn[sel] * fa[sel, None]).sum(0)
            if np.linalg.norm(nrm) < 1e-9:
                continue
            nrm /= np.linalg.norm(nrm)
            up = float(nrm @ UP)
            kind = "top" if up > 0.7 else ("side" if abs(up) < 0.3 else None)
            if kind is None:
                continue
            c = (fc[sel] * fa[sel, None]).sum(0) / area
            e1 = np.cross(nrm, UP) if kind == "side" else np.array([1.0, 0.0, 0.0])
            e1 = e1 - float(e1 @ nrm) * nrm
            if np.linalg.norm(e1) < 1e-6:
                e1 = np.cross(nrm, np.array([0.0, 0.0, 1.0]))
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(nrm, e1)
            B = np.stack([e1, e2])
            # Sample on face (area-weighted), check if anything above in height band
            w = fa[sel] / area
            fi = np.where(sel)[0][np.random.default_rng(0).choice(sel.sum(), size=n_probe, p=w)]
            tri = m.triangles[fi]
            r1, r2 = np.random.default_rng(1).random((2, n_probe))
            sq = np.sqrt(r1)
            pts = (1 - sq)[:, None] * tri[:, 0] + (sq * (1 - r2))[:, None] * tri[:, 1] + (sq * r2)[:, None] * tri[:, 2]
            others_names = [kk for kk in all_pts if kk != k and len(all_pts[kk])]
            others = [all_pts[kk] for kk in others_names]
            occupied = np.zeros(n_probe, bool)
            blk = set()                 # This face's blocking edges: parts currently blocking it (for incremental invalidation)
            if others:
                op = np.concatenate(others)
                owner = np.concatenate([np.full(len(all_pts[kk]), j) for j, kk in enumerate(others_names)])
                tree = cKDTree(op)
                for i, q in enumerate(pts):
                    idx = tree.query_ball_point(q, 0.06 * diag)
                    if idx:
                        h = (op[idx] - q) @ nrm
                        mask = (h > 0.002 * diag) & (h < 0.5 * diag)
                        if np.any(mask):
                            occupied[i] = True
                            blk.update(owner[np.asarray(idx)][mask].tolist())
            free = pts[~occupied]
            frac = float(len(free)) / n_probe
            P2 = (pts - c) @ B.T
            oc, a1, a2, L1, L2 = _obb2(P2)
            # change in-face basis to outline principal axis
            e1n = a1[0] * e1 + a1[1] * e2
            e2n = np.cross(nrm, e1n)
            Bn = np.stack([e1n, e2n])
            oc3 = c + oc @ B
            spots = []
            if frac >= FACE_FREE_MIN:
                # Free space outline measured along **e1/e2** (before, used free points' own principal axis, different from placement basis,
                # canopy reported 1.99×2.08 but actually 1.09×1.47, at=±0.75 dropped outside—D2 reported)
                F2 = (free - oc3) @ Bn.T
                fL1 = float(F2[:, 0].max() - F2[:, 0].min())
                fL2 = float(F2[:, 1].max() - F2[:, 1].min())
                fc2 = np.array([(F2[:, 0].max() + F2[:, 0].min()) / 2,
                                (F2[:, 1].max() + F2[:, 1].min()) / 2])
                free_c = oc3 + fc2 @ Bn
                # Landing spots: grid on face, free cells by "distance to nearest occupied cell / boundary" local max,
                # mutually separated for agent to select by number. More reliable than guessing (u,v) direction (D2 reported)
                spots = _free_spots(P2, ~occupied, oc3, Bn, L1, L2)
            else:
                fL1 = fL2 = 0.0
                free_c = oc3
            _corn = [oc3 + sx * L1 / 2 * e1n + sy * L2 / 2 * e2n
                     for sx in (-1, 1) for sy in (-1, 1)]
            _vol = np.array(_corn + [cc + 0.5 * diag * nrm for cc in _corn])
            faces.append(dict(piece=k, kind=kind, n=nrm, c=oc3, e1=e1n, e2=e2n, L1=L1, L2=L2,
                              area=area, free_frac=frac, free_c=free_c, free_L1=fL1, free_L2=fL2,
                              free_area=area * frac, spots=spots,
                              blockers=sorted({others_names[j] for j in blk}),
                              probe_aabb=[(_vol.min(0) - 0.06 * diag).tolist(),
                                          (_vol.max(0) + 0.06 * diag).tolist()]))
    if raw:
        return faces                    # cache stores pre-threshold raw: occluded faces can revive after blockers move
    return _rank_faces(faces, max_faces)


def _free_spots(P2, free_mask, oc3, B, L1, L2, n_grid=24, max_spots=5):
    """Candidate landing spots in face's free space. Grid face in (u, v), distance-transform free cells (distance to nearest occupied/boundary),
    take mutually-separated first few local maxima. Each spot has uv (in whole-outline -1..1 coords),
    3D position, clearance (usable radius)."""
    if free_mask.sum() < 20:
        return []
    u = P2[:, 0]; v = P2[:, 1]
    umin, umax = float(u.min()), float(u.max())
    vmin, vmax = float(v.min()), float(v.max())
    du = max((umax - umin) / n_grid, 1e-9); dv = max((vmax - vmin) / n_grid, 1e-9)
    gi = np.clip(((u - umin) / du).astype(int), 0, n_grid - 1)
    gj = np.clip(((v - vmin) / dv).astype(int), 0, n_grid - 1)
    tot = np.zeros((n_grid, n_grid), int); fre = np.zeros((n_grid, n_grid), int)
    np.add.at(tot, (gi, gj), 1)
    np.add.at(fre, (gi, gj), free_mask.astype(int))
    ok = (tot > 0) & (fre >= 0.6 * tot)          # cell is free only if 60% or more sample points are empty
    try:
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(np.pad(ok, 1))[1:-1, 1:-1]   # add boundary of occupied cells = boundary
    except Exception:
        return []
    spots = []
    d = dist.copy()
    while len(spots) < max_spots:
        idx = int(np.argmax(d))
        i, j = idx // n_grid, idx % n_grid
        if d[i, j] <= 1.0:
            break
        cu = umin + (i + 0.5) * du; cv = vmin + (j + 0.5) * dv
        clear = float(d[i, j]) * float(min(du, dv))
        spots.append(dict(uv=[round(2 * (cu - (umin + umax) / 2) / max(umax - umin, 1e-9), 3),
                              round(2 * (cv - (vmin + vmax) / 2) / max(vmax - vmin, 1e-9), 3)],
                          p=(oc3 + np.array([cu, cv]) @ B), clear=clear))
        # Erase around this spot so next one doesn't stick to it
        ii, jj = np.ogrid[:n_grid, :n_grid]
        d[((ii - i) ** 2 + (jj - j) ** 2) <= (d[i, j] * 1.2) ** 2] = 0
    return spots


def face_marker(face, diag):
    """Draw free-space outline as magenta thin plate (slight float above face), spheres at each end:
    green sphere at +u (long-edge direction), blue sphere at +v (short-edge direction), for agent to set landing spots."""
    L1, L2 = max(face["free_L1"], 0.02 * diag), max(face["free_L2"], 0.02 * diag)
    R = np.stack([face["e1"], face["e2"], face["n"]], axis=1)
    quad = trimesh.creation.box(extents=[L1, L2, 0.004 * diag])
    quad.apply_transform(_T(R, face["free_c"] + face["n"] * 0.004 * diag))
    quad.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=[255, 0, 255, 255], emissiveFactor=[0.5, 0.0, 0.5]))
    g = trimesh.creation.icosphere(subdivisions=1, radius=0.015 * diag)
    g.apply_translation(face["free_c"] + face["e1"] * L1 / 2 + face["n"] * 0.01 * diag)
    g.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=[0, 220, 0, 255], emissiveFactor=[0.0, 0.5, 0.0]))
    b = trimesh.creation.icosphere(subdivisions=1, radius=0.015 * diag)
    b.apply_translation(face["free_c"] + face["e2"] * L2 / 2 + face["n"] * 0.01 * diag)
    b.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=[0, 80, 255, 255], emissiveFactor=[0.0, 0.1, 0.5]))
    # Spots: #1 brightest, dimmer in sequence; radius by clearance, agent instantly sees where sizes fit
    dots = []
    for i, sp in enumerate(face.get("spots", [])[:5]):
        r = float(np.clip(sp["clear"] * 0.5, 0.012 * diag, 0.06 * diag))
        m = trimesh.creation.icosphere(subdivisions=1, radius=r)
        m.apply_translation(np.asarray(sp["p"]) + face["n"] * (r + 0.004 * diag))
        shade = 1.0 - 0.16 * i
        m.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[int(255 * shade), int(160 * shade), 0, 255],
            emissiveFactor=[0.5 * shade, 0.3 * shade, 0.0]))
        dots.append(m)
    return quad, g, b, dots


def initial_pose_add(q, face, diag, size=0.5, at=(0.0, 0.0), side="bottom", clear=None):
    """Place new part on free face: install-end toward face (-n), landing principal axis aligns face long edge, scale so landing
    long edge = size × face's free-space short edge, landing center to at=(u, v) point (relative free outline,
    u along long edge, v along short edge, -1..1, (0,0) center), lowest point touches face."""
    n = face["n"]
    B = np.stack([face["e1"], face["e2"]])
    q2 = q.copy()
    q2.apply_translation(-q2.vertices.mean(0))
    md = _side_dir(side, q2, -n)
    q2.apply_transform(_T(rot_between(md, -n)))

    def footprint(m, qq=0.25):
        p = samples(m, N_PART)
        hp = p @ n
        foot = p[hp <= np.quantile(hp, qq)]
        return _obb2((foot - face["free_c"]) @ B.T), float(hp.min())
    (oF, aF, bF, LF1, LF2), _ = footprint(q2)
    ang = float(np.arctan2(aF[0] * 0.0 - aF[1] * 1.0, aF @ np.array([1.0, 0.0])))
    ang = float(np.arctan2(-aF[1], aF[0]))
    if ang > np.pi / 2:
        ang -= np.pi
    elif ang < -np.pi / 2:
        ang += np.pi
    q2.apply_transform(_T(rot_about(n, ang)))
    (oF, aF, bF, LF1, LF2), _ = footprint(q2)
    # Spot with available radius sets size (size is "what fraction of spot clearance"), else use free space short edge
    ref = 2.0 * float(clear) if clear else max(min(face["free_L1"], face["free_L2"]), 0.05 * diag)
    s = float(np.clip(size * ref / max(LF1, LF2, 1e-6), 0.02, 50.0))
    q2.apply_scale(s)
    (oF, aF, bF, LF1, LF2), lowest = footprint(q2)
    u, v = at
    # at is relative coords in "whole face outline" (spot uv also measured whole-face), not free outline
    target = np.array([u * face["L1"] / 2.0, v * face["L2"] / 2.0]) + (face["c"] - face["free_c"]) @ B.T
    shift = (target - oF) @ B + n * (float(face["free_c"] @ n) - 0.002 * diag - lowest)
    q2.apply_translation(shift)
    return q2, dict(scale=s, mount_type="plane", side=side, contact_points=0,
                    slot_radius=float(ref / 2), mount_radius=float(max(LF1, LF2) / 2),
                    contact_dir=[float(x) for x in n], yaw_only=False,
                    outline=[face["free_L1"], face["free_L2"]], footprint=[LF1, LF2],
                    plane_normal=[float(x) for x in n], op="add",
                    face=dict(id=face["id"], piece=face["piece"], kind=face["kind"]),
                    size=size, at=[u, v], clear=clear)


def penetration_frac(q, others_tree, diag, depth=0.02):
    """Approx how much of new part's samples lie inside other parts: fraction of points very close to surface.
    Flush placement has small contact band, deep insertion makes this large."""
    d, _ = others_tree.query(samples(q, N_PART))
    return float((d < depth * diag).mean())



def _grid_labeled(pngs, labels, out, title="", cols=4, size=420):
    """Like _grid, but stamps label (candidate number) in upper left of each small image."""
    from PIL import Image, ImageDraw
    ims = []
    for p in pngs:
        im = Image.open(p).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        ims.append(Image.alpha_composite(bg, im).convert("RGB").resize((size, size)))
    rows = (len(ims) + cols - 1) // cols
    top = 22 * (title.count("\n") + 1) + 8
    sh = Image.new("RGB", (size * min(cols, len(ims)), top + size * rows), (255, 255, 255))
    d = ImageDraw.Draw(sh)
    for i, line in enumerate(title.split("\n")):
        d.text((8, 4 + 22 * i), line, fill=(0, 0, 0))
    for i, im in enumerate(ims):
        x, y = (i % cols) * size, top + (i // cols) * size
        sh.paste(im, (x, y))
        if i < len(labels) and labels[i]:
            d.rectangle([x, y, x + 8 + 7 * len(labels[i]), y + 16], fill=(255, 255, 255))
            d.text((x + 4, y + 2), labels[i], fill=(200, 0, 0))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sh.save(out)
    return out


def virtual_frame(g, pieces, diag, frac=0.05):
    """Create contact surface for slot without host contact: find nearest part as neighbor, take closest-to-neighbor
    frac of samples on this slot's surface as contact cluster. Return ((c, d, r, n), info) or (None, None)."""
    from scipy.spatial import cKDTree
    pp = samples(pieces[g], N_PART)
    if len(pp) == 0:
        return None, None
    best = None
    for k, m in pieces.items():
        if k == g:
            continue
        rp = samples(m, N_REST // 4)
        if len(rp) == 0:
            continue
        dist, _ = cKDTree(rp).query(pp)
        if best is None or dist.min() < best[0]:
            best = (float(dist.min()), k, dist)
    if best is None:
        return None, None
    gap, nb, dist = best
    hit = pp[dist <= np.quantile(dist, frac)]
    c = hit.mean(0)
    d = pp.mean(0) - c
    n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else UP.copy()
    rel = hit - c
    perp = rel - np.outer(rel @ d, d)
    r = max(float(np.percentile(np.linalg.norm(perp, axis=1), 90)), 1e-4)
    return (c, d, r, len(hit)), dict(neighbour=nb, gap_rel=gap / diag)


def is_connected_virtual(pieces, diag, virtual, tau=CONTACT_TAU):
    """After one round, is assembly still one whole? Virtual-edge slot's edge to neighbor counts directly (gap checked separately by gate)."""
    keys, adj, _ = CG.build(pieces, diag, tau)
    for g, info in (virtual or {}).items():
        nb = info.get("neighbour")
        if g in keys and nb in keys:
            i, j = keys.index(g), keys.index(nb)
            adj[i, j] = adj[j, i] = True
    return CG.n_components(adj) == 1


def _init_host(oid, out_dir):
    """Host one-time derivatives: grouped parts, contact surface per slot, slot sizes, initial state.
    Write to <out_dir>/ckpt/init.pkl; read if exists (probe stage and round 1 share same)."""
    import slot_registry as SR
    import pickle
    initp = os.path.join(out_dir, "ckpt", "init.pkl")
    reg = SR.get_one(oid)
    if not reg:
        print("  ! No slot registry entry for this object", flush=True); return None
    pieces, diag, anchors = grouped_pieces(oid, reg["groups"])
    if os.path.isfile(initp):
        try:
            ini = _ckpt_read(initp)
            ini["pieces"] = pieces
            return ini
        except UntrustedCheckpoint as e:
            print(f"  ! {e}; recomputing the host state", flush=True)
    ok = [k for k in pieces if k not in anchors]
    roles = reg.get("roles", {})
    print(f"  Swappable slots {ok}  anchors {sorted(anchors)}", flush=True)
    frames = {}
    for pid in pieces:
        rest0 = trimesh.util.concatenate([m for s2, m in pieces.items() if s2 != pid])
        frames[pid] = contact_frame(pieces[pid], rest0, diag)
    # Slots with no host contact (bucket beside cart, floating shield): anchor to **nearest node**, add virtual edge
    # (no ground; floating things have no ground). Virtual contact surface = closest-to-neighbor cluster
    # on this slot's surface; record original gap, preserve during placement: no snap, relax gap gate to original, connectivity includes virtual edge.
    virtual = {}
    floating = [g for g in ok if frames.get(g) is None or frames[g][0] is None]
    if floating:
        # Parts already connected to anchors form "tree"; each iteration anchor closest floating slot to tree, merge it,
        # avoid two buckets anchoring to each other, neither connecting cart (first version tested)
        # Tree starts with anchors and "slots with real contact" (don't use contact graph components: bucket/wheel stray samples
        # just above threshold merge into cart component, two buckets nearest each other, result is mutual anchoring)
        tree = set(anchors) | {g for g in pieces if g not in floating and g not in anchors}
        while floating:
            best = None
            for g in floating:
                fr, info = virtual_frame(g, {k: pieces[k] for k in tree | {g}}, diag)
                if fr is not None and (best is None or info["gap_rel"] < best[2]["gap_rel"]):
                    best = (g, fr, info)
            if best is None:
                break
            g, fr, info = best
            frames[g] = fr
            virtual[g] = info
            tree.add(g)
            floating.remove(g)
            print(f"  Slot {g} no contact, virtual edge to {info['neighbour']} (gap {info['gap_rel']:.3f})", flush=True)
    nc = [g for g in ok if frames.get(g) is None or frames[g][0] is None]
    if nc:
        print(f"  ! Slots {nc} cannot find neighbor to anchor to, remove; remain {[g for g in ok if g not in nc]}", flush=True)
        ok = [g for g in ok if g not in nc]
    # Cleanup: mutually-touching but overall-disconnected "islands" (helmet + plume pair, W3-56 reported) not caught above—
    # each has real contact frame (with each other), but whole component unconnected to anchors, assembly starts "broken". Re-check global connectivity,
    # for each unanchored component add one piece with virtual edge (keep its real contact frame, virtual edge only manages connectivity/gap).
    for _round in range(len(pieces)):
        keys, adj, _ = CG.build(pieces, diag)
        for g_, info_ in virtual.items():
            nb_ = info_.get("neighbour")
            if g_ in keys and nb_ in keys:
                i_, j_ = keys.index(g_), keys.index(nb_)
                adj[i_, j_] = adj[j_, i_] = True
        from scipy.sparse import coo_matrix as _coo
        from scipy.sparse.csgraph import connected_components as _cc
        _, lab = _cc(_coo(adj), directed=False)
        root = {lab[keys.index(a)] for a in anchors if a in keys}
        island = [k for k, l in zip(keys, lab) if l not in root]
        if not island:
            break
        tree = set(k for k in keys if k not in island)
        best = None
        for g in island:
            fr, info = virtual_frame(g, {k: pieces[k] for k in tree | {g}}, diag)
            if fr is not None and (best is None or info["gap_rel"] < best[2]["gap_rel"]):
                best = (g, fr, info)
        if best is None:
            print(f"  ! Island {island} cannot anchor, keep as is", flush=True)
            break
        g, fr, info = best
        if frames.get(g) is None or frames[g][0] is None:
            frames[g] = fr
        virtual[g] = info
        print(f"  Island member {g} add virtual edge to {info['neighbour']} (gap {info['gap_rel']:.3f})", flush=True)
    slot_diag = {k: float(np.linalg.norm(pieces[k].extents)) for k in ok}
    state0 = {g: dict(caption=f"the {g.replace('_',' ')} of the object", slot=g,
                      bbox=_bbox(pieces[g]),
                      queries=(roles.get(g) or {}).get("queries", []),
                      accepts=(roles.get(g) or {}).get("accepts", []),
                      rejects=(roles.get(g) or {}).get("rejects", [])) for g in ok}
    _ckpt_dir(out_dir)   # init.pkl, not turn-1.pkl: that would collide with the turn??.pkl wildcard (review #15)
    ini = dict(frames=frames, slot_diag=slot_diag, diag=diag, ok=ok,
               anchors=anchors, state0=state0, virtual=virtual)
    _ckpt_dump(ini, initp)
    ini["pieces"] = pieces
    return ini


# ------------------------------------------------------------------ chain

def _captions_path():
    # Only one place decides which captions to read, no duplicates across files. See build_replace_chains.
    from build_replace_chains import captions_path
    return captions_path()


def _bbox(m):
    return [m.bounds[0].tolist(), m.bounds[1].tolist()]


def _focus(pieces, g):
    """Close-up camera target: this round's new part's bbox converted to blender_kit normalize='whole'
    coordinates after normalization. kit reads GLB converts y-up to z-up ((x,y,z)->(x,-z,y)),
    normalization center/scale taken from whole-scene bbox—replicate same conversion here,
    so blender_serve just swaps scene.center/diag with these numbers to aim camera at new part."""
    mn = np.min([m.bounds[0] for m in pieces.values()], axis=0)
    mx = np.max([m.bounds[1] for m in pieces.values()], axis=0)
    scale = float(max((mx - mn).max(), 1e-9))
    d = ((pieces[g].bounds[0] + pieces[g].bounds[1]) / 2.0 - (mn + mx) / 2.0) / scale
    return dict(center=[float(d[0]), float(-d[2]), float(d[1])],
                diag=float(np.linalg.norm(pieces[g].extents) / scale))


# ------------------------------------------------------------------ place

_LIB_MEMO = {}


def _lib_memo(pool):
    """In-process cache for retrieval library. serve mode re-enters stage_run each round; without cache
    PoolLibrary re-reads 111MB meta each round, embedding library re-reads 0.5GB each round.
    In nominee-pool mode, don't even touch caps (95MB json)."""
    key = pool or "__embedding__"
    if key not in _LIB_MEMO:
        if pool:
            from build_replace_chains import PoolLibrary
            _LIB_MEMO[key] = PoolLibrary(pool)
        else:
            from build_replace_chains import make_library
            _LIB_MEMO[key] = make_library(json.load(open(_captions_path())))
    return _LIB_MEMO[key]


class UntrustedCheckpoint(Exception):
    """A checkpoint that is not provably written by this user's own runs; it is never unpickled."""


def _ckpt_key():
    """Per-user secret that signs checkpoints. Checkpoints are pickles and unpickling can run code, so a checkpoint is only loaded
    when it carries a valid HMAC-SHA256 signature made with this key, i.e. when one of this user's own runs wrote it. The key lives
    outside every output directory: ~/.config/edithero/checkpoint.key (or $XDG_CONFIG_HOME/edithero), mode 0600."""
    import secrets
    d = os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "edithero")
    p = os.path.join(d, "checkpoint.key")
    if not os.path.isfile(p):
        os.makedirs(d, mode=0o700, exist_ok=True)
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(secrets.token_bytes(32))
        except FileExistsError:
            pass                                   # another process created it first
    st = os.stat(p)
    if st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise UntrustedCheckpoint(f"{p} must belong to the current user with mode 0600")
    return open(p, "rb").read()


def _ckpt_dump(obj, path):
    """Atomic, signed checkpoint write (signature first, then the pickle); the file is created 0600."""
    import pickle, hmac, hashlib
    data = pickle.dumps(obj)
    sig = hmac.new(_ckpt_key(), data, hashlib.sha256).digest()
    tmp = f"{path}.{os.getpid()}.tmp"             # half-written checkpoint on kill cannot survive (review #4)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(sig + data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _ckpt_read(path):
    """Load a checkpoint only if it and its directory belong to the current user, are not writable by group/others, and its
    signature verifies with this user's key; otherwise raise UntrustedCheckpoint without unpickling anything."""
    import pickle, hmac, hashlib
    for p in (path, os.path.dirname(os.path.abspath(path))):
        st = os.stat(p)
        if st.st_uid != os.getuid() or st.st_mode & 0o022:
            raise UntrustedCheckpoint(f"{p}: owned by another user or writable by group/others; checkpoint not loaded")
    raw = open(path, "rb").read()
    sig, data = raw[:32], raw[32:]
    if len(raw) < 33 or not hmac.compare_digest(sig, hmac.new(_ckpt_key(), data, hashlib.sha256).digest()):
        raise UntrustedCheckpoint(f"{path}: no valid signature of this user's key (foreign, modified, truncated or older format); "
                                  "checkpoint not loaded")
    return pickle.loads(data)


def _ckpt_dir(out_dir):
    d = os.path.join(out_dir, "ckpt")
    os.makedirs(d, mode=0o700, exist_ok=True)
    os.chmod(d, 0o700)
    return d


def _ckpt_save(out_dir, turn, obj):
    _ckpt_dump(obj, os.path.join(_ckpt_dir(out_dir), f"turn{turn:02d}.pkl"))


def _ckpt_load(out_dir, turn):
    return _ckpt_read(os.path.join(out_dir, "ckpt", f"turn{turn:02d}.pkl"))


def stage_run(oid, out_dir, n_turns, seed=0, max_tries=8, size_ratio=2.5, veto=None,
              pool=None, yaw_fix=None, resume_from=None, stop_after=None, pin=None,
              cand_hook=None, op=None):
    """Pick-while-placing: if placement fails, backtrack and try next candidate.

    Before, strategy was "decide all 20 rounds' candidates, then place each", with no recovery—
    round 10 snap_to refuses (gap 0.065), part hangs, next rounds all False connectivity.
    Candidate quality only known **after placement**, so these two steps must interleave.

    Each candidate passes five gates; any failure moves to next:

      1. Can acquire contact surface
      2. `snap_to` doesn't refuse—refusal means initial value wrong, don't enlarge max_move
      3. Contact gap ≤ 0.01 (DGL's τ_c)
      4. **Scale doesn't hit boundary**, placed size ≤ original_size × size_ratio—
         block "sword longer than robot" type, kitbash guide puts proportions first
      5. After placement, whole object still one piece
    """
    from mini_articraft.sdk._mesh.core import MeshGeometry
    from mini_articraft.sdk.mesh import snap_to, SnapRefused
    from scipy.spatial import cKDTree
    from build_replace_chains import head_clause
    import slot_registry as SR

    import pickle
    initp = os.path.join(out_dir, "ckpt", "init.pkl")
    resuming = bool(resume_from and resume_from > 1 and os.path.isfile(initp))
    if resuming:
        ini = _ckpt_read(initp)              # raises UntrustedCheckpoint for a checkpoint this user's runs did not write
        frames, slot_diag = ini["frames"], ini["slot_diag"]
        diag, ok, anchors, state0 = ini["diag"], list(ini["ok"]), ini["anchors"], ini["state0"]
        pieces = None                      # Will be overwritten by checkpoint immediately
        virtual = ini.get("virtual", {})
    else:
        ini = _init_host(oid, out_dir)
        if ini is None:
            return
        pieces, frames, slot_diag = ini["pieces"], ini["frames"], ini["slot_diag"]
        diag, ok, anchors, state0 = ini["diag"], list(ini["ok"]), ini["anchors"], ini["state0"]
        virtual = ini.get("virtual", {})
    # Slot install surface type (agent writes after reading probe images); no file = all use old disc model
    mt_p = os.path.join(out_dir, MOUNT_TYPES_FILE)
    mount_types = json.load(open(mt_p)) if os.path.isfile(mt_p) else {}
    if not resuming:
        # Agent can mark too-small / blocked / unplaceable slots as "skip" after probe images, no need to waste round (B1 reported)
        sk = [g for g in ok if str(mount_types.get(g, "")).lower() == "skip"]
        if sk:
            ok = [g for g in ok if g not in sk]
            print(f"  Slots {sk} marked skip by agent, no swap; remain {ok}", flush=True)
    lib = _lib_memo(pool)
    rng = random.Random(seed)
    # Surface sampling (trimesh.sample_surface) uses global numpy random; without seed contact surface/install radius
    # jitter mm-scale each run (tested same seed twice: scale 1.69 vs 1.67).
    # veto / yaw_fix's "deterministic replay of rest of chain" must rest on this line.
    np.random.seed(seed)
    state = {g: dict(v) for g, v in state0.items()}

    os.makedirs(out_dir, exist_ok=True)

    def export(turn):
        sc = trimesh.Scene()
        for pid, m in pieces.items():
            sc.add_geometry(m, geom_name=f"slot_{pid}")
        sc.export(os.path.join(out_dir, f"turn{turn:02d}.glb"))

    if not resuming:
        export(0)          # Resume: pieces is still None (about to be overwritten by checkpoint), turn00 already exists
    reports, turn = [], 0
    last = {g: -1 for g in ok}          # Last turn each slot was swapped
    used = {g: set() for g in ok}      # Parts **already used in this chain** per slot (including failed), never repeat
    askip = {}                          # Count of consecutive batch skip predictions (candidate pool depletion, W3-10)
    dead = {}                          # Count of consecutive batch failures per slot
    # veto: candidates I vetoed after reviewing renders (see review_chain.py).
    # All five gates are geometric; passing doesn't mean "visual change matches instruction" --
    # instruction says "replace with a hat", put it on, can't tell it's a hat from any angle, geometrically perfect.
    # Manual review: I look at images, conclusions fed back here, on rerun these candidates won't be selected.
    tries_left = n_turns * 4           # Max attempts limit to prevent spinning when all fail

    # Resumption: rounds 1-6 already approved, only want re-run from round 7+ (e.g., given yaw_fix).
    # Checkpoint stores **in-memory original** (pickle), not loaded from GLB—GLB is float32,
    # round-trip has 1e-7 differences, strict bitwise replay breaks. Bitwise determinism (sampling seeded)
    # guarantees "resume from checkpoint" and "from start" are identical, so this shortcut is safe.
    if resume_from and resume_from > 1:
        ck = _ckpt_load(out_dir, resume_from - 1)
        pieces, state = ck["pieces"], ck["state"]
        used, dead, last = ck["used"], ck["dead"], ck["last"]
        # ok (swappable slots list) must also be restored -- if not, removed slots revive on resume
        ok = list(ck.get("ok", ok))
        frames = ck.get("frames", frames)
        mount_types = ck.get("mount_types", mount_types)
        virtual = ck.get("virtual", virtual)
        slot_diag = ck.get("slot_diag", slot_diag)
        reports = ck["reports"]
        rng.setstate(ck["rng"])
        # tries_left is spin-prevention budget, **must NOT restore as-is** -- recalculate each round in loop
        # n_turns=K, saved budget decreases monotonically; restore at step 5 becomes 0, loop exits once without running
        # "normally" (plus fake render path). Recalc based on remaining rounds.
        tries_left = (n_turns - (resume_from - 1)) * 4
        turn = resume_from - 1
        import glob as _g
        stale = [f for f in _g.glob(os.path.join(out_dir, "turn*.glb"))
                 if int(os.path.basename(f)[4:6]) >= resume_from]
        stale += [f for f in _g.glob(os.path.join(out_dir, "ckpt", "turn*.pkl"))
                  if int(os.path.basename(f)[4:6]) >= resume_from]
        for f in stale:
            os.remove(f)
        # Clear render cache with skipped rounds (strict > resume_from: this round's faces/host/cand cache
        # geometry unchanged, keep for reuse; later rounds' cache is abandoned branch geometry, agent sees wrong image; review #5)
        import re as _re, shutil as _sh
        for f in _g.glob(os.path.join(out_dir, "img", "*turn[0-9][0-9]*")):
            _m = _re.search(r"turn(\d{2})", os.path.basename(f))
            if _m and int(_m.group(1)) > resume_from:
                (_sh.rmtree(f, ignore_errors=True) if os.path.isdir(f) else os.remove(f))
                stale.append(f)
        print(f"  Resume from round {resume_from} (checkpoint turn{turn:02d}, cleaned {len(stale)} stale artifacts)",
              flush=True)

    # veto must merge **after** checkpoint restore—restore overwrites used entirely, merge before wastes effort.
    # (tested: serve veto then re-run, swaps same candidate.)
    if veto:
        vt = ((_jload(veto, {}) or {}) if isinstance(veto, str) else veto) or {}
        for g_, lst in vt.items():
            if g_ in used:
                used[g_].update((o, p) for o, p in lst)
        print(f"  Vetoed {sum(len(v) for v in vt.values())} candidates", flush=True)
    _uxp = os.path.join(out_dir, "ckpt", "used_extra.json")
    for g_, lst in (_jload(_uxp, {}) or {}).items():
        used.setdefault(g_, set()).update((o, p) for o, p in lst)

    while turn < n_turns and tries_left > 0:
        tries_left -= 1
        if op and op.get("kind") == "retexture" and turn + 1 == (stop_after or turn + 1):
            g = op["slot"]
            if g not in pieces or g in anchors:
                # break makes serve treat this round as assembly failure and wind up (W3-20 code review finding):
                # raise RuntimeError goes to op_failed branch, agent tries different slot
                raise RuntimeError(f"Material failed: {g} not editable slot (anchor or missing)")
            hook = op.get("hook")               # serve provides: render->redraw->bake, return baked glb
            if hook is None:
                raise RuntimeError("retexture only available in serve mode")
            cur = state.get(g, dict(caption=f"the {g.replace('_', ' ')} of the object"))
            try:
                baked_glb = hook(turn + 1, g, pieces[g], op["prompt"])
            except Exception as e:
                raise RuntimeError(f"Material operation failed: {type(e).__name__}: {e}")
            q = trimesh.load(baked_glb, process=False, force="mesh")
            v_old = np.asarray(pieces[g].vertices)
            # Placed part vertices are float64, glb stores only float32, one export truncates to ~5e-08;
            # bitwise-equal assertion explodes on replace/add slots (W3-6 reported X cell blocker).
            # Tolerance check geometry unchanged, then write old vertices back, ensure checkpoint geometry stays bitwise.
            assert np.allclose(np.asarray(q.vertices), v_old, atol=1e-5), "retexture changed geometry, rejected"
            assert np.array_equal(np.asarray(q.faces), np.asarray(pieces[g].faces))
            q.vertices = v_old.copy()
            pieces[g] = q
            turn += 1
            rep = dict(turn=turn, slot=g, op="retexture", scale=1.0, gap_after=0.0,
                       size_ratio=1.0, src="tex", added_oid="", added_pid="",
                       added=op["prompt"], removed=cur["caption"], focus=_focus(pieces, g),
                       connected=True, rejected=[], prompt=op["prompt"],
                       instruction=f"Change the material of {phrase(cur['caption'])} to {op['prompt']}.")
            # Material info in caption must stay in **first clause**: phrase()/head_clause takes only first segment,
            # appended to end gets trimmed, subsequent remove/replace instructions carry stale color words (W3-23 reported)
            from build_replace_chains import head_clause as _hc
            state[g] = dict(state.get(g, {}),
                            caption=f"{_hc(str(cur['caption'])).strip().rstrip('. ')} retextured in {op['prompt']}")
            last[g] = turn
            export(turn)
            reports.append(rep)
            _ckpt_save(out_dir, turn, dict(pieces=pieces, rng=rng.getstate(),
                                           used=used, dead=dead, last=last, state=state,
                                           ok=ok, tries_left=tries_left, reports=reports,
                                           frames=frames, mount_types=mount_types, virtual=virtual,
                                           slot_diag=slot_diag))
            print(f"  Turn {turn:2d} [{g:10s}] Material → {op['prompt'][:48]}", flush=True)
            if stop_after is not None and turn >= stop_after:
                break
            continue
        if op and op.get("kind") == "remove" and turn + 1 == (stop_after or turn + 1):
            g = op["slot"]
            if g not in pieces or g in anchors:
                print(f"  ! Delete failed: {g} not a deletable slot", flush=True); break
            cur = state.get(g, dict(caption=f"the {g.replace('_', ' ')} of the object"))
            focus = _focus(pieces, g)
            pieces.pop(g)
            if g in ok:
                ok.remove(g)
            last.pop(g, None)
            turn += 1
            rep = dict(turn=turn, slot=g, op="remove", scale=0.0, gap_after=0.0, size_ratio=0.0,
                       src="-", added_oid="", added_pid="", added="", removed=cur["caption"],
                       focus=focus, connected=is_connected_virtual(pieces, diag, virtual),
                       rejected=[], instruction=f"Remove {phrase(cur['caption'])}.")
            export(turn)
            reports.append(rep)
            _ckpt_save(out_dir, turn, dict(pieces=pieces, rng=rng.getstate(),
                                           used=used, dead=dead, last=last, state=state,
                                           ok=ok, tries_left=tries_left, reports=reports,
                                           frames=frames, mount_types=mount_types, virtual=virtual,
                                           slot_diag=slot_diag))
            print(f"  Turn {turn:2d} [{g:10s}] Remove → {cur['caption'][:48]}", flush=True)
            if stop_after is not None and turn >= stop_after:
                break
            continue
        if op and op.get("kind") == "add" and turn + 1 == (stop_after or turn + 1):
            face = op["face"]
            query = op["query"]
            gname = op.get("slot") or f"add{turn + 1:02d}_{_slug(query)}"
            fkey = f"add:{query}"
            used.setdefault(fkey, set())
            if veto:
                vt = ((_jload(veto, {}) or {}) if isinstance(veto, str) else veto) or {}
                used[fkey].update((o, p) for o, p in vt.get(fkey, []))
            cur = dict(caption=query, slot=gname, bbox=[[0, 0, 0], [1, 1, 1]],
                       queries=[query], accepts=[], rejects=[])
            cands = lib.rank_replacements(cur, rng, exclude_oid=oid, topk=max_tries,
                                          exclude=used[fkey], sig_free=True) or []
            rest = trimesh.util.concatenate([m for s_, m in pieces.items()])
            tree = cKDTree(samples(rest, N_REST))
            others = (trimesh.util.concatenate([m for s_, m in pieces.items() if s_ != face["piece"]])
                      if len(pieces) > 1 else None)
            otree = cKDTree(samples(others, N_REST)) if others is not None else None
            placed, tried = None, []
            loaded, debris = {}, {}
            for cand in cands:
                key = (cand["oid"], cand["pid"])
                try:
                    if part_glb_size(cand) > 120 * 1024 * 1024:
                        print(f"  Candidate {cand['oid'][:8]}/{cand['pid']} glb over 120MB, skip before load", flush=True)
                        continue
                    q0 = part_mesh(cand)
                    if len(q0.faces) > 400000:
                        print(f"  Candidate {cand['oid'][:8]}/{cand['pid']} face count {len(q0.faces)} too large, skip", flush=True)
                        continue
                    dg = max_component_gap(q0) / max(float(np.linalg.norm(q0.extents)), 1e-9)
                except Exception as e:
                    # Completely flat mesh (mat/sheet) crashes qhull, whole op hangs (D2 reported)
                    print(f"  Candidate {cand['oid'][:8]}/{cand['pid']} cannot load: {type(e).__name__}", flush=True)
                    continue
                loaded[key] = q0
                debris[key] = dg
            cands = [c for c in cands if (c["oid"], c["pid"]) in loaded]
            if cand_hook and getattr(cand_hook, "batch", None):
                cand_hook.batch(turn + 1, gname, "plane(add)",
                                [(c, loaded[(c["oid"], c["pid"])]) for c in cands
                                 if debris[(c["oid"], c["pid"])] <= DEBRIS_TAU], rest)
            for cand in cands:
                q0 = loaded[(cand["oid"], cand["pid"])]
                if debris[(cand["oid"], cand["pid"])] > DEBRIS_TAU:
                    tried.append(f"has debris {debris[(cand['oid'], cand['pid'])]:.3f}"); continue
                verdict = (cand_hook(turn + 1, gname, "plane(add)", cand, q0, len(tried) + 1, rest)
                           if cand_hook else None)
                if verdict and verdict[0] == "skip":
                    tried.append(f"pre-judge skip:{verdict[1]}"); continue
                side = (verdict[1] if verdict else None) or "bottom"
                yx = (yaw_fix or {}).get(str(turn + 1), 0.0)
                q, rep = initial_pose_add(q0, face, diag, size=op.get("size", 0.5),
                                          at=tuple(op.get("at", (0.0, 0.0))), side=side,
                                          clear=op.get("clear"))
                if yx:
                    ctr = q.vertices.mean(0)
                    q.apply_translation(-ctr)
                    q.apply_transform(_T(rot_about(np.asarray(face["n"]), np.deg2rad(float(yx)))))
                    q.apply_translation(ctr)
                gap = float(tree.query(samples(q, N_PART))[0].min()) / diag
                if gap > CONTACT_TAU and face.get("kind") != "ground":
                    tried.append(f"Gap {gap:.3f}"); continue
                if otree is not None:
                    pen = penetration_frac(q, otree, diag)
                    if pen > 0.15:
                        tried.append(f"Penetrates other part {pen:.2f}"); continue
                placed = (q, rep, cand, gap, 1.0); break
            if placed is None:
                for c in (cands or []):
                    used[fkey].add((c["oid"], c["pid"]))
                _used_persist(out_dir, fkey, {(c["oid"], c["pid"]) for c in (cands or [])})
                print(f"  Add [{query}] all {len(cands)} candidates failed: {tried}", flush=True)
                # No longer auto-fetch next batch: wrong drop point/size wastes batches (D2 restarted serve 4 times),
                # hand back to serve with op_failed, let reviewer change drop point / description / face
                raise RuntimeError(f"Add operation failed: {tried}")
            q, rep, cand, gap, ratio = placed
            used[fkey].add((cand["oid"], cand["pid"]))
            pieces[gname] = q
            if face.get("kind") == "ground" and gap > CONTACT_TAU:
                # Ground piece doesn't touch any part: attach virtual edge (same mechanism as registry virtual edge, persists with virtual in ckpt)
                _fr, _vinfo = virtual_frame(gname, pieces, diag)
                if _vinfo is not None:
                    virtual[gname] = _vinfo
            ok.append(gname)
            used[gname] = set([(cand["oid"], cand["pid"])])
            frames[gname] = (np.asarray(face["free_c"]), np.asarray(face["n"]), float(rep["mount_radius"]), 0)
            mount_types[gname] = "plane"
            slot_diag[gname] = float(np.linalg.norm(q.extents))
            turn += 1
            fp = face["piece"]
            if fp.startswith("add"):        # Stacked on previously added part: use that part's description, not slot name
                fdesc = f"the top of {phrase(state.get(fp, {}).get('caption') or fp)}"
            else:
                fdesc = f"the {face['kind']} of the {fp.replace('anchor_', '').replace('_', ' ')}"
            rep.update(turn=turn, slot=gname, mode="added", gap_after=gap, size_ratio=ratio,
                       focus=_focus(pieces, gname), connected=is_connected_virtual(pieces, diag, virtual), removed="", added=cand["cap"],
                       src=f"{cand['oid'][:8]}/{cand['pid']}", added_oid=cand["oid"],
                       added_pid=cand["pid"], rejected=tried, query=query,
                       instruction=f"Add {phrase(cand['cap'])} on {fdesc}.")
            state[gname] = dict(caption=cand["cap"], bbox=_bbox(q), slot=gname,
                                queries=[query], accepts=[], rejects=[])
            last[gname] = turn
            export(turn)
            reports.append(rep)
            _ckpt_save(out_dir, turn, dict(pieces=pieces, rng=rng.getstate(),
                                           used=used, dead=dead, last=last, state=state,
                                           ok=ok, tries_left=tries_left, reports=reports,
                                           frames=frames, mount_types=mount_types, virtual=virtual,
                                           slot_diag=slot_diag))
            print(f"  Turn {turn:2d} [{gname:10s}] Add scale {rep['scale']:.2f} gap {gap:.4f} "
                  f"rejected {len(tried)} candidates → {cand['cap'][:48]}", flush=True)
            if stop_after is not None and turn >= stop_after:
                break
            continue
        # **Prefer least-recently swapped slot**, not pure random.
        # Pure random result on bicycle chain: rear wheel swapped 4x, seat 5x, basket only rounds 2-3,
        # after round 11 mostly just seat and wheels -- those two occupy little screen,
        # 5 of 20 rounds show no change. After fair rotation each slot gets moved.
        # oldest only checks slots in ok: skip slots written by retexture hold global minimum after last,
        # empty pool rng.choice([]) crashes outright (W3-17 reported)
        if not ok:
            raise RuntimeError("No swappable slots (all deleted): only remove / retexture / stop")
        oldest = min(last.get(g_, 0) for g_ in ok)
        pool = [g for g in ok if last.get(g_ := g, 0) <= oldest + 1] or list(ok)  # Keep some randomness, not rigid round-robin
        g = rng.choice(pool)
        pin_t = (pin or {}).get(str(turn + 1))
        if pin_t and len(pin_t) > 2 and pin_t[2] in ok:
            g = pin_t[2]            # yaw redo: slot also pinned, rng replay doesn't follow if selecting different slot (B1 reported)
        cur = state[g]
        cands = lib.rank_replacements(cur, rng, exclude_oid=oid, topk=max_tries,
                                      exclude=used[g]) or []
        if os.environ.get("PX_DEBUG_CANDS"):   # ranked list before the head filter / geometric gates
            print(f"  [cands] Turn {turn + 1} {g}: " + ", ".join(
                f"{c['oid'][:8]}/{c['pid']}({c['score']:.2f})" for c in cands), flush=True)
        # Same-type spinning prevention (Agent A reported): retrieval follows slot examples, boot swaps boot,
        # resulting instruction is "Replace a futuristic armored boot with a futuristic armored boot",
        # no change visible. Reject candidates with identical caption heads -- swap means different things.
        # Comparison must strip preposition-starting adjective tails: bicycle chain "a wooden spoked wheel from the
        # cart" to "a wooden spoked wheel from the rear axle of a wooden cart"
        # full sentence differs, head identical, instruction-level still no change.
        import re as _re
        _hd = lambda s: _re.split(r"\s+(?:from|of|with|for|on|in|at)\b",
                                  s, 1)[0].strip().lower()
        cur_head = phrase(cur["caption"])
        cands = [c for c in cands if _hd(phrase(c["cap"])) != _hd(cur_head)]
        # yaw re-pinned candidate (defect reported by Agent B: rotation placement failure falls
        # silently to next candidate, "fine-tune same" becomes "replace part"). After pinning, either
        # install successfully or fail loudly, with reviewer deciding angle retry or veto -- no silent replacement.
        if pin_t:
            batch = list(cands)
            cands = [c for c in cands if c["oid"] == pin_t[0] and c["pid"] == pin_t[1]]
            if not cands:
                # Pinned candidate may appear in later batch (after first 8 all fail, then its turn, B4 test):
                # treat this batch as "all failed", continue drawing next batch; only fail when tries_left exhausted
                for c in batch:
                    used[g].add((c["oid"], c["pid"]))
                if tries_left > 0:
                    continue
                raise RuntimeError(
                    f"yaw pinned candidate {pin_t[0][:8]}/{pin_t[1]} not found in any batch -- "
                    f"deterministic replay failed, check if used/veto excluded it")
        rest = trimesh.util.concatenate([m for s, m in pieces.items() if s != g])
        tree = cKDTree(samples(rest, N_REST))
        placed, tried = None, []
        # First load all candidates in batch, pass debris check; in serve mode all together for agent
        # pre-judgment (one image, one answer), no longer asking per candidate -- 10-20s per round-trip, B1 spent 20min/round
        loaded, debris = {}, {}
        for cand in cands:
            key = (cand["oid"], cand["pid"])
            # All batch candidates reside simultaneously: single 600k-face library part pushes serve to 150GB (X/5ebf7367 test),
            # like add path, block by file size before load, blocked ones marked used and won't return
            if part_glb_size(cand) > 120 * 1024 * 1024:
                print(f"  Candidate {cand['oid'][:8]}/{cand['pid']} GLB exceeds 120MB, skipping before load", flush=True)
                used[g].add(key); tried.append("GLB limit")
                continue
            q0 = part_mesh(cand)
            if len(q0.faces) > 400000:
                print(f"  Candidate {cand['oid'][:8]}/{cand['pid']} face count {len(q0.faces)} too large, skipping", flush=True)
                used[g].add(key); tried.append("face count limit")
                continue
            loaded[key] = q0
            debris[key] = max_component_gap(q0) / max(float(np.linalg.norm(q0.extents)), 1e-9)
        cands = [c for c in cands if (c["oid"], c["pid"]) in loaded]
        if cand_hook and getattr(cand_hook, "batch", None):
            cand_hook.batch(turn + 1, g, mount_types.get(g) or "?",
                            [(c, loaded[(c["oid"], c["pid"])]) for c in cands
                             if debris[(c["oid"], c["pid"])] <= DEBRIS_TAU], rest)
        for cand in cands:
            q0 = loaded[(cand["oid"], cand["pid"])]
            dgap = debris[(cand["oid"], cand["pid"])]
            if dgap > DEBRIS_TAU:
                tried.append(f"has debris {dgap:.3f}"); continue
            yx = (yaw_fix or {}).get(str(turn + 1), 0.0)
            mt = mount_types.get(g)
            # Pre-judge candidate (serve mode): agent reads candidate part image and says whether
            # it can fit on this mount surface by scaling, and which side the mount end is. Skip ones don't enter geometry gate.
            verdict = (cand_hook(turn + 1, g, mt or "?", cand, q0, len(tried) + 1, rest)
                       if cand_hook else None)
            if verdict and verdict[0] == "skip":
                tried.append(f"pre-judge skip:{verdict[1]}"); continue
            side = verdict[1] if verdict else None
            if mt == "plane":
                q, rep = initial_pose_plane(q0, pieces[g], rest, diag, frame=frames.get(g),
                                            slot_diag=slot_diag[g], side=side or "bottom",
                                            yaw_extra=yx)
            else:
                q, rep = initial_pose(q0, pieces[g], rest, diag, frame=frames.get(g),
                                      slot_diag=slot_diag[g], yaw_extra=yx, side=side)
                if q is not None:
                    rep = dict(rep, mount_type=mt or "disc")
            if q is None:
                tried.append("no contact surface"); continue
            if rep["scale"] <= 0.2501 or rep["scale"] >= 3.999:
                tried.append(f"scale at boundary {rep['scale']:.2f}"); continue
            vinfo = virtual.get(g)          # virtual edge slot: not conforming, preserve original gap
            if vinfo is None:
                try:
                    snapped = snap_to(MeshGeometry.from_trimesh(rest), MeshGeometry.from_trimesh(q))
                    q = q.copy()
                    q.apply_translation(snapped.to_trimesh().vertices.mean(0) - q.vertices.mean(0))
                except SnapRefused:
                    tried.append("snap refused"); continue
                except Exception as e:
                    tried.append(type(e).__name__); continue
            gap = float(tree.query(samples(q, N_PART))[0].min()) / diag
            gap_lim = CONTACT_TAU + (vinfo["gap_rel"] if vinfo else 0.0)
            if gap > gap_lim:
                tried.append(f"gap {gap:.3f}"); continue
            ratio = float(np.linalg.norm(q.extents)) / max(slot_diag[g], 1e-9)
            if ratio > size_ratio:
                tried.append(f"too large {ratio:.1f}x"); continue
            trial = dict(pieces); trial[g] = q
            if not is_connected_virtual(trial, diag, virtual):
                tried.append("disassembles"); continue
            placed = (q, rep, cand, gap, ratio); break
        if placed is None and pin_t:
            raise RuntimeError(
                f"yaw re-pin failed: original candidate {pin_t[0][:8]}/{pin_t[1]} cannot install after rotation "
                f"({tried}). Try another angle or veto to replace -- no silent replacement.")
        if placed is None:
            # **Failed candidates must be recorded.** If not, next retrieval returns same 8 as-is
            # (library scoring deterministic, sampling shuffles order only), retrying same batch over,
            # wasting time without reaching full rounds -- test shows one chain 10 of 20 rounds, another
            # only 4 rounds in 5400s before killed.
            for c in (cands or []):
                used[g].add((c["oid"], c["pid"]))
            _used_persist(out_dir, g, {(c["oid"], c["pid"]) for c in (cands or [])})
            # Entire batch is agent pre-judge skip: semantically unwanted not geometrically impossible: not slot failure,
            # only consumes tries_left (B3 reported: turret 3 batches all skip treated as "consecutive failure" removed)
            # Pre-filter elimination (agent skip / debris / face count / GLB limit) shouldn't mark slot dead:
            # only if candidate truly enters placement and still fails does it show slot geometry problem (W3-65: one batch agent skip
            # mixed one "face count limit" marked dead, removing good slots prematurely)
            _PRE = ("pre-judge skip", "has debris", "face count limit", "GLB limit", "scale at boundary")  # size mismatch is candidate's problem, don't mark slot dead (review #8)
            all_skip = bool(tried) and all(t.startswith(_PRE) for t in tried)
            if not all_skip:
                dead[g] = dead.get(g, 0) + 1
                askip.pop(g, None)
            else:
                # Semantic skip not counted as slot failure, but infinite loop when candidate pool only junk (W3-10 reported):
                # also remove slot if 5 consecutive batches all skip
                askip[g] = askip.get(g, 0) + 1
                if askip[g] >= 5:
                    ok = [x for x in ok if x != g]
                    last.pop(g, None)
                    print(f"  ! Slot {g}: {askip[g]} consecutive batches all pre-judged skip, candidate pool exhausted, removed from chain; remaining {ok}", flush=True)
                    if len(ok) < 2:
                        print(f"  ! Fewer than 2 replaceable slots, ending early (completed {turn} rounds)", flush=True)
                        break
                    continue
            print(f"  Slot {g}: all {max_tries} candidates failed (attempt {dead.get(g, 0)}"
                  f"{', all pre-judged skip' if all_skip else ''}): {tried}", flush=True)
            if dead.get(g, 0) >= 3:
                # Three consecutive batches won't install, slot's contact geometry itself hard to place, don't waste more time
                ok = [x for x in ok if x != g]
                last.pop(g, None)
                print(f"  ! Slot {g}: 3 consecutive batch failures, removed from chain; remaining {ok}", flush=True)
                if len(ok) < 2:
                    print(f"  ! Fewer than 2 replaceable slots, ending early (completed {turn} rounds)", flush=True)
                    break
            continue
        q, rep, cand, gap, ratio = placed
        # Per-round similarity transform (FORMULATION section 8): library part -> placement 4x4, part vertices one-to-one,
        # using centered least squares exact solution; record in place_report, chain assembly/replay needs no further inversion
        try:
            src_v = np.asarray(part_mesh(cand).vertices)
            dst_v = np.asarray(q.vertices)
            if src_v.shape == dst_v.shape and len(src_v) >= 4:
                idx_t = np.linspace(0, len(src_v) - 1, min(2000, len(src_v))).astype(int)
                A0, B0 = src_v[idx_t], dst_v[idx_t]
                ca, cb = A0.mean(0), B0.mean(0)
                A1, B1 = A0 - ca, B0 - cb
                sc_ = float(np.sqrt((B1 ** 2).sum() / max((A1 ** 2).sum(), 1e-12)))
                U, _, Vt = np.linalg.svd(A1.T @ B1)
                R_ = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
                T4 = np.eye(4); T4[:3, :3] = sc_ * R_; T4[:3, 3] = cb - sc_ * (R_ @ ca)
                rep["transform"] = [[round(float(x), 8) for x in row] for row in T4]
        except Exception:
            pass
        used[g].add((cand["oid"], cand["pid"]))
        pieces[g] = q
        turn += 1
        rep.update(turn=turn, slot=g, mode="snapped", gap_after=gap, size_ratio=ratio,
                   ds=cand.get("ds", "pv"), focus=_focus(pieces, g),
                   connected=True, removed=cur["caption"], added=cand["cap"],
                   src=f"{cand['oid'][:8]}/{cand['pid']}",
                   added_oid=cand["oid"], added_pid=cand["pid"], rejected=tried,
                   instruction=f"Replace {phrase(cur['caption'])} with {phrase(cand['cap'])}.")
        state[g] = dict(caption=cand["cap"], bbox=cur["bbox"], slot=g,
                        queries=cur["queries"],
                        accepts=cur["accepts"], rejects=cur["rejects"])
        last[g] = turn
        export(turn)
        reports.append(rep)
        _ckpt_save(out_dir, turn, dict(pieces=pieces, rng=rng.getstate(),
                                       used=used, dead=dead, last=last, state=state,
                                       ok=ok, tries_left=tries_left, reports=reports,
                                       frames=frames, mount_types=mount_types, virtual=virtual,
                                       slot_diag=slot_diag))
        if stop_after is not None and turn >= stop_after:
            print(f"  Stopped at round {turn} (--stop-after, per-round review mode)", flush=True)
            break
        print(f"  Round {turn:2d} [{g:10s}] scale {rep['scale']:.2f} size ratio {ratio:.1f}x "
              f"gap {gap:.4f}  replaced {len(tried)} candidates  → {cand['cap'][:48]}", flush=True)
    _jdump(reports, os.path.join(out_dir, "place_report.json"), indent=1, ensure_ascii=False)
    write_provenance(oid, reports, out_dir)


def write_provenance(host_oid, reports, out_dir):
    """Write source of all assets used by each chain -- publication credit and link-only distribution both rely on it.

    Our 32-bit object ids **are exactly Objaverse 1.0 uids**, i.e., Sketchfab model uids
    (37,493 ids match Objaverse full table at 100%; official code in
    `partverse/encode_latent_from_imgs.py:82` also uses
    `https://sketchfab.com/3d-models/{id}` to query metadata). So source chain is complete,
    needs no extra mapping table.

    Why it must be written: CC 4.0 "Share" definition includes reproduction and public display,
    **our rendered images are the ground truth and core deliverable**, so even link-only without geometry,
    CC-BY asset credit obligation is triggered. CC BY 4.0 section 3(a)(2) allows using "link pointing to
    resource with necessary information" to fulfill it, so one centralized credits file suffices, no need to print on each image.
    """
    used = {host_oid: dict(role="host", turns=[])}
    for r in reports:
        src = r.get("src", "")
        full = r.get("added_oid") or (src.split("/")[0] if "/" in src else None)
        pid = src.split("/")[1] if "/" in src else None
        if not full:
            continue
        used.setdefault(full, dict(role="replacement", turns=[], parts=set(), ds=r.get("ds", "pv")))
        used[full]["turns"].append(r["turn"])
        if pid:
            used[full].setdefault("parts", set()).add(pid)
    out = []
    for o, v in used.items():
        e = dict(role=v["role"], turns=sorted(v["turns"]), parts=sorted(v.get("parts", [])))
        if v.get("ds", "pv") == "hy3d":
            e.update(dataset="hy3d", asset_id=o)     # HY3D ids are not Objaverse uids
        else:
            e.update(objaverse_uid=o, sketchfab_url=f"https://sketchfab.com/3d-models/{o}")
        out.append(e)
    json.dump(dict(
        note="objaverse_uid is Sketchfab model uid. Per-asset license and author refer to "
             "Objaverse 1.0 annotations (including username/displayName/profileUrl). "
             "This benchmark distributes only assembly recipes and renders, not assets themselves.",
        assets=sorted(out, key=lambda x: (x["role"] != "host", (x.get("objaverse_uid") or x.get("asset_id", ""))))),
        open(os.path.join(out_dir, "provenance.json"), "w"), indent=1, ensure_ascii=False)
    print(f"  Source records: {len(out)} assets -> provenance.json", flush=True)


# ------------------------------------------------------------------ check

def stage_check(oid, out_dir, res=256):
    """Dual grid surface voxel intersection -- detect visible interpenetration."""
    import torch
    from o_voxel.convert import mesh_to_flexible_dual_grid

    _, _, seg = object_pieces(oid)
    vmin, vmax = seg.bounds
    center, scale = (vmin + vmax) / 2, 0.99999 / (vmax - vmin).max()
    AABB = torch.tensor([[-0.6] * 3, [0.6] * 3], dtype=torch.float32)

    def voxels(m):
        v = torch.from_numpy(((np.asarray(m.vertices) - center) * scale).astype(np.float32))
        f = torch.from_numpy(np.asarray(m.faces)).long()
        idx, _, _ = mesh_to_flexible_dual_grid(vertices=v, faces=f, grid_size=res, aabb=AABB)
        return {tuple(x) for x in idx.cpu().numpy().tolist()}

    rep = json.load(open(os.path.join(out_dir, "place_report.json")))
    for r in rep:
        p = os.path.join(out_dir, f"turn{r['turn']:02d}.glb")
        sc = trimesh.load(p, process=False)
        geoms = dict(sc.geometry) if hasattr(sc, "geometry") else {}
        key = f"slot_{r['slot']}"
        if key not in geoms:
            r["penetration"] = None
            continue
        newp = geoms[key]
        rest = trimesh.util.concatenate([g for k, g in geoms.items() if k != key])
        Vq, Vr = voxels(newp), voxels(rest)
        inter = Vq & Vr
        r["penetration"] = len(inter) / max(len(Vq), 1)
        r["n_vox_new"] = len(Vq)
        print(f"  Round {r['turn']:2d} penetration {r['penetration']:.4f} "
              f"(new part {len(Vq)} voxels, intersect {len(inter)})", flush=True)
    json.dump(rep, open(os.path.join(out_dir, "place_report.json"), "w"),
              indent=1, ensure_ascii=False)


# ------------------------------------------------------------------ render

def free_gpus(min_free_mb=20000):
    """Which local GPUs are free (by available memory). Rendering is the only GPU step in full pipeline, directly uses local GPUs.
    When caller has restricted GPUs via CUDA_VISIBLE_DEVICES, only pick from those (else worker overrides restriction,
    and runs on GPUs reserved for other jobs)."""
    import subprocess
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None and vis.strip():
        try:
            return [int(x) for x in vis.split(",") if x.strip() != ""]
        except ValueError:
            pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        g = []
        for ln in out.strip().splitlines():
            i, used, tot = [int(x) for x in ln.split(",")]
            if tot - used >= min_free_mb:
                g.append(i)
        return g or [0]
    except Exception:
        return [0]


def stage_render(out_dir, img_dir, res=420, samples_n=24, from_turn=None, workers=None):
    """Render per-round state. Both time-saving switches rely on blender_kit's existing capabilities:

    * **Incremental** (`from_turn`): after review correction at round k, states before k unchanged, skip re-render.
      Paired with `--resume-from`.
    * **Parallel** (`workers`): blender_kit manifest has built-in `--rank/--world` slicing
      (jobs[rank::world]), spawn N blender processes each taking one slice, each binding one local free GPU.
      Inside each job already **load scene once, camera at 4 positions**, no world reloading.
    """
    import subprocess
    from local_paths import RENDER_SCRIPT as KIT
    from local_paths import BLENDER
    jobs = []
    for f in sorted(os.listdir(out_dir)):
        if not f.endswith(".glb") or f.endswith("_cut.glb"):
            continue
        t = int(f[4:6]) if f[4:6].isdigit() else None
        if from_turn is not None and (t is None or t < from_turn):
            continue
        fr = sphere_frame(os.path.join(out_dir, f))
        jobs.append(dict(scene="mesh", mesh=os.path.join(out_dir, f),
                         material="file_embedded", outputs="rgb",
                         out_dir=os.path.join(img_dir, f[:-4]), **fr))
        record_frame(out_dir, f[:-4], fr, f)
        prev = os.path.join(out_dir, f"turn{t-1:02d}.glb") if t else None
        if prev and os.path.isfile(prev):        # previous round state, this round camera: gate reference image
            jobs.append(dict(scene="mesh", mesh=prev, material="file_embedded", outputs="rgb",
                             out_dir=os.path.join(img_dir, f[:-4] + "_prev"), **fr))
            record_frame(out_dir, f[:-4] + "_prev", fr, prev)
    if not jobs:
        print("  No states to render"); return
    os.makedirs(img_dir, exist_ok=True)
    man = os.path.join(img_dir, "man.jsonl")
    open(man, "w").write("\n".join(json.dumps(j) for j in jobs))

    gpus = free_gpus()
    W = min(workers or len(gpus), len(jobs))
    base = [BLENDER, "-b", "--python", KIT, "--", "--manifest", man,
            "--hdri", "studio.exr", "--hdri_strength", "1.6",
            "--trajectory", "circle", "--frames", "4", "--start_az", "35",
            "--elevation", "25", "--distance", "1.7",
            "--res", str(res), "--samples", str(samples_n), "--continue_on_error"]
    procs = []
    for r in range(W):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[r % len(gpus)]))
        procs.append(subprocess.Popen(base + ["--rank", str(r), "--world", str(W)],
                                      env=env, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL))
    fails = sum(p.wait() != 0 for p in procs)
    n = sum(len(fs) for _, _, fs in os.walk(img_dir) if fs)
    print(f"  Rendered {len(jobs)} states × {W} workers (GPUs {gpus[:W]}), produced {n} files"
          + (f", {fails} workers exited abnormally" if fails else ""))



# ------------------------------------------------------------------ serve

def farm_submit(spool, payload, timeout, tag):
    """texfarm: submit a job to global resident worker spool and wait for result.
    Difference from RestyleServer/TexServer: don't spawn process, don't clean spool, don't check proc --
    worker managed resident by texfarm.sh (model not repeatedly loaded, 600s fake timeout obsolete).
    job name uses time_ns ensuring unique per concurrent chain; multiple chains same GPU texture jobs naturally serialize in queue."""
    import time as _t
    os.makedirs(spool, exist_ok=True)
    p = os.path.join(spool, f"job_{_t.time_ns()}.json")
    json.dump(payload, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
    t0 = _t.time()
    started = None
    while not os.path.exists(p + ".done"):
        if started is None and os.path.exists(p + ".started"):
            started = _t.time()      # worker started: timeout counted from work start, queue doesn't consume budget (review #11; old worker no marker uses queue timeline)
        base = started if started is not None else t0
        budget = timeout if started is not None else timeout * 4
        if _t.time() - base > budget:
            raise RuntimeError(f"{tag} farm timeout {budget}s ({'compute' if started else 'queue'} phase; check texfarm.sh status)")
        _t.sleep(0.3)
    msg = open(p + ".done").read()
    if not msg.startswith("ok"):
        raise RuntimeError(f"{tag} farm failed: {msg}")
    return _t.time() - t0


class RestyleServer:
    """Driver seat for resident Qwen-Image-Edit restyle worker (spool protocol same as TexServer)."""

    def __init__(self, spool, gpu=None):
        import subprocess
        self.spool = spool
        os.makedirs(spool, exist_ok=True)
        for f in os.listdir(spool):
            os.remove(os.path.join(spool, f))
        env = dict(os.environ)
        env.pop("LD_PRELOAD", None)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.log = open(os.path.join(spool, "worker.log"), "a")
        self.proc = subprocess.Popen(
            [os.environ.get("PXFORM_RESTYLE_PYTHON", __import__("local_paths").ENCODER_PYTHON),
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "restyle_serve.py"), "--spool", spool],
            env=env, stdout=self.log, stderr=subprocess.STDOUT)
        self._n = 0

    def restyle(self, img, prompt, out, timeout=600, seed=0):
        import time
        self._n += 1
        p = os.path.join(self.spool, f"job_{self._n:04d}.json")
        tmp = p + ".tmp"
        json.dump(dict(img=img, prompt=prompt, out=out, seed=seed), open(tmp, "w"))
        os.replace(tmp, p)
        t0 = time.time()
        while not os.path.exists(p + ".done"):
            if self.proc.poll() is not None:
                raise RuntimeError("restyle worker died, check its output")
            if time.time() - t0 > timeout:
                raise RuntimeError(f"restyle timeout {timeout}s")
            time.sleep(0.3)
        msg = open(p + ".done").read()
        if not msg.startswith("ok"):
            raise RuntimeError(f"restyle failed: {msg}")
        return time.time() - t0

    def stop(self):
        open(os.path.join(self.spool, "STOP"), "w").write("")
        try:
            self.proc.wait(timeout=60)
        except Exception:
            self.proc.kill()


class TexServer:
    """Driver seat for resident TRELLIS.2 texturing worker (same pattern as RenderServer).
    Runs in trellis2_turbo environment, separated from articraft environment, communicates via spool directory."""

    def __init__(self, spool, gpu=None):
        import subprocess
        self.spool = spool
        os.makedirs(spool, exist_ok=True)
        for f in os.listdir(spool):
            os.remove(os.path.join(spool, f))
        HERE = os.path.dirname(os.path.abspath(__file__))
        env = dict(os.environ)
        env.pop("LD_PRELOAD", None)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.log = open(os.path.join(spool, "worker.log"), "a")
        self.proc = subprocess.Popen(
            [os.environ.get("PXFORM_TEX_PYTHON", __import__("local_paths").ENCODER_PYTHON), os.path.join(HERE, "tex_serve.py"),
             "--spool", spool],
            env=env, stdout=self.log, stderr=subprocess.STDOUT)
        self._n = 0

    def retexture(self, mesh, ref, out, timeout=900, seed=0):
        import time
        self._n += 1
        p = os.path.join(self.spool, f"job_{self._n:04d}.json")
        tmp = p + ".tmp"
        json.dump(dict(mesh=mesh, ref=ref, out=out, seed=seed), open(tmp, "w"))
        os.replace(tmp, p)
        t0 = time.time()
        while not os.path.exists(p + ".done"):
            if self.proc.poll() is not None:
                raise RuntimeError("texture worker died, check its output")
            if time.time() - t0 > timeout:
                raise RuntimeError(f"texture timeout {timeout}s: {mesh}")
            time.sleep(0.3)
        msg = open(p + ".done").read()
        if not msg.startswith("ok"):
            raise RuntimeError(f"texture failed: {msg}")
        return time.time() - t0

    def stop(self):
        open(os.path.join(self.spool, "STOP"), "w").write("")
        try:
            self.proc.wait(timeout=60)
        except Exception:
            self.proc.kill()


class RenderServer:
    """Driver seat for resident blender worker. Write job to spool, wait for .done receipt."""

    def __init__(self, spool, gpu=None):
        import subprocess
        self.spool = spool
        os.makedirs(spool, exist_ok=True)
        for f in os.listdir(spool):                      # clean up leftover from last run
            os.remove(os.path.join(spool, f))
        from local_paths import BLENDER
        HERE = os.path.dirname(os.path.abspath(__file__))
        env = dict(os.environ, BK_SPOOL=spool)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.proc = subprocess.Popen(
            [BLENDER, "-b", "--python", os.path.join(HERE, "blender_serve.py"), "--",
             "--hdri", "studio.exr", "--hdri_strength", "1.6",
             "--trajectory", "circle", "--frames", "4", "--start_az", "35",
             "--elevation", "25", "--distance", "1.7",
             "--res", "420", "--samples", "24"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._n = 0

    def render(self, mesh, out_dir, timeout=180, focus=None, distance=None, frame=None):
        import time
        self._n += 1
        p = os.path.join(self.spool, f"job_{self._n:04d}.json")
        job = dict(scene="mesh", mesh=mesh, material="file_embedded",
                   normalize="whole", outputs="rgb", out_dir=out_dir)
        if not focus:
            # Full view: frame by bounding sphere; when frame passed use given camera position (render turnKK_prev reference).
            # Close-up still frames by part bbox, different coordinate system, don't move.
            fr = frame or sphere_frame(mesh)
            job.update(fr)
            record_frame(os.path.dirname(os.path.abspath(mesh)),
                         os.path.basename(os.path.normpath(out_dir)), fr, mesh)
        if focus:
            # Close-up: camera aims focus.center, distance by focus.diag (blender_serve intercepts
            # this key to modify scene.center/diag). Default distance slightly larger than full view --
            # close-up diag is single part, 1.7 pushes into face, joints out of frame. Still unclear pass smaller
            # distance and render again (judge's magnifier when unsure about image).
            job.update(focus=focus, distance=(distance or 2.4))
        elif distance:
            job["distance"] = distance
        # Write temp name then rename. worker's glob sees file the moment it's created (still 0 bytes),
        # json.load reports "Expecting value: line 1 column 1", entire serve crashes -- hit this with 3 agents
        # running simultaneously. Receipt side already does this.
        tmp = p + ".tmp"
        json.dump(job, open(tmp, "w"))
        os.replace(tmp, p)
        t0 = time.time()
        while not os.path.exists(p + ".done"):
            if self.proc.poll() is not None:
                raise RuntimeError("blender worker died, check its output")
            if time.time() - t0 > timeout:
                raise RuntimeError(f"render timeout {timeout}s: {mesh}")
            time.sleep(0.2)
        msg = open(p + ".done").read()
        if not msg.startswith("ok"):
            raise RuntimeError(f"render failed: {msg}")
        return time.time() - t0

    def stop(self):
        open(os.path.join(self.spool, "STOP"), "w").write("")
        try:
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()



def stage_probe(oid, out_dir, srv, img_dir, emit):
    """Before chain starts, 'read image 1': render per-slot close-up of host with old part removed showing mount surface
    (old part contact points marked magenta), emit probe event, wait for agent to write <out_dir>/mount_types.json {slot: plane|axis|point}."""
    import time
    from scipy.spatial import cKDTree
    ini = _init_host(oid, out_dir)
    if ini is None:
        raise RuntimeError("object not in registry")
    pieces, diag, ok = ini["pieces"], ini["diag"], ini["ok"]
    pdir = os.path.join(out_dir, "probe")
    os.makedirs(pdir, exist_ok=True)
    imgs = {}
    for g in ok:
        rest_items = {k: m for k, m in pieces.items() if k != g}
        rest = trimesh.util.concatenate(list(rest_items.values()))
        rp, pp = samples(rest, N_REST), samples(pieces[g], N_PART)
        dist, _ = cKDTree(rp).query(pp)
        hit = pp[dist < CONTACT_TAU * diag]
        vinfo = ini.get("virtual", {}).get(g)
        if len(hit) == 0 and vinfo:           # virtual edge: magenta points drawn at nearest cluster to neighbor
            hit = pp[dist <= np.quantile(dist, 0.05)]
        sc = trimesh.Scene()
        for k, m in rest_items.items():
            sc.add_geometry(m, geom_name=f"slot_{k}")
        if len(hit):
            idx = np.linspace(0, len(hit) - 1, min(40, len(hit))).astype(int)
            marker = trimesh.util.concatenate(
                [trimesh.creation.icosphere(subdivisions=1, radius=0.012 * diag)
                 .apply_translation(hit[i]) for i in idx])
            marker.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[255, 0, 255, 255], emissiveFactor=[0.6, 0.0, 0.6]))
            sc.add_geometry(marker, geom_name="contact_markers")
        glb = os.path.join(pdir, f"{g}.glb")
        sc.export(glb)
        mn = np.min([m.bounds[0] for m in rest_items.values()], axis=0)
        mx = np.max([m.bounds[1] for m in rest_items.values()], axis=0)
        scale = float(max((mx - mn).max(), 1e-9))
        ctr = (pieces[g].bounds[0] + pieces[g].bounds[1]) / 2.0
        dd = (ctr - (mn + mx) / 2.0) / scale
        focus = dict(center=[float(dd[0]), float(-dd[2]), float(dd[1])],
                     diag=float(np.linalg.norm(pieces[g].extents) / scale))
        od = os.path.join(img_dir, f"probe_{g}")
        srv.render(glb, od, focus=focus, distance=2.4)
        imgs[g] = _grid([os.path.join(od, f"f{v:04d}.png") for v in range(4)],
                        os.path.join(out_dir, "review", f"probe_{g}.png"),
                        title=f"PROBE  slot={g}   host with the old part removed; "
                              f"magenta = where the old part touched the host"
                              + (f"   [VIRTUAL EDGE: no contact, anchored to nearest part '{vinfo['neighbour']}', gap {vinfo['gap_rel']:.3f}]" if vinfo else "")
                              + "\nanswer in mount_types.json: plane / axis / point")
    emit(event="probe", slots=ok, img=imgs,
         contacts={g: int(ini["frames"][g][3]) for g in ok},
         note="write mount_types.json {slot: plane|axis|point} for every slot")
    mt_p = os.path.join(out_dir, MOUNT_TYPES_FILE)
    while True:
        if os.path.isfile(mt_p):
            try:
                mt = json.load(open(mt_p))
            except Exception:
                mt = None
            if mt and all(g in mt for g in ok):
                return mt
        time.sleep(1.0)


def stage_serve(oid, out_dir, n_turns, seed=0, pool=None, img_dir=None, ops=("replace",)):
    """Per-round review resident mode (**default operation**): assembly state and search library stay in memory,
    blender resident, ~10 seconds per round for images; between rounds wait for my review instructions.

    Protocol (all via files):
      Status: <out_dir>/serve_status.jsonl  append per line {"turn":k,"img":...,"event":...}
      Command: <out_dir>/serve_cmd          I write one line, delete after reading:
            ok            round's two PASS stages both pass, continue next
            veto          semantically unacceptable—record veto, redo round (swap candidate)
            yaw <degrees> orientation wrong—record yaw_fix, redo round (same candidate rotate)
            unveto        undo this slot's latest veto—redo returns to that candidate
                          ("redo 4 times pick best" relies on this to backtrack)
            stop          finish

    Note: after veto redo, may land on **different slot** (slot selection before candidate selection);
    reviewing, check instruction in ready event, don't assume still same slot.
    """
    import time
    img_dir = img_dir or os.path.join(out_dir, "img")
    status_p = os.path.join(out_dir, "serve_status.jsonl")
    cmd_p = os.path.join(out_dir, "serve_cmd")
    veto_p = os.path.join(out_dir, "veto.json")
    yawf_p = os.path.join(out_dir, "yaw_fix.json")
    os.makedirs(out_dir, exist_ok=True)

    def emit(**kw):
        kw["t"] = time.strftime("%H:%M:%S")
        kw["ts"] = round(time.time(), 1)
        with open(status_p, "a") as f:
            f.write(json.dumps(kw, ensure_ascii=False) + "\n")
        print(f"  [serve] {kw}", flush=True)

    def wait_cmd():
        empty_since = None
        while True:
            if os.path.isfile(cmd_p):
                c = open(cmd_p).read().strip()
                if c:
                    os.remove(cmd_p)
                    return c
                # echo > create empty file first then write: if reads empty don't consume, wait until write done; if still empty after 5s treat as garbage (review #4 tested)
                empty_since = empty_since or time.time()
                if time.time() - empty_since > 5:
                    os.remove(cmd_p); empty_since = None
            else:
                empty_since = None
            time.sleep(0.5)

    # When running multiple serve instances in parallel, each must bind to one GPU—free_gpus would let all instances pick the same GPU.
    gpu = os.environ.get("PXFORM_GPU")
    srv = RenderServer(os.path.join(out_dir, "spool"),
                       gpu=(int(gpu) if gpu is not None else free_gpus()[0]))
    _lib_memo(pool)                     # warm up search library, don't count the 0.5GB loading into round 1 assembly
    cand_p = os.path.join(out_dir, "cand_cmd")
    import hashlib
    import slot_registry as _SR
    _roles = (_SR.get_one(oid) or {}).get("roles", {})

    def _sig(g):
        """Verdict cache key: slots with identical retrieval queries (left/right arms) share one verdict, no need to ask twice (reported by C1)"""
        qs = (_roles.get(g) or {}).get("queries") or [g]
        return hashlib.sha1(json.dumps(sorted(qs), ensure_ascii=False).encode()).hexdigest()[:8]
    verdicts_p = os.path.join(out_dir, CAND_VERDICTS_FILE)
    verdicts = _jload(verdicts_p, {}) or {}
    _fcache = {}          # incremental cache for faces/slots: fp (fingerprint per piece) + raw (faces below threshold) + slots
    cdir = os.path.join(out_dir, "probe", "cand")
    os.makedirs(cdir, exist_ok=True)

    def wait_file(path):
        empty_since = None
        while True:
            if os.path.isfile(path):
                c = open(path).read().strip()
                if c:
                    os.remove(path)
                    return c
                empty_since = empty_since or time.time()
                if time.time() - empty_since > 5:
                    os.remove(path); empty_since = None
            else:
                empty_since = None
            time.sleep(0.5)

    def _used_in(c):
        """Which slots in the chain this candidate is actually installed in (RULING 09-07: reuse is legal as long as it's reasonable
        and the instruction contains precise location words—don't filter across slots, just mark 'where it's already installed'
        for the agent to make informed decisions).
        Get real assembly history from place_report.json, don't use used—used contains parts skipped in prescreening, will mislabel.
        (stage_serve's scope has no stage_run's reports local variable, direct reference causes NameError and kills serve—W3-88 tested on 2fa3f68;
        the persisted place_report is the same history.)"""
        try:
            _rp = os.path.join(out_dir, "place_report.json")
            _reps = json.load(open(_rp)) if os.path.isfile(_rp) else []
        except Exception:
            _reps = []
        return sorted({f"{r.get('slot','?')}(t{r['turn']})" for r in _reps
                       if r.get("added_oid") == c["oid"] and str(r.get("added_pid")) == str(c["pid"])})

    def cand_hook(turn_no, g, mt, cand, q0, k, rest=None):
        """"Image 2": before placing the candidate part, render it alone in 4 views + the probe image of this slot, emit candidate event,
        wait for agent to write cand_cmd (`fit <bottom|top|side|end|rim|xpos|xneg|ypos|yneg|zpos|zneg>` / `skip <reason>`). Ask each candidate
        only once; verdict is recorded in cand_verdicts.json, redo after veto and serve restart won't ask again."""
        key = f"{cand['oid']}/{cand['pid']}"
        vkey = f"{_sig(g)}:{key}"    # record by "retrieval query signature + candidate": tray and canopy positions separate, left/right arms share
        if vkey in verdicts:
            return tuple(verdicts[vkey])
        glb = os.path.join(cdir, f"turn{turn_no:02d}_{k}_{cand['oid'][:8]}_{cand['pid']}.glb")
        sc = trimesh.Scene(); sc.add_geometry(q0, geom_name="candidate"); sc.export(glb)
        od = os.path.join(img_dir, f"cand_turn{turn_no:02d}_{k}")
        srv.render(glb, od)
        pngs = [os.path.join(od, f"f{v:04d}.png") for v in range(4)]
        # bottom row: host of current state (this round's slot removed), render once per slot per turn; if not exists fall back to probe image from startup
        hd = os.path.join(img_dir, f"host_turn{turn_no:02d}_{g}")
        if rest is not None and not _frames_ok(hd):
            hg = os.path.join(cdir, f"host_turn{turn_no:02d}_{g}.glb")
            hs = trimesh.Scene(); hs.add_geometry(rest, geom_name="host"); hs.export(hg)
            srv.render(hg, hd)
        src = hd if _frames_ok(hd) else os.path.join(img_dir, f"probe_{g}")
        pngs += [p for p in (os.path.join(src, f"f{v:04d}.png") for v in range(4)) if os.path.isfile(p)]
        _ui = _used_in(cand)
        _uitag = f"  [ALREADY INSTALLED ON: {','.join(_ui)}]" if _ui else ""
        png = _grid(pngs, os.path.join(out_dir, "review", f"cand_turn{turn_no:02d}_{k}.png"),
                    title=f"CANDIDATE  turn {turn_no}  slot={g}  slot type={mt}  cand={key[:8]}/{cand['pid']}{_uitag}\n"
                          f"{cand['cap'][:110]}\n"
                          f"top: candidate part alone (4 views)   bottom: current host with this slot's part removed\n"
                          f"answer in cand_cmd:  fit <bottom|top|side|end|rim|xpos|xneg|ypos|yneg|zpos|zneg>   or   skip <reason>")
        emit(turn=turn_no, event="candidate", k=k, slot=g, type=mt,
             cand=f"{key[:8]}/{cand['pid']}", cap=cand["cap"][:80], img=png,
             **({"used_in": _ui} if _ui else {}))
        c = wait_file(cand_p)
        parts = c.split(None, 1)
        if parts and parts[0].lower() == "fit":
            side = (parts[1].strip().split()[0].lower() if len(parts) > 1 else "bottom")
            v = ["fit", side if side in MOUNT_SIDES else "bottom"]
        else:
            v = ["skip", (parts[1].strip() if len(parts) > 1 else "")[:40]]
        verdicts[vkey] = v
        json.dump(verdicts, open(verdicts_p + ".tmp", "w"), indent=1, ensure_ascii=False)
        os.replace(verdicts_p + ".tmp", verdicts_p)
        emit(turn=turn_no, event="cand_verdict", cand=f"{key[:8]}/{cand['pid']}", verdict=v)
        return tuple(v)

    _lazy = {}

    def _retex_hook(turn_no, g, piece_mesh, prompt):
        """Retexture round: render current slot mesh alone → Qwen redraw → TRELLIS.2 bake texture back. Return baked glb."""
        farm = os.environ.get("PXFORM_TEXFARM_DIR")   # if set, use global persistent worker, don't create per-chain
        gpu_t = os.environ.get("PXFORM_TEX_GPU", os.environ.get("PXFORM_GPU"))
        if not farm:
            # dead worker can't stay in cache (after first OOM, fail instantly every time)
            for wk in ("restyle", "tex"):
                if wk in _lazy and _lazy[wk].proc.poll() is not None:
                    _lazy.pop(wk)
            if "restyle" not in _lazy:
                _lazy["restyle"] = RestyleServer(os.path.join(out_dir, "spool_restyle"), gpu=gpu_t)
            if "tex" not in _lazy:
                _lazy["tex"] = TexServer(os.path.join(out_dir, "spool_tex"), gpu=gpu_t)
        rd = os.path.join(out_dir, "probe", "retex")
        os.makedirs(rd, exist_ok=True)
        src_glb = os.path.join(rd, f"turn{turn_no:02d}_{g}_src.glb")
        _t0 = time.time()
        piece_mesh.export(src_glb)
        # directory includes slot: when retrying with different prompt in same turn, part unchanged, directly reuse reference image;
        # if switch slot, falls into different directory, won't feed old slot's image to Qwen
        vd = os.path.join(img_dir, f"retex_turn{turn_no:02d}_{g}_src")
        if not _frames_ok(vd):
            srv.render(src_glb, vd)                   # part alone in 4 views
        _t_render = time.time() - _t0
        ref0 = os.path.join(vd, "f0000.png")
        ref1 = os.path.join(rd, f"turn{turn_no:02d}_{g}_ref.png")
        _rprompt = (f"Change the material to {prompt}. Keep the shape, "
                    f"geometry and camera angle exactly the same. White background.")
        baked = os.path.join(rd, f"turn{turn_no:02d}_{g}_baked.glb")
        if farm:
            _t_restyle = farm_submit(os.path.join(farm, "spool_restyle"),
                                     dict(img=ref0, prompt=_rprompt, out=ref1, seed=0), 600, "redraw")
            _t_tex = farm_submit(os.path.join(farm, "spool_tex"),
                                 dict(mesh=src_glb, ref=ref1, out=baked, seed=0), 900, "texture")
        else:
            _t1 = time.time()
            _lazy["restyle"].restyle(ref0, _rprompt, ref1)
            _t_restyle = time.time() - _t1
            _t1 = time.time()
            _lazy["tex"].retexture(src_glb, ref1, baked)
            _t_tex = time.time() - _t1
        emit(turn=turn_no, event="retex_done", slot=g, prompt=prompt[:60], ref=ref1,
             t_render=round(_t_render, 1), t_restyle=round(_t_restyle, 1), t_tex=round(_t_tex, 1))
        return baked

    def cand_batch(turn_no, g, mt, items, rest):
        """Batch judgment: one image with all candidates (each 2 views, numbered) + current host 4 views, emit candidates event,
        wait for agent to answer in cand_cmd line by line `<#> fit <side>` / `<#> skip <reason>`.
        Unanswered candidates recorded as skip. Candidates with verdicts don't appear again."""
        todo = [(c, q0) for c, q0 in items if f"{_sig(g)}:{c['oid']}/{c['pid']}" not in verdicts]
        if not todo:
            return
        _t0 = time.time()
        pngs, labels, listing = [], [], []
        for i, (c, q0) in enumerate(todo, 1):
            glb = os.path.join(cdir, f"turn{turn_no:02d}_{c['oid'][:8]}_{c['pid']}.glb")
            sc = trimesh.Scene(); sc.add_geometry(q0, geom_name="candidate"); sc.export(glb)
            od = os.path.join(img_dir, f"cand_turn{turn_no:02d}_{c['oid'][:8]}_{c['pid']}")
            if not _frames_ok(od):
                srv.render(glb, od)
            pngs += [os.path.join(od, "f0000.png"), os.path.join(od, "f0002.png")]
            labels += [f"#{i}", f"#{i}"]
            _ui = _used_in(c)
            listing.append(dict(i=i, cand=f"{c['oid'][:8]}/{c['pid']}", cap=c["cap"][:80],
                                **({"used_in": _ui} if _ui else {})))
        hd = os.path.join(img_dir, f"host_turn{turn_no:02d}_{g}")
        if rest is not None and not _frames_ok(hd):
            hg = os.path.join(cdir, f"host_turn{turn_no:02d}_{g}.glb")
            hs = trimesh.Scene(); hs.add_geometry(rest, geom_name="host"); hs.export(hg)
            srv.render(hg, hd)
        # if host render fails, don't splice non-existent paths into grid (_grid_labeled would FileNotFoundError
        # and kill serve, W3-51 reported; cand path side already has isfile filtering, align standards)
        _host_pngs = [p for p in (os.path.join(hd, f"f{v:04d}.png") for v in range(4))
                      if os.path.isfile(p)]
        pngs += _host_pngs
        labels += ["host"] * len(_host_pngs)
        caps = "\n".join(f"#{e['i']} {e['cand']}  {e['cap'][:95]}"
                         + (f"  [ALREADY INSTALLED ON: {','.join(e['used_in'])}]" if e.get("used_in") else "")
                         for e in listing)
        png = _grid_labeled(pngs, labels,
                            os.path.join(out_dir, "review", f"cands_turn{turn_no:02d}_{g}.png"),
                            title=f"CANDIDATES  turn {turn_no}  slot={g}  slot type={mt}   "
                                  f"(each candidate: 2 views; last row: current host without this slot)\n"
                                  f"{caps}\nanswer in cand_cmd, one line per candidate:  <#> fit <bottom|top|side|end|rim|xpos|xneg|ypos|yneg|zpos|zneg>   or   <#> skip <reason>")
        emit(turn=turn_no, event="candidates", slot=g, type=mt, n=len(listing),
             cands=listing, img=png, dur=round(time.time() - _t0, 1))
        txt = wait_file(cand_p)
        answers = {}
        for line in txt.splitlines():
            parts = line.strip().lstrip("#").split(None, 2)
            if len(parts) >= 2 and parts[0].isdigit():
                idx = int(parts[0]); verb = parts[1].lower()
                if verb == "fit":
                    side = (parts[2].strip().split()[0].lower() if len(parts) > 2 else "bottom")
                    answers[idx] = ["fit", side if side in MOUNT_SIDES else "bottom"]
                else:
                    answers[idx] = ["skip", (parts[2].strip() if len(parts) > 2 else "")[:40]]
        for e in listing:
            c = todo[e["i"] - 1][0]
            verdicts[f"{_sig(g)}:{c['oid']}/{c['pid']}"] = answers.get(e["i"], ["skip", "unanswered"])
        json.dump(verdicts, open(verdicts_p + ".tmp", "w"), indent=1, ensure_ascii=False)
        os.replace(verdicts_p + ".tmp", verdicts_p)
        emit(turn=turn_no, event="cand_verdicts", slot=g,
             verdicts={e["cand"]: answers.get(e["i"], ["skip", "unanswered"]) for e in listing})
    cand_hook.batch = cand_batch

    def _pieces_at(k):
        """Assembly state before round k starts (checkpoint from previous round, or original object)."""
        ck = os.path.join(out_dir, "ckpt", f"turn{k - 1:02d}.pkl")
        if k > 1 and os.path.isfile(ck):
            return _ckpt_load(out_dir, k - 1)["pieces"]
        return _init_host(oid, out_dir)["pieces"]

    def ask_op(k):
        """faces event: render available mounting faces on current state (magenta thin plate + green/blue sphere markers +u/+v), list removable slots,
        wait for agent to write op_cmd: `add <face#> "<description>" size=<ratio> at=<u,v>` / `remove <slot>` / `replace` / `stop`."""
        import shlex
        _t0 = time.time()
        pieces = _pieces_at(k)
        ini = _init_host(oid, out_dir)
        diag = ini["diag"]
        _raw, _fnote = _faces_update(_fcache, pieces, diag)   # hit=geometry unchanged (retexture/re-ask) all skip
        faces = _rank_faces(_raw)
        if "add" in ops:
            _gf = ground_face(pieces, ini["diag"])
            if _gf is not None:
                faces = faces + [_gf]           # id fixed to 0, doesn't take up one of host's 8 slots
        pngs, labels = [], []
        # mounting face image only for add: chains that can't add (pure texture, pure delete/replace) skip face-by-face rendering
        # (W3-1 reported pure texture version; R format remove/replace chains faces took 2/3 time then generalized to judge by ops)
        for f in (faces if "add" in ops else []):
            sc = trimesh.Scene()
            for kk, m in pieces.items():
                sc.add_geometry(m, geom_name=f"slot_{kk}")
            quad, gdot, bdot, sdots = face_marker(f, diag)
            sc.add_geometry(quad, geom_name="face"); sc.add_geometry(gdot, geom_name="u"); sc.add_geometry(bdot, geom_name="v")
            for si, sd in enumerate(sdots, 1):
                sc.add_geometry(sd, geom_name=f"spot{si}")
            glb = os.path.join(cdir, f"faces_turn{k:02d}_{f['id']}.glb")
            sc.export(glb)
            od = os.path.join(img_dir, f"faces_turn{k:02d}_{f['id']}")
            if not _frames_ok(od):        # don't re-render when re-asking op after veto in same turn (geometry unchanged, face id stable within turn)
                srv.render(glb, od)
            pngs += [os.path.join(od, "f0000.png"), os.path.join(od, "f0002.png")]
            labels += [f"#{f['id']}", f"#{f['id']}"]
        _slots_cached = (_fnote == "hit" and _fcache.get("slots") is not None)
        slots = list(_fcache["slots"]) if _slots_cached else []
        for g in (() if _slots_cached else pieces):
            if g in ini["anchors"]:
                continue
            # if we delete it, who becomes orphaned (wooden bucket stacked on wooden box, delete box bucket floats)—D1 reported
            trial = {kk: m for kk, m in pieces.items() if kk != g}
            orphan = []
            if len(trial) > 1:
                keys, adj, _ = CG.build(trial, diag)
                from scipy.sparse.csgraph import connected_components as _cc
                from scipy.sparse import coo_matrix as _coo
                lab = _cc(_coo(adj), directed=False)[1]
                root = {lab[keys.index(a)] for a in ini["anchors"] if a in keys}
                orphan = [kk for kk, l in zip(keys, lab) if l not in root]
            slots.append(g + (f"(deleting would leave {'/'.join(orphan)} orphaned)" if orphan else ""))
        if not _slots_cached:
            _fcache["slots"] = slots
        lines = []
        for f in faces:
            sp = "  ".join(f"spot{i}(clear {s['clear'] / diag:.2f})" for i, s in enumerate(f.get("spots", []), 1))
            lines.append(f"#{f['id']} {f['kind']} of {f['piece']}  face {f['L1'] / diag:.2f} x {f['L2'] / diag:.2f} diag, "
                         f"free {f['free_L1'] / diag:.2f} x {f['free_L2'] / diag:.2f}   landing spots (orange, brightest = spot1): {sp or 'none'}")
        png = None
        if pngs:
            png = _grid_labeled(pngs, labels, os.path.join(out_dir, "review", f"faces_turn{k:02d}.png"),
                                title="FACES  turn %d   free mounting faces (magenta), green dot = +u (long side), blue dot = +v (short side)\n" % k
                                      + "\n".join(lines) + "\nremovable slots: " + ", ".join(slots) + "\n"
                                      "answer in op_cmd:  add <#> \"<what to add>\" spot=<n> size=<0..1 of the spot clearance>   |   retexture <slot> \"<new material>\"   |   remove <slot>   |   abort   |   replace   |   stop")
        emit(turn=k, event="faces", img=png,
             faces=[dict(id=f["id"], piece=f["piece"], kind=f["kind"],
                         face=[round(f["L1"] / diag, 3), round(f["L2"] / diag, 3)],
                         free=[round(f["free_L1"] / diag, 3), round(f["free_L2"] / diag, 3)],
                         e1=[round(float(x), 2) for x in f["e1"]], e2=[round(float(x), 2) for x in f["e2"]],
                         spots=[dict(i=i, uv=s["uv"], clear=round(s["clear"] / diag, 3))
                                for i, s in enumerate(f.get("spots", []), 1)]) for f in faces],
             slots=slots, cache=_fnote, dur=round(time.time() - _t0, 1))
        txt = wait_file(os.path.join(out_dir, "op_cmd")).strip()
        try:
            parts = shlex.split(txt)
        except ValueError:
            parts = txt.split()
        if not parts:
            emit(turn=k, event="bad_op", cmd=txt)
            return ask_op(k)
        kind = parts[0].lower()
        if kind == "add":
            # if parse fails never silently degrade, also never let ValueError kill serve (review #3)
            try:
                if len(parts) < 3:
                    raise ValueError("add needs <face#> and description")
                fid = int(parts[1].lstrip("#"))
                face = next((f for f in faces if f["id"] == fid), None)
                if face is None:
                    raise ValueError(f"no face #{fid}")
                opd = dict(kind="add", face=dict(face), query=parts[2], size=0.5, at=[0.0, 0.0])
                for extra in parts[3:]:
                    if extra.startswith("size="):
                        opd["size"] = float(extra[5:])
                    elif extra.startswith("at="):
                        u, v = extra[3:].split(",")
                        opd["at"] = [float(u), float(v)]
                    elif extra.startswith("spot="):
                        si = int(extra[5:]) - 1
                        sp = (face.get("spots") or [])
                        if 0 <= si < len(sp):
                            opd["spot"] = si + 1
                            opd["at"] = list(sp[si]["uv"])
                            opd["clear"] = float(sp[si]["clear"])
            except (ValueError, IndexError):
                emit(turn=k, event="bad_op", cmd=txt)
                return ask_op(k)
            emit(turn=k, event="op", op="add", face=fid, query=parts[2], size=opd["size"], at=opd["at"])
            return opd
        if kind == "retexture" and len(parts) >= 3:
            emit(turn=k, event="op", op="retexture", slot=parts[1], prompt=parts[2][:80])
            return dict(kind="retexture", slot=parts[1], prompt=parts[2], hook=_retex_hook)
        if kind == "remove" and len(parts) >= 2:
            emit(turn=k, event="op", op="remove", slot=parts[1])
            return dict(kind="remove", slot=parts[1])
        if kind in ("stop", "abort"):
            return dict(kind="stop" if kind == "stop" else "abort")
        if kind == "replace":
            emit(turn=k, event="op", op="replace")
            return dict(kind="replace")
        # unparseable commands never silently downgrade to replace (review #3: M format chains have produced replace rounds)
        emit(turn=k, event="bad_op", cmd=txt)
        return ask_op(k)

    try:
        import glob as _g
        if not os.path.isfile(os.path.join(out_dir, MOUNT_TYPES_FILE)):
            stage_probe(oid, out_dir, srv, img_dir, emit)      # read image 1: slot mounting face types
        done = []
        for f in _g.glob(os.path.join(out_dir, "ckpt", "turn??.pkl")):
            try:                     # truncated, foreign or modified ckpt: treat as non-existent and move it aside, never unpickle it (review #4)
                _ckpt_read(f)
                done.append(int(os.path.basename(f)[4:6]))
            except Exception:
                os.replace(f, f + ".corrupt")
        k = max(done) + 1 if done else 1        # serve crashed/was stopped, restart continues from here
        if k > 1:
            emit(event="resume_serve", start_turn=k)
        pinned, rep = None, None
        autov = {}          # count of auto veto per turn (visibility change gate, see change_meter)
        op_cur = None
        while k <= n_turns:
            yf = _jload(yawf_p)
            t0 = time.time()
            if ("add" in ops or "remove" in ops or "retexture" in ops) and op_cur is None:
                op_cur = ask_op(k)              # faces event: agent decides this turn's add / remove / replace
                if op_cur is not None and op_cur.get("kind") == "stop":
                    emit(event="stopped_by_reviewer"); break
                if op_cur is not None and op_cur.get("kind") == "abort":
                    op_cur = None; continue          # abandon this turn's op, re-ask faces
                if op_cur is not None and op_cur.get("kind") == "replace":
                    op_cur = None
            try:
                stage_run(oid, out_dir, n_turns=n_turns, seed=seed, pool=pool,
                          veto=(veto_p if os.path.isfile(veto_p) else None),
                          yaw_fix=yf, pin=pinned,
                          resume_from=(k if k > 1 else None), stop_after=k,
                          cand_hook=cand_hook, op=op_cur)
                ran = True
            except RuntimeError as e:
                # yaw pin redo failed: original candidate can't fit after rotation. **don't silently swap**,
                # stop and wait for reviewer to fix instruction (yaw rotate / veto swap).
                if op_cur is not None:
                    emit(turn=k, event="op_failed", op=op_cur.get("kind"), err=str(e)[:200])
                    op_cur = None            # re-ask what to do this turn
                    continue
                emit(turn=k, event="yaw_pin_failed", err=str(e)[:160])
                ran = False
            if ran:
                glb = os.path.join(out_dir, f"turn{k:02d}.glb")
                if not os.path.isfile(glb):
                    emit(turn=k, event="assemble_failed")
                    break
                t_asm = time.time() - t0
                if k == 1 and not os.path.isdir(os.path.join(img_dir, "turn00")):
                    srv.render(os.path.join(out_dir, "turn00.glb"),
                               os.path.join(img_dir, "turn00"))
                t_r = srv.render(glb, os.path.join(img_dir, f"turn{k:02d}"))
                # render previous turn state at this turn's camera, gate compares two shots at same camera
                srv.render(os.path.join(out_dir, f"turn{k-1:02d}.glb"),
                           os.path.join(img_dir, f"turn{k:02d}_prev"), frame=sphere_frame(glb))
                rep = json.load(open(os.path.join(out_dir, "place_report.json")))[-1]
                # visibility change gate (the screen must show obvious
                # human-recognizable change). Swaps below hard gate auto-veto, no work for reviewer;
                # fail gate 5 times in a row means slot may be invisible, return to reviewer to decide.
                from change_meter import change_frac, VIS_TAU
                metric, per = change_frac(out_dir, k, img_dir=img_dir)
                if metric is not None:
                    vis_p = os.path.join(out_dir, "vis.json")
                    vis = _jload(vis_p, {}) or {}
                    vis[str(k)] = dict(metric=round(metric, 4), per=per)
                    _jdump(vis, vis_p, indent=1)
                # add/remove small parts inherently low change (3%–10%), hard gate meaningless: delete has no
                # candidates to swap (writing ["",""] to veto.json is no-op, re-run identical), add should be
                # reviewer's decision "is it visible", system shouldn't try 5 times (D1 measured 12 mins idle spin)
                op_kind = rep.get("op")
                if metric is not None and metric < VIS_TAU and op_kind in ("add", "remove", "retexture"):
                    emit(turn=k, event="low_vis_op", op=op_kind, metric=round(metric, 4),
                         note="add/remove/retexture skip auto-veto, you judge if it's visible")
                elif metric is not None and metric < VIS_TAU and autov.get(k, 0) < 5:
                    autov[k] = autov.get(k, 0) + 1
                    vt = _jload(veto_p, {}) or {}
                    vt.setdefault(rep["slot"], []).append(
                        [rep["added_oid"], rep["added_pid"]])
                    _jdump(vt, veto_p, indent=1)
                    if os.path.isfile(yawf_p):
                        yf0 = _jload(yawf_p, {}) or {}
                        if yf0.pop(str(k), None) is not None:
                            _jdump(yf0, yawf_p, indent=1)
                    pinned = None
                    emit(turn=k, event="auto_veto_invisible",
                         metric=round(metric, 4), cand=rep["src"],
                         n=autov[k], slot=rep["slot"])
                    continue
                # close-up: same camera renders "before swap" and "after swap" each 4 views, judge pose from it—
                # on panorama new part often only ~100 pixels, wrong pose totally invisible.
                if rep.get("focus"):
                    srv.render(glb, os.path.join(img_dir, f"turn{k:02d}_close"),
                               focus=rep["focus"])
                    srv.render(os.path.join(out_dir, f"turn{k-1:02d}.glb"),
                               os.path.join(img_dir, f"turn{k:02d}_close_prev"),
                               focus=rep["focus"])
                emit(turn=k, event="ready", asm=round(t_asm, 1), render=round(t_r, 1),
                     connected=rep.get("connected", True),
                     img=f"{img_dir}/turn{k:02d}/f0000.png",
                     close=f"{img_dir}/turn{k:02d}_close/f0000.png",
                     vis=(round(metric, 4) if metric is not None else None),
                     rejected=rep.get("rejected", []),
                     low_vis_stuck=(autov.get(k, 0) >= 5) or None,
                     instruction=rep["instruction"], added=rep["added"][:60])
            while True:
                c = wait_cmd()
                if c in ("ok", "veto", "unveto", "stop") or c.startswith("yaw"):
                    break
                # garbled/unknown commands don't re-run this turn (review #4: once deleted approved round and re-placed), wait in place
                emit(event="unknown_cmd", cmd=c)
            if c == "ok":
                if not ran:
                    emit(turn=k, event="cannot_ok", note="last assembly failed, do yaw/veto first")
                    continue
                k += 1
                pinned = None
                op_cur = None
            elif c == "veto":
                if not ran:
                    # failed assembly turn's rep is still last turn's, using directly kills wrong part (review #1).
                    # after yaw pin fails, veto's legal intent is "abandon pinned candidate": get from pinned.
                    pt = (pinned or {}).get(str(k))
                    if not pt:
                        emit(turn=k, event="cannot_veto",
                             note="last assembly failed and no pinned, rep is last turn's; resend op or wait for re-run")
                        continue
                    vt = _jload(veto_p, {}) or {}
                    _g0 = pt[2]
                    _q0 = tuple((_roles.get(_g0) or {}).get("queries") or [_g0])
                    for _g in {_g0, *[gg for gg in _roles
                                      if tuple((_roles.get(gg) or {}).get("queries") or [gg]) == _q0]}:
                        vt.setdefault(_g, []).append([pt[0], pt[1]])
                    _jdump(vt, veto_p, indent=1)
                    yf = _jload(yawf_p, {}) or {}
                    if yf.pop(str(k), None) is not None:
                        _jdump(yf, yawf_p, indent=1)
                    if "add" in ops or "remove" in ops or "retexture" in ops:
                        op_cur = None
                    pinned = None
                    emit(turn=k, event="veto", cand=f"{pt[0][:8]}/{pt[1]}")
                    continue
                vt = _jload(veto_p, {}) or {}
                if rep.get("op") == "add":
                    vt.setdefault(f"add:{rep.get('query', '')}", []).append([rep["added_oid"], rep["added_pid"]])
                elif rep.get("op") in ("remove", "retexture"):
                    pass                       # delete/texture rejected: don't record veto table, re-ask op next turn (texture should swap with one prompt)
                else:
                    # paired left/right slots share retrieval query: killed part written to veto table of **all slots with same signature**,
                    # otherwise same bad part swaps to opposite side and comes again (W3-8 / W3-19 each reported once, wasted 4 rounds)
                    _g0 = rep["slot"]
                    _q0 = tuple(( _roles.get(_g0) or {}).get("queries") or [_g0])
                    for _g in {_g0, *[gg for gg in _roles
                                      if tuple((_roles.get(gg) or {}).get("queries") or [gg]) == _q0]}:
                        vt.setdefault(_g, []).append([rep["added_oid"], rep["added_pid"]])
                if rep.get("op") in ("add", "remove", "retexture") or                         ("add" in ops or "remove" in ops or "retexture" in ops):
                    op_cur = None              # re-ask agent what to do this turn
                _jdump(vt, veto_p, indent=1)
                # after veto swaps, this turn's leftover yaw angle aimed at old candidate, meaningless for new part
                if os.path.isfile(yawf_p):
                    yf = _jload(yawf_p, {}) or {}
                    if yf.pop(str(k), None) is not None:
                        _jdump(yf, yawf_p, indent=1)
                pinned = None
                emit(turn=k, event="veto", cand=rep["src"])
            elif c.startswith("yaw"):
                if not ran:
                    emit(turn=k, event="cannot_yaw", note="last assembly failed, rep is last turn's; veto swap first or wait for re-run")
                    continue
                try:
                    deg = float(c.split()[1])
                except (IndexError, ValueError):
                    emit(turn=k, event="unknown_cmd", cmd=c)
                    continue
                yf = _jload(yawf_p, {}) or {}
                yf[str(k)] = deg
                _jdump(yf, yawf_p, indent=1)
                # pin current candidate: yaw is "fine-tune same candidate", can't fit after turn must fail loudly, not swap
                pinned = {str(k): [rep["added_oid"], rep["added_pid"], rep["slot"]]}
                emit(turn=k, event="yaw", deg=deg, pin=rep["src"])
            elif c == "unveto":
                if not ran:
                    emit(turn=k, event="cannot_unveto", note="last assembly failed, rep is last turn's")
                    continue
                # undo this slot's latest veto (C reported: "redo 4 times still unsatisfied pick best"
                # and veto irreversible conflict). Deterministic guarantee redo returns to that undone candidate.
                vt = _jload(veto_p, {}) or {}
                lst = vt.get(rep["slot"]) or []
                if lst:
                    popped = lst.pop()
                    _jdump(vt, veto_p, indent=1)
                    emit(turn=k, event="unveto", cand=f"{popped[0][:8]}/{popped[1]}")
                else:
                    emit(turn=k, event="unveto_nothing")
            elif c == "stop":
                emit(event="stopped_by_reviewer"); break
        else:
            emit(event="chain_done", turns=n_turns)
    finally:
        srv.stop()
        for w in _lazy.values():
            try:
                w.stop()
            except Exception:
                pass



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["run", "step", "serve", "check", "render"])
    ap.add_argument("--turn", type=int, default=None, help="step: assemble and render which round")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oid", default=None)
    ap.add_argument("--out-dir", default=None)   # not needed for chain stage
    ap.add_argument("--img-dir", default=None)
    ap.add_argument("--n-turns", type=int, default=12)   # standard chain length
    ap.add_argument("--from-turn", type=int, default=None,
                    help="render: only render state from round K onward (pairs with --resume-from)")
    ap.add_argument("--workers", type=int, default=None,
                    help="render: number of parallel blender processes, default=local free GPUs")
    ap.add_argument("--resume-from", type=int, default=None,
                    help="resume from round K (rounds 1..K-1 use checkpoints, no recompute)")
    ap.add_argument("--yaw-fix", default=None,
                    help="orientation correction json {round: additional angle}, use after visual review")
    ap.add_argument("--pool", default=None,
                    help="fixed candidate pool json (for LLM nomination control experiment), format {slot:[[oid,pid],..]})")
    ap.add_argument("--veto", default=None,
                    help="reviewed rejected candidates (generated by review_chain.py --veto-from)")
    ap.add_argument("--ops", default="replace",
                    help="serve-allowed edit operations, comma-separated: replace,add,remove. When including add/remove, emit faces event each round to ask agent")
    a = ap.parse_args()

    if a.stage == "serve":
        stage_serve(a.oid, a.out_dir, a.n_turns, a.seed, pool=a.pool,
                    img_dir=a.img_dir, ops=tuple(x.strip() for x in a.ops.split(",") if x.strip()))
        return
    if a.stage == "step":
        # per-round review mode: assemble round K -> render only round K -> print comparison image path.
        # if pass, step K+1; if fail, re-run step K with --veto / --yaw-fix (downstream zero waste).
        k = a.turn
        yf = json.load(open(a.yaw_fix)) if a.yaw_fix else None
        stage_run(a.oid, a.out_dir, n_turns=max(a.n_turns, k), seed=a.seed,
                  veto=a.veto, pool=a.pool,
                  yaw_fix=yf, resume_from=(k if k > 1 else None), stop_after=k)
        if not os.path.isfile(os.path.join(a.out_dir, f"turn{k:02d}.glb")):
            print(f"  !! round {k} failed to assemble (all candidates rejected or slot removed), see log above",
                  flush=True)
            sys.exit(3)
        img = a.img_dir or (a.out_dir + "/img")
        # round 1 renders turn00 (original state) together, review needs "before swap" for comparison
        stage_render(a.out_dir, img, from_turn=(0 if k == 1 else k), workers=1)
        print(f"  view image: {img}/turn{k:02d}/f0000.png (4 views total)", flush=True)
        return
    if a.stage == "run":
        yf = json.load(open(a.yaw_fix)) if a.yaw_fix else None
        stage_run(a.oid, a.out_dir, a.n_turns, a.seed, veto=a.veto, pool=a.pool,
                  yaw_fix=yf, resume_from=a.resume_from)
        return
    if a.stage == "check":
        stage_check(a.oid, a.out_dir)
        return
    if a.stage == "render":
        # render only recognizes glbs in out_dir, doesn't need chain json. `--stage run` outputs glb directly,
        # doesn't write chain file, so this branch must split before load_chain.
        stage_render(a.out_dir, a.img_dir or (a.out_dir + "/img"),
                     from_turn=a.from_turn, workers=a.workers)
        return


if __name__ == "__main__":
    main()
