#!/usr/bin/env python3
"""Standalone chain assembler for the chains1k recipe release.

Rebuilds every turn GLB of a chain from
  (a) the PartVerseXL textured part GLBs you downloaded yourself, and
  (b) this repo's JSON recipes (+ the small retex/ and hy3d_parts/ artifacts),
then verifies the result against checks.json (vertex count, |v| sum, texture
sha1 per node). Only deps: numpy, trimesh, Pillow.

Usage:
  python assemble.py --pv-root /path/to/textured_part_glbs \
                     --chain   <repo>/A/1a3267df_s0 \
                     [--hy3d-root <repo>/hy3d_parts] [--out <dir>] [--no-verify]
"""
import argparse, hashlib, json, os, sys
import numpy as np, trimesh
from PIL import Image
from manual_pose import posed


def tex_sha(mat, attr):
    im = getattr(mat, attr, None)
    if im is None:
        return None
    return hashlib.sha1(np.asarray(im.convert('RGBA')).tobytes()).hexdigest()[:12]


def sig(g):
    m = getattr(g.visual, 'material', None)
    return dict(nv=int(len(g.vertices)),
                vs=round(float(np.abs(np.asarray(g.vertices)).sum()), 4),
                bc=tex_sha(m, 'baseColorTexture') if m is not None else None,
                mr=tex_sha(m, 'metallicRoughnessTexture') if m is not None else None)


def load_part(pv_root, hy3d_root, ds, oid, pid):
    root = hy3d_root if ds == 'hy3d' else pv_root
    p = os.path.join(root, oid, f'{pid}.glb')
    return trimesh.load(p, process=False, force='mesh')


def build_turn00(man, pv_root, chain_dir):
    nodes = {}
    for name, e in man['nodes'].items():
        if e.get('override_glb'):
            sc = trimesh.load(os.path.join(chain_dir, e['override_glb']),
                              process=False, file_type='glb')
            nodes[name] = sc.geometry.get(name) or next(iter(sc.geometry.values()))
            if e['override_glb'].replace('\\', '/').startswith('baseline/'):
                # Explicit baseline production validates corner normals before
                # exporting each state. Materialize the identical normals here
                # too, including when the source GLB omits a NORMAL accessor,
                # so both paths serialize the same float32 rendering payload.
                _ = nodes[name].vertex_normals
            continue
        ms = [trimesh.load(os.path.join(pv_root, man['host_oid'], f'{p}.glb'),
                           process=False, force='mesh') for p in e['pids']]
        nodes[name] = trimesh.util.concatenate(ms) if len(ms) > 1 else ms[0]
    return nodes


def apply_T(m, T):
    T = np.asarray(T, float)
    out = trimesh.Trimesh(vertices=m.vertices @ T[:3, :3].T + T[:3, 3],
                          faces=m.faces, process=False)
    out.visual = m.visual
    return out


def source_normals_for_legacy_mesh(pv_root, hy3d_root, r, legacy):
    """Read source shading before Scene.dump/copy can discard its normal cache.

    The returned rows are proven to match the existing source flattening's
    vertex/face order. This helper never concatenates or changes materials/UVs.
    """
    root = hy3d_root if r.get('ds', 'pv') == 'hy3d' else pv_root
    path = os.path.join(root, r['added_oid'], f"{r['added_pid']}.glb")
    source = trimesh.load(path, process=False, force='scene')
    vertices, faces, normals = [], [], []
    offset = 0
    for node_name in source.graph.nodes_geometry:
        graph_T, geometry_name = source.graph[node_name]
        original = source.geometry[geometry_name]
        if not isinstance(original, trimesh.Trimesh):
            raise ValueError('Source-normal preservation requires mesh instances')
        # Read the original loaded mesh, never a copy: Trimesh.copy does not
        # retain the authored NORMAL cache in the project runtime.
        authored = np.array(original.vertex_normals, dtype=float, copy=True)
        graph_T = np.asarray(graph_T, dtype=float)
        if graph_T.shape != (4, 4) or not np.isfinite(graph_T).all():
            raise ValueError('Invalid source scene placement')
        if np.linalg.det(graph_T[:3, :3]) == 0:
            raise ValueError('Singular source scene placement')
        n = authored @ np.linalg.inv(graph_T[:3, :3])
        lengths = np.linalg.norm(n, axis=1, keepdims=True)
        if not np.isfinite(n).all() or (lengths <= 1e-30).any():
            raise ValueError('Source contains invalid shading normals')
        n /= lengths
        # Use exactly Scene.dump's geometry operation, including reflected
        # winding, solely to prove the row order of the legacy loaded mesh.
        placed = original.copy()
        placed.apply_transform(graph_T)
        vertices.append(np.asarray(placed.vertices))
        faces.append(np.asarray(placed.faces) + offset)
        normals.append(n)
        offset += len(placed.vertices)
    if not vertices or not np.array_equal(np.vstack(vertices), legacy.vertices):
        raise ValueError('Source-normal vertex order differs from legacy mesh')
    if not np.array_equal(np.vstack(faces), legacy.faces):
        raise ValueError('Source-normal face order differs from legacy mesh')
    return np.vstack(normals)


def build_piece(r, pv_root, hy3d_root, chain_dir=None):
    if r.get('source_instances') is not None:
        if any(r.get(k) is not None for k in ('asset_copy', 'reinstall_snapshot', 'component_transforms', 'added_pids')):
            raise ValueError('source_instances cannot combine with another source construction')
        from component_piece import build_source_instances
        return build_source_instances(r, pv_root, hy3d_root, chain_dir=chain_dir)
    # A derived asset remains distinct from its original PV/Hy3D provenance.
    # It is stored in source coordinates; the recipe retains its placement.
    if r.get('asset_copy'):
        if chain_dir is None:
            raise ValueError('asset_copy requires its owning recipe directory')
        record = r['asset_copy']
        base = os.path.realpath(chain_dir)
        relative = record['path']
        path = os.path.realpath(os.path.join(base, relative))
        if os.path.isabs(relative) or os.path.commonpath([base, path]) != base:
            raise ValueError('asset_copy must remain inside its owning recipe')
        with open(path, 'rb') as handle:
            if hashlib.sha256(handle.read()).hexdigest() != record['sha256']:
                raise ValueError('asset_copy SHA-256 mismatch')
        mesh = trimesh.load(path, process=False, force='mesh')
        return apply_T(mesh, r.get('transform') or r['transform_affine'])
    preserve = r.get('preserve_source_normals', False)
    if not isinstance(preserve, bool):
        raise ValueError('preserve_source_normals must be a boolean')
    snapshot = r.get('reinstall_snapshot')
    if snapshot:
        if chain_dir is None:
            raise ValueError('reinstall_snapshot requires its owning recipe directory')
        relative = snapshot['path']
        base = os.path.realpath(chain_dir)
        path = os.path.realpath(os.path.join(base, relative))
        if os.path.isabs(relative) or os.path.commonpath([base, path]) != base:
            raise ValueError('reinstall_snapshot must be inside its owning recipe')
        with open(path, 'rb') as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if digest != snapshot['sha256']:
            raise ValueError('reinstall_snapshot SHA-256 mismatch')
        if int(snapshot['source_turn']) < 0 or not snapshot.get('source_version'):
            raise ValueError('reinstall_snapshot requires its source version and turn')
        scene = trimesh.load(path, process=False, force='scene')
        source_node = snapshot['node']
        if source_node not in scene.geometry:
            raise ValueError('reinstall_snapshot node missing')
        m = scene.geometry[source_node]
        for graph_node in scene.graph.nodes_geometry:
            placement, geometry_name = scene.graph[graph_node]
            if geometry_name == source_node and not np.array_equal(placement, np.eye(4)):
                raise ValueError('reinstall_snapshot must contain world-space geometry')
        T = np.asarray(r['transform'], dtype=float)
        if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
            raise ValueError('Invalid reinstall snapshot placement transform')
        if np.array_equal(T, np.eye(4)):
            if preserve:
                # The saved leaf carries the authored normals of its earlier
                # installation. Copying a mesh alone discards that cache.
                normals = np.array(m.vertex_normals, copy=True)
                out = m.copy()
                out.vertex_normals = normals
                return out
            return m.copy()
        # Preserve the saved shading normals as well as every UV/PBR channel.
        normals = m.vertex_normals @ np.linalg.inv(T[:3, :3])
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(lengths, 1e-30)
        faces = m.faces[:, ::-1] if np.linalg.det(T[:3, :3]) < 0 else m.faces
        out = trimesh.Trimesh(vertices=m.vertices @ T[:3, :3].T + T[:3, 3],
                             faces=faces, vertex_normals=normals, process=False)
        out.visual = m.visual.copy()
        return out
    if r.get('component_transforms') is not None:
        from component_piece import build_component_piece
        return build_component_piece(r, pv_root, hy3d_root)
    pids = r.get('added_pids')
    if pids is not None:
        # A complete logical leaf may span several source parts sharing one
        # exact material. Never invoke texture packing or merge unlike PBRs.
        from node_payload import material_payload, _feed
        if not isinstance(pids, list) or len(pids) < 2 or len(set(map(str, pids))) != len(pids):
            raise ValueError('added_pids requires at least two distinct source parts')
        parts = []
        root = hy3d_root if r.get('ds', 'pv') == 'hy3d' else pv_root
        for pid in pids:
            path = os.path.join(root, r['added_oid'], f'{pid}.glb')
            source = trimesh.load(path, process=False, force='scene')
            instances = list(source.graph.nodes_geometry)
            if len(instances) != 1 or len(source.geometry) != 1:
                raise ValueError('Each added_pids source must contain exactly one mesh instance')
            graph_T, geometry_name = source.graph[instances[0]]
            part = source.geometry[geometry_name].copy()
            # apply_transform handles source graph normals and reflected face
            # winding. The source UV arrays and material payload stay intact.
            if not np.array_equal(graph_T, np.eye(4)):
                _ = part.vertex_normals
                part.apply_transform(graph_T)
            parts.append(part)
        def material_digest(mesh):
            if mesh.visual.kind != 'texture' or mesh.visual.uv is None:
                raise ValueError('added_pids requires texture visuals with explicit UVs')
            if getattr(mesh.visual, 'face_materials', None) is not None:
                raise ValueError('added_pids does not flatten per-face materials')
            digest = hashlib.sha256()
            _feed(digest, material_payload(mesh.visual.material))
            return digest.digest()
        digests = [material_digest(part) for part in parts]
        if any(digest != digests[0] for digest in digests[1:]):
            raise ValueError('added_pids parts must share the exact complete material payload; different materials require a scene bundle')
        offsets = np.cumsum([0] + [len(part.vertices) for part in parts[:-1]])
        vertices = np.vstack([part.vertices for part in parts])
        normals = np.vstack([part.vertex_normals for part in parts])
        faces = np.vstack([part.faces + offset for part, offset in zip(parts, offsets)])
        uv = np.vstack([part.visual.uv for part in parts])
        T = np.asarray(r.get('transform') or r.get('transform_affine'), dtype=float)
        if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
            raise ValueError('Invalid added_pids placement transform')
        normals = normals @ np.linalg.inv(T[:3, :3])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
        if np.linalg.det(T[:3, :3]) < 0:
            faces = faces[:, ::-1]
        out = trimesh.Trimesh(vertices=vertices @ T[:3, :3].T + T[:3, 3], faces=faces,
                              vertex_normals=normals, process=False)
        out.visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=parts[0].visual.material.copy())
    else:
        m = load_part(pv_root, hy3d_root, r.get('ds', 'pv'), r['added_oid'], str(r['added_pid']))
        placement = r.get('transform') or r.get('transform_affine')
        if preserve:
            # Explicit recipe opt-in: retain authored source shading through
            # the affine placement. Legacy recipes keep their original path.
            T = np.asarray(placement, dtype=float)
            if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
                raise ValueError('Invalid source-normal placement transform')
            normals = source_normals_for_legacy_mesh(pv_root, hy3d_root, r, m) @ np.linalg.inv(T[:3, :3])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
            faces = m.faces[:, ::-1] if np.linalg.det(T[:3, :3]) < 0 else m.faces
            out = trimesh.Trimesh(vertices=m.vertices @ T[:3, :3].T + T[:3, 3],
                                  faces=faces, vertex_normals=normals, process=False)
            out.visual = m.visual
        else:
            out = apply_T(m, placement)
    mo = r.get('material_override')
    if mo and any(key in mo for key in ('base_color_factor', 'metallic_factor', 'roughness_factor', 'remove_base_color_texture')):
        if hasattr(out.visual, 'material'):
            mat = out.visual.material.copy()
        else:
            # Generated parts may omit a material entirely. Do not bake
            # trimesh's synthetic gray fallback as if it were source color.
            # Explicit vertex colors need a separate color-preserving path:
            # TextureVisuals.copy() drops COLOR_0 in this trimesh version.
            if out.visual.kind in ('vertex', 'face'):
                raise ValueError('PBR factor override of explicit vertex colors requires a color-preserving material baseline')
            mat = trimesh.visual.material.PBRMaterial()
            out.visual = trimesh.visual.texture.TextureVisuals(material=mat)
        for field, attribute in [('metallic_factor', 'metallicFactor'), ('roughness_factor', 'roughnessFactor')]:
            if field in mo:
                value = float(mo[field])
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'Invalid material_override {field}')
                setattr(mat, attribute, value)
        if 'base_color_factor' in mo:
            value = np.asarray(mo['base_color_factor'], dtype=float)
            if value.shape != (4,) or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
                raise ValueError('Invalid material_override base_color_factor')
            mat.baseColorFactor = np.rint(value * 255).astype(np.uint8)
        if 'remove_base_color_texture' in mo:
            if not isinstance(mo['remove_base_color_texture'], bool):
                raise ValueError('Invalid material_override remove_base_color_texture; expected boolean')
            if mo['remove_base_color_texture']:
                mat.baseColorTexture = None
        out.visual.material = mat
    if mo and 'roughness_floor' in mo:
        mat = out.visual.material
        mr = np.asarray(mat.metallicRoughnessTexture.convert('RGB')).copy()
        mr[..., 1] = np.maximum(mr[..., 1], int(round(mo['roughness_floor'] * 255)))
        mat.metallicRoughnessTexture = Image.fromarray(mr)
    if mo and mo.get('finish') == 'brushed_steel':
        mat = out.visual.material.copy()
        if mat.baseColorTexture is not None:
            rgb = np.asarray(mat.baseColorTexture.convert('RGB'), dtype=float)
            value = 105 + 45 * rgb.mean(axis=2) / 255
            steel = np.clip(value[..., None] * np.array([.92, .97, 1.]), 0, 255).astype(np.uint8)
            mat.baseColorTexture = Image.fromarray(steel)
        mat.baseColorFactor = [180, 185, 190, 255]
        mat.metallicRoughnessTexture = None
        mat.metallicFactor = .88
        mat.roughnessFactor = .30
        mat.emissiveTexture = None
        mat.emissiveFactor = [0., 0., 0.]
        out.visual.material = mat
    return out


def check(nodes, want, t):
    if set(nodes) != set(want):
        return f'turn{t:02d} node set diff {set(nodes) ^ set(want)}'
    for n, g in nodes.items():
        s, w = sig(g), want[n]
        if s['nv'] != w['nv']:
            return f'turn{t:02d} {n} nv {s["nv"]} vs {w["nv"]}'
        if abs(s['vs'] - w['vs']) > 1e-2:
            return f'turn{t:02d} {n} vs {s["vs"]} vs {w["vs"]}'
        if (w.get('bc'), w.get('mr')) != (s['bc'], s['mr']):
            return f'turn{t:02d} {n} texture hash diff'
    return None


def assemble(chain, pv_root, hy3d_root, out_dir, verify=True, manual=True, only_turn=None, export=True, state_callback=None, export_turns=None):
    rep = json.load(open(os.path.join(chain, 'place_report.json')))
    man = json.load(open(os.path.join(chain, 'turn00_manifest.json')))
    checks = (json.load(open(os.path.join(chain, 'checks.json')))
              if verify and os.path.isfile(os.path.join(chain, 'checks.json')) else None)
    os.makedirs(out_dir, exist_ok=True)
    nodes = build_turn00(man, pv_root, chain)
    initial_nodes = dict(nodes)
    # Exact source-restoration patches retain authored float32 NORMAL values.
    recovery_path = os.path.join(chain, 'baseline_recovery.json')
    # Explicit baseline copies may retain already serialized authored normals.
    # Opt in only: old manifests retain their established export behavior.
    preserve_authored_normals = man.get('preserve_authored_normals') is True
    if os.path.isfile(recovery_path):
        recovery = json.load(open(recovery_path))
        preserve_authored_normals = preserve_authored_normals or (
            recovery.get('kind') == 'ORIGINAL_SOURCE_MISSING_SURFACES'
            and recovery.get('preserve_authored_normals') is True)

    records = {0: man, **{int(row['turn']): row for row in rep}}

    def dump(t):
        current_nodes = posed(nodes, records[t]) if manual else nodes
        if state_callback is not None:
            state_callback(t, current_nodes)
        if not export or (only_turn is not None and t != only_turn) or (export_turns is not None and t not in export_turns):
            return
        sc = trimesh.Scene()
        for n, g in current_nodes.items():
            if preserve_authored_normals and man.get('materialize_missing_normals', True):
                # Preserve existing authored values; materialize only when the
                # explicitly opted-in baseline has no cached NORMAL accessor.
                _ = g.vertex_normals
            sc.add_geometry(g, node_name=n, geom_name=n)
        sc.export(os.path.join(out_dir, f'turn{t:02d}.glb'), unitize_normals=not preserve_authored_normals)

    def verify_t(t):
        if checks is None:
            return None
        return check(posed(nodes, records[t]), checks[f'turn{t:02d}'], t)

    dump(0)
    err = verify_t(0)
    if err:
        return err
    # A turn may hold several records; its state is dumped and verified once, after the last of them.
    last_index = {int(row['turn']): i for i, row in enumerate(rep)}
    for i, r in enumerate(rep):
        if only_turn is not None and int(r['turn']) > only_turn:
            break
        t, op = r['turn'], r.get('op') or 'replace'
        key = r.get('glb_node') or ('slot_' + r['slot'])
        if op == 'replace' and r.get('self_transform') is not None:
            nodes[key] = apply_T(nodes[key], r['self_transform'])
        elif op in ('add', 'replace') and r.get('initial_node'):
            nodes[key] = apply_T(initial_nodes[r['initial_node']], r['transform'])
        elif op in ('add', 'replace'):
            nodes[key] = build_piece(r, pv_root, hy3d_root, chain)
        elif op == 'remove':
            for k in (['slot_' + s for s in r.get('removed_slots', [])] or [key]):
                del nodes[k]
        elif op == 'retexture':
            rp = os.path.join(chain, 'retex', f'turn{t:02d}_{key}.glb')
            sc = trimesh.load(rp, process=False, file_type='glb')
            nodes[key] = sc.geometry[key] if key in sc.geometry \
                else next(iter(sc.geometry.values()))
            # Baked appearances retain their original placement. A later recipe
            # repair can move that appearance without modifying the asset copy.
            if r.get('retexture_transform') is not None:
                T = np.asarray(r['retexture_transform'], dtype=float)
                if (T.shape != (4, 4) or not np.isfinite(T).all()
                        or not np.allclose(T[3], [0, 0, 0, 1])
                        or abs(np.linalg.det(T[:3, :3])) < 1e-12):
                    raise ValueError(f'turn{t:02d}: invalid retexture_transform')
                mesh = nodes[key].copy()
                normals = np.asarray(nodes[key].vertex_normals).copy()
                mesh.apply_transform(T)
                # Inverse transpose is required for nonuniform scaling.
                normals = normals @ np.linalg.inv(T[:3, :3])
                lengths = np.linalg.norm(normals, axis=1, keepdims=True)
                mesh.vertex_normals = normals / np.maximum(lengths, 1e-30)
                nodes[key] = mesh
        # Explicit placement corrections for retained dependent parts, e.g. a
        # pot carried onto a replacement hat. Apply deltas to current states.
        for dependent, transform in r.get('dependent_transforms', {}).items():
            if dependent == key or dependent not in nodes:
                raise ValueError(f'turn{t:02d}: invalid dependent node {dependent}')
            nodes[dependent] = apply_T(nodes[dependent], transform)
        if i != last_index[int(t)]:
            continue
        dump(t)
        err = verify_t(t)
        if err:
            return err
    return None


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pv-root', required=True)
    ap.add_argument('--chain', required=True)
    ap.add_argument('--hy3d-root', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-verify', action='store_true')
    a = ap.parse_args()
    chain = a.chain.rstrip('/')
    hy3d = a.hy3d_root or os.path.join(os.path.dirname(os.path.dirname(chain)), 'hy3d_parts')
    out = a.out or os.path.join(chain, 'assembled')
    err = assemble(chain, a.pv_root, hy3d, out, verify=not a.no_verify)
    name = '/'.join(chain.split('/')[-2:])
    if err:
        print(f'FAIL {name}: {err}'); sys.exit(1)
    print(f'PASS {name}' + ('' if a.no_verify else ' (verified vs checks.json)'))
