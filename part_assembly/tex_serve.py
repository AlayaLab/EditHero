"""Resident TRELLIS.2 texturing worker. Same pattern as blender_serve: it watches a spool directory for jobs and stays alive.

Why resident: loading the 4B model takes about two minutes, and material chains run it every turn, so a cold start costs more than inference.

    <spool>/job_XXXX.json   {"mesh": target part glb, "ref": reference image png, "out": output glb,
                             optional "uv": "trellis" (default) | "keep", "node": node name in the output glb,
                             "preserve": {"mask": texels of the old UV changed after the bake (255)} or {"prev": previous part, "faces": face id json}}
    <spool>/job_XXXX.json.done   "ok <seconds> {...}" or "error ..."
    <spool>/STOP            stop

**Where the UV comes from (default "trellis")**: original part UVs often overlap (tiling, mirroring, stacking, front and
back sharing texels). One texel is shared by surfaces far apart and can only store the colour of one of them, so patterned materials get
chopped up. By default cumesh therefore unwraps a new non-overlapping UV for the part (what TRELLIS.2 does anyway when no UV is given),
and the bake uses it.
**The 3D mesh does not change**, by construction: the unwrap returns a map from each new vertex to its original vertex; output positions
and normals are taken from the original through this map, and the triangle set (compared by the coordinates of the three vertices) is
checked to equal the original. The glb vertex array gains duplicate positions along UV seams, because glTF stores one UV per vertex;
the geometry itself does not change.
"keep" (option) reuses the original UV, and output vertices, faces and UVs are bit-identical to the original.

**Regions that keep their colour** (preserve): some material turns change only part of a part (only the hat, keeping the face and eyes).
Given the previous part and the faces to keep, the texels of these faces take their colour from the previous texture by 3D position and
are written into the texture of the new UV.

On the reuse-UV path, normals come from trimesh's read-only cache and postprocess modifies them in place, which raises
"assignment destination is read-only", so a writable copy is passed in.
Only the two output textures are used (base color, metallic-roughness).

Run (must be in the trellis2 environment, with TRITON_* pointing at triton's bundled ptxas, see run_tex.sh):
    python tex_serve.py --spool <directory>
"""
import os, sys, json, time, glob, argparse, traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.environ.get("PXFORM_TRELLIS_ROOT", "./TRELLIS.2"))
import numpy as np
import trimesh
from PIL import Image


class _Writable(trimesh.Trimesh):
    @property
    def vertex_normals(self):
        return np.array(trimesh.Trimesh.vertex_normals.fget(self), dtype=np.float64)


def _unwrap(vertices, faces):
    """Non-overlapping UV for (vertices, faces): cumesh first (what TRELLIS.2 uses), xatlas when cumesh fails (CUDA launch
    errors on some meshes) or drops a face. Returns vmap (new vertex -> input vertex), new faces, uv (cumesh convention)."""
    import torch
    try:
        import cumesh
        cm = cumesh.CuMesh()
        cm.init(torch.from_numpy(np.asarray(vertices)).float().cuda(), torch.from_numpy(np.asarray(faces)).int().cuda())
        _, f2, uv2, vmap = cm.uv_unwrap(return_vmaps=True)
        f2 = f2.cpu().numpy().astype(np.int64)
        if len(f2) == len(faces):
            return vmap.cpu().numpy().astype(np.int64), f2, uv2.cpu().numpy().astype(np.float64)
        print(f"[tex_serve] cumesh dropped faces ({len(faces)} -> {len(f2)}), using xatlas", flush=True)
    except Exception as e:
        print(f"[tex_serve] cumesh unwrap failed ({type(e).__name__}: {str(e)[:80]}), using xatlas", flush=True)
    import xatlas
    vmap, f2, uv2 = xatlas.parametrize(np.asarray(vertices, np.float32), np.asarray(faces, np.uint32))
    return vmap.astype(np.int64), f2.astype(np.int64), uv2.astype(np.float64)


def build_pipeline():
    from trellis2.pipelines import Trellis2TexturingPipeline
    pipe = Trellis2TexturingPipeline.from_pretrained(
        os.environ.get("PXFORM_TRELLIS_MODEL", "microsoft/TRELLIS.2-4B"), config_file="texturing_pipeline.json")
    pipe.cuda()
    _pre, _post = pipe.preprocess_mesh, pipe.postprocess_mesh
    pipe._vmap = None

    def pre(mesh):
        out = _pre(mesh)
        if getattr(mesh, "visual", None) is not None and getattr(mesh.visual, "uv", None) is not None:
            out.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv.copy())
        return out

    def post(mesh, *a, **kw):
        uv = getattr(getattr(mesh, "visual", None), "uv", None)
        if uv is not None:                                   # "keep": reuse the original UV
            m2 = _Writable(vertices=np.array(mesh.vertices, dtype=np.float64),
                           faces=np.array(mesh.faces), process=False)
            m2.visual = mesh.visual
            return _post(m2, *a, **kw)
        # "trellis": unwrap with cumesh as TRELLIS.2 does when no UV is given (xatlas on failure), but keep the vertex map
        vmap, f2, uv2 = _unwrap(mesh.vertices, mesh.faces)
        pipe._vmap = vmap
        m2 = _Writable(vertices=np.asarray(mesh.vertices, dtype=np.float64)[vmap], faces=f2, process=False)
        # the reuse-UV path of postprocess flips v before rasterising and flips it back on output; flip once here,
        # so rasterisation uses cumesh's original UV, exactly as on TRELLIS.2's own unwrap path
        uv_in = uv2.copy(); uv_in[:, 1] = 1 - uv_in[:, 1]
        m2.visual = trimesh.visual.TextureVisuals(uv=uv_in)
        m2._cache["vertex_normals"] = np.asarray(mesh.vertex_normals, dtype=np.float64)[vmap]
        return _post(m2, *a, **kw)

    pipe.preprocess_mesh, pipe.postprocess_mesh = pre, post
    return pipe


def _canon(P, F):
    """Each triangle as a tuple of its three vertex positions, independent of the start vertex and keeping orientation; returns the sorted array, used to compare triangle sets."""
    T = P[F]                                                   # (n, 3, 3)
    keys = [tuple(map(tuple, t)) for t in T]
    out = []
    for k in keys:
        i = min(range(3), key=lambda j: k[j])
        out.append(k[i:] + k[:i])
    out.sort()
    return out


def _tex_arrays(mat, size_hint):
    """Previous material -> (base color RGBA uint8 HxWx4, metallic-roughness uint8 HxWx3); constants when there is no texture."""
    bc = getattr(mat, "baseColorTexture", None)
    if bc is None and getattr(mat, "image", None) is not None:
        bc = mat.image
    if bc is not None:
        bc = np.asarray(bc.convert("RGBA"))
    else:
        f = getattr(mat, "baseColorFactor", None)
        if f is None:
            f = getattr(mat, "main_color", None)
        f = np.array([255, 255, 255, 255] if f is None else f, dtype=np.float64).ravel()
        if f.max() <= 1.0:
            f = f * 255
        bc = np.tile(np.clip(f[:4], 0, 255).astype(np.uint8), (size_hint, size_hint, 1))
    mr = getattr(mat, "metallicRoughnessTexture", None)
    if mr is not None:
        mr = np.asarray(mr.convert("RGB"))
    else:
        met = getattr(mat, "metallicFactor", None); rou = getattr(mat, "roughnessFactor", None)
        met = 0.0 if met is None else float(met); rou = 1.0 if rou is None else float(rou)
        mr = np.tile(np.array([0, int(rou * 255), int(met * 255)], np.uint8), (size_hint, size_hint, 1))
    return bc, mr


def _bilinear(img, u, v):
    """Bilinear lookup of trimesh uv (v up) in an HxWxC image; values outside [0,1] wrap as tiling."""
    H, W = img.shape[:2]
    x = (u % 1.0) * W - 0.5; y = (1.0 - (v % 1.0)) * H - 0.5
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64); fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    x0m, x1m = x0 % W, (x0 + 1) % W; y0m, y1m = np.clip(y0, 0, H - 1), np.clip(y0 + 1, 0, H - 1)
    I = img.astype(np.float64)
    out = (I[y0m, x0m] * (1 - fx) * (1 - fy) + I[y0m, x1m] * fx * (1 - fy)
           + I[y1m, x0m] * (1 - fx) * fy + I[y1m, x1m] * fx * fy)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _preserve(mesh, src_faces_of_new, keep_src_faces, prev, base, mr):
    """mesh is the part with the new UV (positions are the original positions). Set the texels on the original faces keep_src_faces to the colour of the previous part prev:
    rasterise the new UV to get the 3D position of every texel, find the same triangle in the previous part (matched by its three vertex positions),
    compute barycentric coordinates, interpolate its UV and sample its texture."""
    import torch, nvdiffrast.torch as dr
    keep = np.isin(src_faces_of_new, np.asarray(sorted(keep_src_faces), dtype=np.int64))
    if not keep.any():
        return base, mr, 0
    H = base.shape[0]
    V = np.asarray(mesh.vertices, np.float64); F = np.asarray(mesh.faces, np.int64)
    uv = np.asarray(mesh.visual.uv, np.float64).copy(); uv[:, 1] = 1 - uv[:, 1]
    clip = np.concatenate([uv * 2 - 1, np.zeros((len(uv), 1)), np.ones((len(uv), 1))], 1)
    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, torch.from_numpy(clip).float().cuda()[None], torch.from_numpy(F).int().cuda(), resolution=[H, H])
    tri = rast[0, ..., 3].long().cpu().numpy() - 1
    pos = dr.interpolate(torch.from_numpy(V).float().cuda()[None], rast, torch.from_numpy(F).int().cuda())[0][0].cpu().numpy()
    sel = tri >= 0
    sel[sel] = keep[tri[sel]]
    ys, xs = np.nonzero(sel)
    if len(ys) == 0:
        return base, mr, 0
    PV = np.asarray(prev.vertices, np.float64); PF = np.asarray(prev.faces, np.int64)
    puv = np.asarray(prev.visual.uv, np.float64)
    pkey = {}
    for j, t in enumerate(PV[PF]):
        pkey.setdefault(tuple(sorted(map(tuple, t))), j)
    nkey = np.array([pkey.get(tuple(sorted(map(tuple, t))), -1) for t in V[F]])
    pj = nkey[tri[ys, xs]]
    ok = pj >= 0
    ys, xs, pj = ys[ok], xs[ok], pj[ok]
    p = pos[ys, xs].astype(np.float64)
    A, B, C = PV[PF[pj, 0]], PV[PF[pj, 1]], PV[PF[pj, 2]]
    v0, v1, v2 = B - A, C - A, p - A
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1); d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01; den[np.abs(den) < 1e-30] = 1e-30
    b1 = (d11 * d20 - d01 * d21) / den; b2 = (d00 * d21 - d01 * d20) / den; b0 = 1 - b1 - b2
    U = b0[:, None] * puv[PF[pj, 0]] + b1[:, None] * puv[PF[pj, 1]] + b2[:, None] * puv[PF[pj, 2]]
    pbc, pmr = _tex_arrays(prev.visual.material, 4)
    base = base.copy(); mr = mr.copy()
    base[ys, xs] = _bilinear(pbc, U[:, 0], U[:, 1])
    mr[ys, xs] = _bilinear(pmr, U[:, 0], U[:, 1])
    return base, mr, int(len(ys))


def _preserve_mask(mesh, src, mask, base, mr):
    """mesh: the new-UV part (its vertices are src's, reordered); src: the part as it was (old UV, old texture);
    mask: HxW uint8 in src's texture space, 255 where the old texture was changed after baking (stickers, eyes, restored
    regions). Every new texel whose surface point falls on a masked old texel takes the old colour."""
    import torch, nvdiffrast.torch as dr
    H = base.shape[0]
    V = np.asarray(mesh.vertices, np.float64); F = np.asarray(mesh.faces, np.int64)
    uv = np.asarray(mesh.visual.uv, np.float64).copy(); uv[:, 1] = 1 - uv[:, 1]
    clip = np.concatenate([uv * 2 - 1, np.zeros((len(uv), 1)), np.ones((len(uv), 1))], 1)
    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, torch.from_numpy(clip).float().cuda()[None], torch.from_numpy(F).int().cuda(), resolution=[H, H])
    tri = rast[0, ..., 3].long().cpu().numpy() - 1
    pos = dr.interpolate(torch.from_numpy(V).float().cuda()[None], rast, torch.from_numpy(F).int().cuda())[0][0].cpu().numpy()
    ys, xs = np.nonzero(tri >= 0)
    PV = np.asarray(src.vertices, np.float64); PF = np.asarray(src.faces, np.int64); puv = np.asarray(src.visual.uv, np.float64)
    pkey = {}
    for j, t in enumerate(PV[PF]):
        pkey.setdefault(tuple(sorted(map(tuple, t))), j)
    nkey = np.array([pkey.get(tuple(sorted(map(tuple, t))), -1) for t in V[F]])
    pj = nkey[tri[ys, xs]]; ok = pj >= 0; ys, xs, pj = ys[ok], xs[ok], pj[ok]
    p = pos[ys, xs].astype(np.float64)
    A, B, C = PV[PF[pj, 0]], PV[PF[pj, 1]], PV[PF[pj, 2]]
    v0, v1, v2 = B - A, C - A, p - A
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1); d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01; den[np.abs(den) < 1e-30] = 1e-30
    b1 = (d11 * d20 - d01 * d21) / den; b2 = (d00 * d21 - d01 * d20) / den; b0 = 1 - b1 - b2
    U = b0[:, None] * puv[PF[pj, 0]] + b1[:, None] * puv[PF[pj, 1]] + b2[:, None] * puv[PF[pj, 2]]
    mh, mw = mask.shape
    mx = np.clip(((U[:, 0] % 1.0) * mw).astype(np.int64), 0, mw - 1); my = np.clip(((1.0 - U[:, 1] % 1.0) * mh).astype(np.int64), 0, mh - 1)
    sel = mask[my, mx] > 127
    ys, xs, U = ys[sel], xs[sel], U[sel]
    if len(ys) == 0:
        return base, mr, 0
    pbc, pmr = _tex_arrays(src.visual.material, 4)
    base = base.copy(); mr = mr.copy()
    base[ys, xs] = _bilinear(pbc, U[:, 0], U[:, 1])
    mr[ys, xs] = _bilinear(pmr, U[:, 0], U[:, 1])
    global _LAST_WRITTEN
    _LAST_WRITTEN = (ys, xs)
    return base, mr, int(len(ys))


_LAST_WRITTEN = None


def _map_new_to_old(mesh, src, H):
    """For every texel of the new-UV part (H x H): its triangle in the new mesh, the same triangle in src (matched by the
    three vertex positions) and the point's UV in src. Returns ys, xs, U (src uv), pj (src face), nf (new face)."""
    import torch, nvdiffrast.torch as dr
    V = np.asarray(mesh.vertices, np.float64); F = np.asarray(mesh.faces, np.int64)
    uv = np.asarray(mesh.visual.uv, np.float64).copy(); uv[:, 1] = 1 - uv[:, 1]
    clip = np.concatenate([uv * 2 - 1, np.zeros((len(uv), 1)), np.ones((len(uv), 1))], 1)
    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, torch.from_numpy(clip).float().cuda()[None], torch.from_numpy(F).int().cuda(), resolution=[H, H])
    tri = rast[0, ..., 3].long().cpu().numpy() - 1
    pos = dr.interpolate(torch.from_numpy(V).float().cuda()[None], rast, torch.from_numpy(F).int().cuda())[0][0].cpu().numpy()
    ys, xs = np.nonzero(tri >= 0)
    PV = np.asarray(src.vertices, np.float64); PF = np.asarray(src.faces, np.int64); puv = np.asarray(src.visual.uv, np.float64)
    pkey = {}
    for j, t in enumerate(PV[PF]):
        pkey.setdefault(tuple(sorted(map(tuple, t))), j)
    nkey = np.array([pkey.get(tuple(sorted(map(tuple, t))), -1) for t in V[F]])
    nf = tri[ys, xs]; pj = nkey[nf]; ok = pj >= 0
    ys, xs, pj, nf = ys[ok], xs[ok], pj[ok], nf[ok]
    p = pos[ys, xs].astype(np.float64)
    A, B, C = PV[PF[pj, 0]], PV[PF[pj, 1]], PV[PF[pj, 2]]
    v0, v1, v2 = B - A, C - A, p - A
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1); d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01; den[np.abs(den) < 1e-30] = 1e-30
    b1 = (d11 * d20 - d01 * d21) / den; b2 = (d00 * d21 - d01 * d20) / den; b0 = 1 - b1 - b2
    U = b0[:, None] * puv[PF[pj, 0]] + b1[:, None] * puv[PF[pj, 1]] + b2[:, None] * puv[PF[pj, 2]]
    return ys, xs, U, pj, nf


def _face_tbn(P, Fc, UV):
    """Per-face tangent, bitangent, normal (orthonormal, handedness kept) from positions and UVs."""
    p0, p1, p2 = P[Fc[:, 0]], P[Fc[:, 1]], P[Fc[:, 2]]; t0, t1, t2 = UV[Fc[:, 0]], UV[Fc[:, 1]], UV[Fc[:, 2]]
    e1, e2 = p1 - p0, p2 - p0; d1, d2 = t1 - t0, t2 - t0
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]; r = np.where(np.abs(r) < 1e-20, 1e-20, r)
    Tg = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / r[:, None]; Bg = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / r[:, None]
    N = np.cross(e1, e2); N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-30)
    Tg = Tg - N * (N * Tg).sum(1, keepdims=True); Tg /= np.maximum(np.linalg.norm(Tg, axis=1, keepdims=True), 1e-30)
    hand = np.sign((np.cross(N, Tg) * Bg).sum(1, keepdims=True)); hand[hand == 0] = 1
    return Tg, np.cross(N, Tg) * hand, N


def _carry_channels(mesh, src, base):
    """Channels TRELLIS.2 never generates are carried over from src for every texel of the new UV: alpha (when the
    material is MASK/BLEND), emissive, occlusion, and the normal map (tangent space re-expressed in the new UV's
    tangent frame). Returns base (alpha updated) and the material keyword arguments."""
    import cv2
    m = src.visual.material; T = base.shape[0]
    kw = dict(doubleSided=getattr(m, 'doubleSided', None))
    am = getattr(m, 'alphaMode', None)
    extra = [k for k in ('emissiveTexture', 'occlusionTexture', 'normalTexture') if getattr(m, k, None) is not None]
    if not extra and am in (None, 'OPAQUE'):
        return base, kw
    ys, xs, U, pj, nf = _map_new_to_old(mesh, src, T)
    hole = np.ones((T, T), np.uint8); hole[ys, xs] = 0
    if am in ('MASK', 'BLEND'):
        bc = getattr(m, 'baseColorTexture', None)
        if bc is not None:
            a = np.asarray(bc.convert('RGBA'))[..., 3:]
            base = base.copy(); base[ys, xs, 3] = _bilinear(a, U[:, 0], U[:, 1])[:, 0]
        kw.update(alphaMode=am, alphaCutoff=getattr(m, 'alphaCutoff', None))
    for k in ('emissiveTexture', 'occlusionTexture'):
        t = getattr(m, k, None)
        if t is None: continue
        arr = np.zeros((T, T, 3), np.uint8); arr[ys, xs] = _bilinear(np.asarray(t.convert('RGB')), U[:, 0], U[:, 1])
        kw[k] = Image.fromarray(cv2.inpaint(arr, hole, 1, cv2.INPAINT_TELEA))
    if getattr(m, 'emissiveFactor', None) is not None:
        kw['emissiveFactor'] = m.emissiveFactor
    t = getattr(m, 'normalTexture', None)
    if t is not None:
        nt = _bilinear(np.asarray(t.convert('RGB')), U[:, 0], U[:, 1]).astype(np.float64) / 255 * 2 - 1
        To, Bo, No = _face_tbn(np.asarray(src.vertices, np.float64), np.asarray(src.faces), np.asarray(src.visual.uv, np.float64))
        Tn, Bn, Nn = _face_tbn(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces), np.asarray(mesh.visual.uv, np.float64))
        n = To[pj] * nt[:, 0:1] + Bo[pj] * nt[:, 1:2] + No[pj] * nt[:, 2:3]
        q = np.stack([(n * Tn[nf]).sum(1), (n * Bn[nf]).sum(1), (n * Nn[nf]).sum(1)], 1)
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
        arr = np.zeros((T, T, 3), np.uint8); arr[..., 2] = 255; arr[..., :2] = 128
        arr[ys, xs] = np.clip(np.rint((q * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
        kw['normalTexture'] = Image.fromarray(cv2.inpaint(arr, hole, 1, cv2.INPAINT_TELEA))
    return base, kw


def retexture(pipe, mesh_path, ref_path, out_path, seed=0, resolution=512, texture_size=2048,
              uv="trellis", node=None, preserve=None, bake_mesh=None):
    src = trimesh.load(mesh_path, process=False, force="mesh")
    if uv == "transfer":
        # no generation: unwrap a new non-overlapping UV and carry every point's colour from the current texture by 3D position (stickers, eyes, hand-fixed regions and flat fills alike)
        import torch, cumesh, cv2
        P0 = np.asarray(src.vertices, np.float64); F0 = np.asarray(src.faces, np.int64)
        N0 = np.asarray(src.vertex_normals, np.float64).copy()
        vmap, F2, uvn = _unwrap(P0, F0)
        uvn = uvn.copy(); uvn[:, 1] = 1 - uvn[:, 1]                                     # same convention as the UV output by TRELLIS.2
        P2 = P0[vmap]
        if len(F2) != len(F0) or _canon(P2, F2) != _canon(P0, F0):
            raise RuntimeError(f"triangle set differs from the original after unwrapping: {len(F0)} -> {len(F2)} faces")
        baked = trimesh.Trimesh(vertices=P2, faces=F2, vertex_normals=N0[vmap], process=False,
                                visual=trimesh.visual.TextureVisuals(uv=uvn))
        T = int(texture_size)
        base = np.zeros((T, T, 4), np.uint8); mr = np.zeros((T, T, 3), np.uint8)
        base, mr, n = _preserve_mask(baked, src, np.full((8, 8), 255, np.uint8), base, mr)
        hole = np.ones((T, T), np.uint8); hole[_LAST_WRITTEN] = 0                     # texels not covered by any triangle: fill the margin (transparent cut-out texels excluded)
        base = np.concatenate([cv2.inpaint(np.ascontiguousarray(base[..., :3]), hole, 3, cv2.INPAINT_TELEA),
                               np.where(hole[..., None] > 0, 255, base[..., 3:])], -1).astype(np.uint8)
        mr = cv2.inpaint(np.ascontiguousarray(mr), hole, 1, cv2.INPAINT_TELEA)
        base, kw = _carry_channels(baked, src, base)
        baked.visual = trimesh.visual.TextureVisuals(uv=uvn, material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(base), metallicRoughnessTexture=Image.fromarray(mr), **kw))
        baked._cache["vertex_normals"] = N0[vmap]
        info = dict(uv="transfer", vertices=int(len(P0)), faces=int(len(F0)), glb_vertices=int(len(P2)), transferred_texels=n)
    elif uv == "keep":
        if getattr(src.visual, "uv", None) is None:
            raise RuntimeError("part has no UV, cannot bake back")
        out = pipe.run(src, Image.open(ref_path), seed=seed,
                       resolution=resolution, texture_size=texture_size)
        m = out.visual.material
        baked = trimesh.Trimesh(vertices=src.vertices.copy(), faces=src.faces.copy(), process=False)
        baked.visual = trimesh.visual.TextureVisuals(
            uv=src.visual.uv.copy(),
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=m.baseColorTexture,
                metallicRoughnessTexture=getattr(m, "metallicRoughnessTexture", None)))
        # bitwise check: vertices and faces must equal the original
        assert np.array_equal(np.asarray(baked.vertices), np.asarray(src.vertices))
        assert np.array_equal(np.asarray(baked.faces), np.asarray(src.faces))
        info = dict(uv="keep", vertices=int(len(src.vertices)), faces=int(len(src.faces)))
    elif bake_mesh:
        # the original bake was not made on this part (the part moved after baking, or was baked merged with another part into one mesh): regenerate
        # on that original geometry with a new UV, then map each corner's new UV back to the current part by triangle. Positions and normals are the current part's; the 3D mesh does not change.
        from scipy.spatial import cKDTree
        G = trimesh.load(bake_mesh, process=False, force="mesh")
        PG = np.asarray(G.vertices, np.float64); FG = np.asarray(G.faces, np.int64)
        bare = trimesh.Trimesh(vertices=PG.copy(), faces=FG.copy(), process=False)
        bare._cache["vertex_normals"] = np.asarray(G.vertex_normals, np.float64).copy()
        pipe._vmap = None
        out = pipe.run(bare, Image.open(ref_path), seed=seed, resolution=resolution, texture_size=texture_size)
        vmapG = pipe._vmap; F2G = np.asarray(out.faces, np.int64); uvG = np.asarray(out.visual.uv, np.float64)
        gkey = {}
        for gi, t in enumerate(PG[FG]):
            gkey.setdefault(tuple(sorted(map(tuple, t))), gi)
        corner_uv = {}                                   # (G face, G vertex) -> new uv
        for i, f in enumerate(F2G):
            gi = gkey[tuple(sorted(map(tuple, PG[vmapG[f]])))]
            for c in range(3):
                corner_uv[(gi, int(vmapG[f[c]]))] = uvG[f[c]]
        P0 = np.asarray(src.vertices, np.float64); F0 = np.asarray(src.faces, np.int64)
        N0 = np.asarray(src.vertex_normals, np.float64).copy()
        size = np.linalg.norm(P0.max(0) - P0.min(0))
        if len(F0) == len(FG) and np.array_equal(F0, FG):
            gj = np.arange(len(F0)); A0, AG = P0, PG; mode = "index"
        else:
            nrm = lambda X, R: (X - (R.min(0) + R.max(0)) / 2) / (R.max(0) - R.min(0)).max()
            d, gj = cKDTree(PG[FG].mean(1)).query(P0[F0].mean(1)); A0, AG, mode = P0, PG, "position"
            if (d > 1e-4 * size).any():
                d, gj = cKDTree(nrm(PG, PG)[FG].mean(1)).query(nrm(P0, P0)[F0].mean(1)); A0, AG, mode = nrm(P0, P0), nrm(PG, PG), "position_normalised"
                if (d > 1e-3).any():
                    raise RuntimeError(f"{int((d > 1e-3).sum())} faces of the part are not in the bake mesh")
        uvs = np.zeros((len(F0) * 3, 2))
        for k in range(len(F0)):
            g = gj[k]
            if mode == "index":
                order = [0, 1, 2]
            else:
                dd = np.linalg.norm(A0[F0[k]][:, None, :] - AG[FG[g]][None, :, :], axis=2); order = list(dd.argmin(1))
                if len(set(order)) != 3: raise RuntimeError(f"corner pairing failed on face {k}")
            for c in range(3):
                uvs[3 * k + c] = corner_uv[(int(g), int(FG[g, order[c]]))]
        Vc = P0[F0].reshape(-1, 3); Fc = np.arange(len(F0) * 3).reshape(-1, 3); Nc = N0[F0].reshape(-1, 3)
        m = out.visual.material
        base = np.asarray(m.baseColorTexture.convert("RGBA")); mr = np.asarray(m.metallicRoughnessTexture.convert("RGB"))
        baked = trimesh.Trimesh(vertices=Vc, faces=Fc, vertex_normals=Nc, process=False, visual=trimesh.visual.TextureVisuals(uv=uvs))
        n_keep = 0
        if preserve and preserve.get("mask"):
            base, mr, n_keep = _preserve_mask(baked, src, np.asarray(Image.open(preserve["mask"]).convert("L")), base, mr)
        base, kw = _carry_channels(baked, src, base)
        baked.visual = trimesh.visual.TextureVisuals(uv=uvs, material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(base), metallicRoughnessTexture=Image.fromarray(mr), **kw))
        baked._cache["vertex_normals"] = Nc
        info = dict(uv="trellis_on_bake_mesh", match=mode, vertices=int(len(P0)), faces=int(len(F0)), glb_vertices=int(len(Vc)),
                    bake_mesh_faces=int(len(FG)), preserved_texels=n_keep)
    else:
        P0 = np.asarray(src.vertices, np.float64); F0 = np.asarray(src.faces, np.int64)
        N0 = np.asarray(src.vertex_normals, np.float64).copy()
        bare = trimesh.Trimesh(vertices=P0.copy(), faces=F0.copy(), process=False)   # no UV -> postprocess unwraps
        bare._cache["vertex_normals"] = N0.copy()
        pipe._vmap = None
        out = pipe.run(bare, Image.open(ref_path), seed=seed,
                       resolution=resolution, texture_size=texture_size)
        vmap = pipe._vmap
        F2 = np.asarray(out.faces, np.int64)
        P2 = P0[vmap]
        # 3D mesh unchanged: positions taken from the original through the map (bit-identical), triangle set checked by coordinates
        if len(F2) != len(F0) or _canon(P2, F2) != _canon(P0, F0):
            raise RuntimeError(f"triangle set differs from the original after unwrapping: {len(F0)} -> {len(F2)} faces")
        m = out.visual.material
        base = np.asarray(m.baseColorTexture.convert("RGBA"))
        mr = np.asarray(m.metallicRoughnessTexture.convert("RGB"))
        baked = trimesh.Trimesh(vertices=P2, faces=F2, vertex_normals=N0[vmap], process=False,
                                visual=trimesh.visual.TextureVisuals(uv=np.asarray(out.visual.uv, np.float64)))
        n_keep = 0
        if preserve and preserve.get("mask"):
            # texels changed after the bake (stickers, eyes, restored regions): carried from the original's old texture to the new UV by 3D position
            base, mr, n_keep = _preserve_mask(baked, src, np.asarray(Image.open(preserve["mask"]).convert("L")), base, mr)
        elif preserve:
            # new face -> original face id (matched by coordinates)
            idx = {}
            for i, t in enumerate(P0[F0]):
                idx.setdefault(tuple(sorted(map(tuple, t))), []).append(i)
            src_of_new = np.array([idx[tuple(sorted(map(tuple, t)))].pop(0) for t in P2[F2]], dtype=np.int64)
            prev = trimesh.load(preserve["prev"], process=False, force="mesh")
            faces_keep = json.load(open(preserve["faces"]))
            base, mr, n_keep = _preserve(baked, src_of_new, faces_keep, prev, base, mr)
        base, kw = _carry_channels(baked, src, base)
        baked.visual = trimesh.visual.TextureVisuals(
            uv=np.asarray(out.visual.uv, np.float64),
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(base), metallicRoughnessTexture=Image.fromarray(mr), **kw))
        baked._cache["vertex_normals"] = N0[vmap]
        info = dict(uv="trellis", vertices=int(len(P0)), faces=int(len(F0)), glb_vertices=int(len(P2)),
                    preserved_texels=n_keep)
    if node:
        sc = trimesh.Scene(); sc.add_geometry(baked, node_name=node, geom_name=node)
        sc.export(out_path)
    else:
        baked.export(out_path)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spool", required=True)
    ap.add_argument("--idle-timeout", type=float, default=3600.0)
    a = ap.parse_args()
    os.makedirs(a.spool, exist_ok=True)
    pipe = build_pipeline()
    print("[tex_serve] ready", flush=True)
    idle = 0.0
    while True:
        if os.path.exists(os.path.join(a.spool, "STOP")):
            print("[tex_serve] stop", flush=True); return
        todo = sorted(p for p in glob.glob(os.path.join(a.spool, "job_*.json"))
                      if not os.path.exists(p + ".done"))
        if not todo:
            time.sleep(0.3); idle += 0.3
            if idle > a.idle_timeout:
                print("[tex_serve] idle timeout, exiting", flush=True); return
            continue
        idle = 0.0
        for p in todo:
            try:
                open(p + ".started", "w").write(str(time.time()))   # split queue time from compute time (review #11)
                job = json.load(open(p))
                t0 = time.time()
                info = retexture(pipe, job["mesh"], job["ref"], job["out"],
                                 seed=int(job.get("seed", 0)),
                                 resolution=int(job.get("resolution", 512)),
                                 texture_size=int(job.get("texture_size", 2048)),
                                 uv=job.get("uv", "trellis"), node=job.get("node"),
                                 preserve=job.get("preserve"), bake_mesh=job.get("bake_mesh"))
                msg = f"ok {time.time() - t0:.1f}s {json.dumps(info)}"
            except Exception as e:
                traceback.print_exc(limit=3)
                msg = f"error {type(e).__name__}: {e}"
            tmp = p + ".done.tmp"
            open(tmp, "w").write(msg)
            os.replace(tmp, p + ".done")
            print(f"[tex_serve] {os.path.basename(p)} -> {msg[:120]}", flush=True)
            if msg.startswith("error") and ("CUDA" in msg or "cuda" in msg or "Accelerator" in msg):
                # after the CUDA context is poisoned every job fails instantly while still showing ALIVE (review #12): exit loudly; texfarm.sh restarts it
                print("[tex_serve] CUDA context poisoned, exiting; restart this GPU's worker with texfarm.sh start", flush=True)
                os._exit(3)


if __name__ == "__main__":
    main()
