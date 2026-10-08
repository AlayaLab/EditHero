"""Rigidly install disconnected source pieces without changing their UV/PBR/topology.

Recipe component_transforms partitions source vertex indices. Each component
transform acts in source coordinates BEFORE r.transform. A source SHA binds
indices; all triangles must belong entirely to one partition. Reflection,
shear, anisotropic scale, incomplete partitions and multi-instance sources
are rejected. Ordinary build_piece recipes do not call this module.
"""
import hashlib,os
import numpy as np
import trimesh

def build_source_instances(r, pv_root, hy3d_root, chain_dir=None):
    """Place repeated source meshes from JSON, preserving their full UV/PBR.

    Every instance has its own source SHA and proper similarity transform.
    Sources must carry one mesh in identity coordinates and the same complete
    material. No welding, texture atlas, mesh simplification or asset rewrite
    occurs. Instances transform source-local coordinates, then r.transform
    places the entire logical part. This supports e.g. three/five leg groups.
    An explicit asset_copy may reference a SHA-bound derived appearance inside
    the owning recipe. Its placement remains entirely in the instance JSON.
    """
    from node_payload import material_payload, _feed
    specs = r['source_instances']
    if not isinstance(specs, list) or not 2 <= len(specs) <= 32:
        raise ValueError('source_instances requires 2 to 32 source instances')
    if r.get('material_override') or r.get('component_transforms'):
        raise ValueError('source_instances cannot silently alter source materials or partitions')
    global_T = similarity(r.get('transform', np.eye(4)), 'instance group transform')
    vertices, normals, faces, uvs = [], [], [], []
    offset = 0
    material, material_digest = None, None
    for index, spec in enumerate(specs):
        if any(k in spec for k in ('source_instances', 'component_transforms', 'reinstall_snapshot', 'material_override')):
            raise ValueError('source_instances requires a direct source or explicit asset copy')
        if spec.get('asset_copy') is not None:
            if chain_dir is None:
                raise ValueError('instance asset_copy requires its owning recipe directory')
            record = spec['asset_copy']
            if not isinstance(record, dict) or not isinstance(record.get('path'), str):
                raise ValueError('invalid instance asset_copy record')
            base = os.path.realpath(chain_dir)
            relative = record['path']
            path = os.path.realpath(os.path.join(base, relative))
            if os.path.isabs(relative) or os.path.commonpath([base, path]) != base:
                raise ValueError('instance asset_copy must remain inside its owning recipe')
            expected_digest = record.get('sha256')
            if spec.get('source_sha256') not in (None, expected_digest):
                raise ValueError('conflicting instance source SHA')
        else:
            if spec.get('ds', 'pv') not in ('pv', 'hy3d'):
                raise ValueError('unsupported instance source dataset')
            root = hy3d_root if spec.get('ds', 'pv') == 'hy3d' else pv_root
            path = os.path.join(root, str(spec['added_oid']), str(spec['added_pid']) + '.glb')
            expected_digest = spec.get('source_sha256')
        with open(path, 'rb') as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if digest != expected_digest:
            raise ValueError('instance source SHA mismatch')
        scene = trimesh.load(path, process=False, force='scene')
        nodes = list(scene.graph.nodes_geometry)
        if len(nodes) != 1 or len(scene.geometry) != 1:
            raise ValueError('instance source requires one mesh instance')
        graph_T, name = scene.graph[nodes[0]]
        if not np.array_equal(graph_T, np.eye(4)):
            raise ValueError('instance source graph must use identity coordinates')
        mesh = scene.geometry[name]
        if not isinstance(mesh, trimesh.Trimesh) or mesh.visual.kind != 'texture' or mesh.visual.uv is None:
            raise ValueError('instance source requires a textured triangle mesh and explicit UV')
        if getattr(mesh.visual, 'face_materials', None) is not None or set(mesh.visual.vertex_attributes.keys()) - {'uv'}:
            raise ValueError('instance source has unsupported per-face material or custom vertex attributes')
        h = hashlib.sha256()
        _feed(h, material_payload(mesh.visual.material))
        if material_digest is None:
            material_digest = h.digest()
            material = mesh.visual.material.copy()
        elif h.digest() != material_digest:
            raise ValueError('source_instances require identical complete PBR; no atlas packing')
        T = global_T @ similarity(spec['transform'], 'source instance ' + str(index))
        n = np.asarray(mesh.vertex_normals) @ np.linalg.inv(T[:3, :3])
        lengths = np.linalg.norm(n, axis=1, keepdims=True)
        if not np.isfinite(n).all() or (lengths <= 1e-30).any():
            raise ValueError('invalid instance source normals')
        vertices.append(mesh.vertices @ T[:3, :3].T + T[:3, 3])
        normals.append(n / lengths)
        faces.append(mesh.faces + offset)
        uvs.append(mesh.visual.uv.copy())
        offset += len(mesh.vertices)
    result = trimesh.Trimesh(vertices=np.vstack(vertices), faces=np.vstack(faces),
                             vertex_normals=np.vstack(normals), process=False)
    result.visual = trimesh.visual.texture.TextureVisuals(uv=np.vstack(uvs), material=material)
    return result

def similarity(value,label):
    T=np.asarray(value,dtype=float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):raise ValueError(label+' invalid homogeneous transform')
    A=T[:3,:3];s2=float(np.trace(A.T@A)/3)
    if s2<=0 or np.linalg.det(A)<=0 or not np.allclose(A.T@A,np.eye(3)*s2,atol=1e-9,rtol=1e-7):raise ValueError(label+' requires proper rigid rotation and positive uniform scale')
    return T

def build_component_piece(r,pv_root,hy3d_root):
    if r.get('material_override'):
        raise ValueError('component transforms preserve source material; material overrides require a separate operation')
    spec=r['component_transforms']
    if spec.get('stage')!='source_local_before_global':raise ValueError('component transforms must declare source_local_before_global')
    pids=r.get('added_pids') or [str(r['added_pid'])]
    if len(set(map(str,pids)))!=len(pids):raise ValueError('duplicate component source parts')
    pieces=[]
    for pid in pids:
        p=os.path.join(hy3d_root if r.get('ds')=='hy3d' else pv_root,r['added_oid'],str(pid)+'.glb')
        with open(p,'rb') as f:digest=hashlib.sha256(f.read()).hexdigest()
        expected=spec.get('source_sha256') if len(pids)==1 else spec.get('source_sha256',{}).get(str(pid))
        if digest!=expected:raise ValueError('component source SHA mismatch')
        sc=trimesh.load(p,process=False,force='scene');instances=list(sc.graph.nodes_geometry)
        if len(instances)!=1 or len(sc.geometry)!=1:raise ValueError('each component source must have one geometry instance')
        graph_T,name=sc.graph[instances[0]]
        if not np.array_equal(graph_T,np.eye(4)):raise ValueError('component source graph must be identity; bake graph explicitly first')
        pieces.append(sc.geometry[name])
    if len(pieces)==1:m=pieces[0]
    else:
        from node_payload import material_payload,_feed
        digests=[]
        for piece in pieces:
            if piece.visual.kind!='texture' or piece.visual.uv is None or getattr(piece.visual,'face_materials',None) is not None:raise ValueError('multi-part components require explicit UV and one material per source')
            if set(piece.visual.vertex_attributes.keys())-{'uv'}:raise ValueError('multi-part component custom vertex attributes need explicit preservation')
            h=hashlib.sha256();_feed(h,material_payload(piece.visual.material));digests.append(h.digest())
        if any(h!=digests[0] for h in digests):raise ValueError('multi-part component sources require identical complete PBR; no atlas packing')
        offsets=np.cumsum([0]+[len(q.vertices) for q in pieces[:-1]])
        m=trimesh.Trimesh(vertices=np.vstack([q.vertices for q in pieces]),faces=np.vstack([q.faces+o for q,o in zip(pieces,offsets)]),vertex_normals=np.vstack([q.vertex_normals for q in pieces]),process=False)
        m.visual=trimesh.visual.texture.TextureVisuals(uv=np.vstack([q.visual.uv for q in pieces]),material=pieces[0].visual.material)
    nv=len(m.vertices);parts=spec.get('parts',[])
    if len(parts)<2:raise ValueError('component transform requires at least two independent pieces')
    labels=np.full(nv,-1,dtype=np.int64);V=np.empty_like(m.vertices);N=np.empty_like(m.vertex_normals)
    for i,part in enumerate(parts):
        ids=np.asarray(part['vertex_indices'])
        if ids.ndim!=1 or len(ids)==0 or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=len(ids) or ids.min()<0 or ids.max()>=nv:raise ValueError('invalid component vertex indices')
        if np.any(labels[ids]!=-1):raise ValueError('overlapping component partitions')
        labels[ids]=i;T=similarity(part['transform'],part.get('name',str(i)))
        V[ids]=m.vertices[ids]@T[:3,:3].T+T[:3,3]
        ns=m.vertex_normals[ids]@np.linalg.inv(T[:3,:3]);N[ids]=ns/np.maximum(np.linalg.norm(ns,axis=1,keepdims=True),1e-30)
    if np.any(labels<0):raise ValueError('incomplete component partition')
    if np.any(np.ptp(labels[m.faces],axis=1)!=0):raise ValueError('a triangle crosses component partitions')
    T=similarity(r['transform'],'global transform');V=V@T[:3,:3].T+T[:3,3];N=N@np.linalg.inv(T[:3,:3]);N/=np.maximum(np.linalg.norm(N,axis=1,keepdims=True),1e-30)
    out=trimesh.Trimesh(vertices=V,faces=m.faces.copy(),vertex_normals=N,process=False);out.visual=m.visual
    return out
