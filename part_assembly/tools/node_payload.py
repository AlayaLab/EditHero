"""Content signature for lossless node replay, including rendering attributes."""
import hashlib
import json

import numpy as np
from PIL import Image


def _feed(digest, value):
    if isinstance(value, Image.Image):
        _feed(digest, ('image', value.mode, value.size, np.asarray(value)))
    elif isinstance(value, np.ndarray):
        _feed(digest, ('array', str(value.dtype), value.shape))
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        digest.update(b'{')
        for key in sorted(value):
            _feed(digest, key)
            _feed(digest, value[key])
        digest.update(b'}')
    elif isinstance(value, (list, tuple)):
        digest.update(b'[')
        for item in value:
            _feed(digest, item)
        digest.update(b']')
    elif isinstance(value, np.generic):
        _feed(digest, value.item())
    else:
        digest.update(json.dumps(value, sort_keys=True).encode())
        digest.update(b'\0')


def material_payload(material):
    if material is None:
        return None
    if hasattr(material, 'materials'):
        return [material_payload(m) for m in material.materials]
    if hasattr(material, '_data'):
        return dict(kind=type(material).__name__, data=material._data)
    # SimpleMaterial (OBJ-derived assets) does not use PBRMaterial._data.
    return dict(kind=type(material).__name__, **{
        key: getattr(material, key, None) for key in
        ('image', 'ambient', 'diffuse', 'specular', 'glossiness', 'kwargs')})


def node_signature(mesh):
    digest = hashlib.sha256()
    _feed(digest, mesh.vertices)
    _feed(digest, mesh.faces)
    _feed(digest, mesh.vertex_normals)
    visual = mesh.visual
    _feed(digest, visual.kind)
    if visual.kind == 'texture':
        _feed(digest, getattr(visual, 'uv', None))
        _feed(digest, getattr(visual, 'face_materials', None))
        _feed(digest, material_payload(visual.material))
    elif visual.kind == 'vertex':
        _feed(digest, visual.vertex_colors)
    elif visual.kind == 'face':
        _feed(digest, visual.face_colors)
    return digest.hexdigest()
