"""Non-accumulating per-state world-space placement deltas, in recipe JSON."""
import numpy as np

def matrix(value):
 t=np.asarray(value,dtype=float)
 if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1],atol=1e-10):
  raise ValueError('Pose must be a finite 4x4 affine matrix')
 sv=np.linalg.svd(t[:3,:3],compute_uv=False)
 if sv.min()<1e-5 or sv.max()>1e4 or np.linalg.det(t[:3,:3])<=0:
  raise ValueError('Scale must be positive and within valid range')
 return t

def posed(nodes, record):
 result=dict(nodes)
 for node,raw in record.get('manual_pose_overrides',{}).items():
  if node not in nodes:raise ValueError('Pose references non-existent part: '+node)
  t=matrix(raw)
  original=nodes[node]
  normals=np.array(original.vertex_normals,copy=True)
  mesh=original.copy();mesh.apply_transform(t)
  normals=normals @ np.linalg.inv(t[:3,:3])
  mesh.vertex_normals=normals / np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-30)
  result[node]=mesh
 return result
