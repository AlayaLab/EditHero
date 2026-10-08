"""Machine paths of the data engine, from environment variables.

    PXFORM_LIBRARY_ROOT     working directory of the part library: pv_textured/textured_part_glbs, captions and embeddings
    PXFORM_BLENDER          Blender executable
    PXFORM_RENDER_SCRIPT    blender_kit render script (scripts/render.py)
    PXFORM_ENCODER_PYTHON   interpreter of the TRELLIS.2 environment
    PXFORM_HY3D_MESH_ROOT   HY3D-Bench whole-object meshes, <object>/mesh.glb
    PXFORM_HY3D_PART_ROOTS  HY3D-Bench part directories, <object>/<part>.glb (os.pathsep separated)

Used only by part_assembly/annotate/ (captioning and assembly chains):
    PXFORM_PARTVERSE_ANNO   PartVerse-XL anno_infos (segmented meshes and face labels) for part_captions.py
    PXFORM_HY3D_COND_ROOT   HY3D-Bench renders with part masks, <object>.npz, for hy3d_captions.py
    PXFORM_ASSEMBLY_WORK    working folder of the assembly-chain pipeline; PXFORM_ASSEMBLY_HOSTS its host list
"""
import os

DATA_ROOT = os.environ.get("PXFORM_LIBRARY_ROOT", "./library")
BLENDER = os.environ.get("PXFORM_BLENDER", "blender")
RENDER_SCRIPT = os.environ.get("PXFORM_RENDER_SCRIPT", "./third_party/blender_kit/scripts/render.py")
ENCODER_PYTHON = os.environ.get("PXFORM_ENCODER_PYTHON", "python")
HY3D_MESH_ROOT = os.environ.get("PXFORM_HY3D_MESH_ROOT", "./library/hy3d/meshes")
HY3D_PART_ROOTS = tuple(filter(None, os.environ.get("PXFORM_HY3D_PART_ROOTS", "./data/parts/hy3d").split(os.pathsep)))
