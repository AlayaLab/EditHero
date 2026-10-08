"""Folders of the assembly-chain pipeline (chains that start from a host object's core and add its own parts back, one
to three per turn). Everything is configured by environment variables:

    PXFORM_LIBRARY_ROOT     part library: pv_textured/textured_part_glbs/<oid>/<pid>.glb and captions_qwen3.json
    PXFORM_ASSEMBLY_WORK    working folder of this pipeline (default ./work/assembly)
    PXFORM_ASSEMBLY_HOSTS   host list, one object id per line (default <work>/hosts.txt)

Layout of the working folder:
    candidates.json                 screen_hosts.py
    hosts.txt                       select_hosts.py
    preview/img/<oid>/f000V.png     preview_hosts.py
    hosts/<oid>/parts.json          export_parts.py   (planner input)
    plans/<oid>.json                the assembly plan (an LLM agent, prompts/plan_assembly.md; check_plan.py)
    chains/<oid>/                   build_recipe.py (recipe, turnNN.glb, img/), then the instruction files of each step
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(os.path.dirname(HERE))          # part_assembly/
sys.path.insert(0, ENGINE)
from local_paths import DATA_ROOT as LIB, BLENDER, RENDER_SCRIPT

TEX_PART = os.path.join(LIB, 'pv_textured', 'textured_part_glbs')
CAPTIONS = os.path.join(LIB, 'captions_qwen3.json')
WORK = os.environ.get('PXFORM_ASSEMBLY_WORK', './work/assembly')
HOSTS = os.environ.get('PXFORM_ASSEMBLY_HOSTS', os.path.join(WORK, 'hosts.txt'))


def hosts():
    return [l.strip() for l in open(HOSTS) if l.strip()]


def chain_dir(oid):
    return os.path.join(WORK, 'chains', oid)


def chain_id(oid):
    """Chain id of an assembly chain: family A (additions), first 8 characters of the host id, seed 0."""
    return f'A/{oid[:8]}_s0'
