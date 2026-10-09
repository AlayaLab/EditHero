# The data engine: building a new chain

A chain starts from a textured, part-segmented **host** object and applies a sequence of edits to it, one per **turn**.
Every turn is one operation on the current state:

| Operation | What happens | Where the new geometry comes from |
|---|---|---|
| `replace` | a slot's part is swapped for another part | retrieved from the part library by caption, placed on the old part's contact surface |
| `add` | a new part is mounted on a free face of the object (or on the ground beside it) | retrieved from the part library by caption |
| `remove` | a slot's part is taken away (the engine refuses removals that would leave other parts floating) | none |
| `retexture` | a part gets a new material | the part is restyled in an image (Qwen-Image-Edit) and re-textured in 3D (TRELLIS.2) |

The host is divided into **slots**: named groups of library parts (`head`, `left_arm`, `roof`, ...). Slots whose name
starts with `anchor` are the fixed core and are never edited. Each placement must pass five checks before it is kept:
a contact surface exists, the snap to the host succeeds, the contact gap is at most 1% of the object size, the part is
not scaled past 2.5 times the old one, and the object is still one connected piece. In interactive chains, a turn whose
change is too small to see in the renders is rejected automatically.

The result of a chain is a **recipe**: `place_report.json` (one record per part and turn: operation, slot, source part,
transform, instruction), `turn00_manifest.json` (the start state), `checks.json` (signatures of every state) and
`provenance.json` (source objects and licenses). `tools/assemble.py` rebuilds every state from the recipe and the part
library and verifies it.

## 1. What you need

| | Used for | Setting |
|---|---|---|
| Python environment with mini-articraft, trimesh, scipy | placement (`chain_run.py`) | run the engine with it |
| Blender 4.2 and [blender_kit](https://github.com/AuroraRyan0301/Blender-Visualization-Skill) | all renders | `PXFORM_BLENDER`, `PXFORM_RENDER_SCRIPT` |
| Part library | retrieval and assembly | `PXFORM_LIBRARY_ROOT` |
| Python environment with torch and transformers | encoding retrieval queries (the placement environment needs no torch) | `PXFORM_ENCODER_PYTHON` |
| Qwen-Image-Edit-2509 environment | retexture turns (restyle) | `PXFORM_RESTYLE_PYTHON`, `PXFORM_RESTYLE_MODEL` |
| TRELLIS.2 environment | retexture turns (3D texturing) | `PXFORM_TEX_PYTHON`, `PXFORM_TRELLIS_ROOT`, `PXFORM_TRELLIS_MODEL` |

The part library folder holds the PartVerse-XL parts (`pv_textured/textured_part_glbs/<object>/<part>.glb`, from the
PartVerse-XL release; the dataset's `data/parts/` has every part the released chains use), the captions and their
embeddings:

```bash
cd part_assembly/annotate
python merge_captions.py --from-release EditHero/data/captions/captions_qwen3.json   # captions -> library files
python ingest_hy3d.py --from-release EditHero/data/captions/hy3d_captions.json      # optional: HY3D-Bench parts
python ../embed_library.py                                                           # caption embeddings for retrieval
```

## 2. Define the host's slots

The engine reads the slots of a host from the slot registry (`<library>/slot_registry.json`). For a host of the dataset,
take them from one of its chains:

```bash
python part_assembly/tools/registry_from_manifest.py EditHero/data/recipes/G/02593609_s1/turn00_manifest.json
```

For a new host, group its parts yourself and store the division:

```python
import slot_registry as SR
SR.put("b927ce627b6841a688067331853302d6", "coarse",
       groups={"anchor_torso": ["17", "2", "15", "0", "1", "9", "10", "20", "35"], "head": ["29"],
               "left_arm": ["42", "11", "36", "32", "33", "28", "39"], "right_arm": ["6", "37", "13", "16", "26", "5", "22", "3", "12"]},
       roles={"head": {"queries": ["a helmet worn on the head", "an animal head", "a round lantern"],
                       "rejects": ["wall", "floor", "wheel"]}},
       verified=True)
```

`roles` says what a slot may receive: `queries` are example sentences compared with the part captions, `rejects` are
words a candidate's caption must not contain. Without roles, a slot is matched by its own name.

## 3. Build the chain

**Automatic replacement chain.** Every turn replaces one slot with a retrieved part; no decisions are needed:

```bash
cd part_assembly
python chain_run.py --stage run --oid <host> --out-dir chains/<host>_s0 --n-turns 6 --seed 0
python chain_run.py --stage render --out-dir chains/<host>_s0          # chains/<host>_s0/img/turnNN/f000V.png
```

**Interactive chain (all four operations).** The engine keeps the state, the library and Blender in memory and asks for
a decision at each step; a person or an LLM agent answers by writing small text files in the chain folder:

```bash
python chain_run.py --stage serve --oid <host> --out-dir chains/<host>_s0 --n-turns 6 --ops replace,add,remove,retexture
```

Progress is appended to `serve_status.jsonl`; every event points to the image to look at.

| Event | The engine shows | Answer |
|---|---|---|
| `probe` (once) | each slot with its old part removed and the contact points marked | `mount_types.json`: `{"<slot>": "plane" \| "axis" \| "point" \| "skip"}` |
| `faces` (each turn) | the free mounting faces (numbered, with landing spots) and the removable slots | `op_cmd`: `add <face#> "<what to add>" spot=<n> size=<0..1>`, `retexture <slot> "<new material>"`, `remove <slot>`, `replace` or `stop` |
| `candidates` | the retrieved parts for the slot, numbered, next to the current object | `cand_cmd`, one line per candidate: `<#> fit <bottom\|top\|side\|end\|rim\|xpos\|xneg\|ypos\|yneg\|zpos\|zneg>` or `<#> skip <reason>` |
| `ready` | the four views after the turn | `serve_cmd`: `ok`, `veto` (try another candidate), `yaw <degrees>` (same part, rotated), `unveto`, `stop` |

A stopped or crashed chain resumes from its last finished turn when the same command is run again.

**Assembly chains.** Chains that start from a host's core and add its own parts back have their own pipeline:
`part_assembly/annotate/README.md`.

## 4. Finalise and verify

```bash
python part_assembly/tools/finalize_chain.py chains/<host>_s0
# chains/<host>_s0: 6 turns; rebuild from the recipe: PASS chains/<host>_s0 (verified vs checks.json)
```

This writes `turn00_manifest.json`, `checks.json` and the baked parts of retexture turns (`retex/`), then rebuilds the
chain from the recipe alone and checks every state against the engine's output.

## 5. Write the instructions

Each record of `place_report.json` carries the engine's template sentence in `instruction`. Rewrite it into the final
instruction (and `instruction_zh`) with the rules of `docs/INSTRUCTIONS.md`, keeping the template as
`instruction_original`. Instructions do not enter `checks.json`, so the chain still verifies after the rewrite.
