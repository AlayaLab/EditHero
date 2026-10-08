# Annotation: part captions and assembly chains

Two parts of the data engine that use vision-language models and LLM agents. Both read and write the same recipe format
as the rest of the engine, so `tools/assemble.py` rebuilds and verifies their chains like any other.

## Part captions (`part_captions.py`, `merge_captions.py`)

Every part of the PartVerse-XL library has a short and a detailed English caption, and every object an object caption.
They drive part retrieval (`embed_library.py`, `embed_retrieve.py`) and the template instructions of the engine.

| Step | Command | Output |
|---|---|---|
| Render | `python part_captions.py --stage render --objects oids.txt --work WORK [--rank R --world W]` | per object: 8 textured views, the best view and 2D box of every part (`select.json`), every part rendered alone |
| Caption | `python part_captions.py --stage caption --objects oids.txt --work WORK` (Qwen3-VL-32B-Instruct) | `WORK/<oid>/captions.json` |
| Merge | `python merge_captions.py --work WORK` | `captions_qwen3.json`, `text_captions_qwen3.json`, `objects_qwen3.json` in the library root, object-caption embeddings |
| Embed | `python ../embed_library.py` | part-caption embeddings for retrieval |

The model sees, per part, the whole object with a red box around the part next to the part rendered alone, the layout
and the Chinese prompt of the FullPart authors with two added rules (describe only the part itself; always name what the
part is). The object caption is generated first and given to the part prompt as context. The prompts are in
`part_captions.py` exactly as used. The released dataset contains the result (`data/captions/captions_qwen3.json`), so
`python merge_captions.py --from-release EditHero/data/captions/captions_qwen3.json` sets up retrieval without captioning again.

### HY3D-Bench parts (`hy3d_captions.py`, `ingest_hy3d.py`)

The HY3D-Bench parts get one caption each in the same brief style, from Qwen3-VL-8B-Instruct: per part a strip of
three panels (the part alone from its two most visible of the 42 HY3D-Bench views, and the whole object cropped to its
silhouette with the part boxed in red). Inputs are the part-segmented HY3D-Bench meshes (`PXFORM_HY3D_MESH_ROOT`) and
their renders with part masks (`PXFORM_HY3D_COND_ROOT`).

    python hy3d_captions.py --list oids.txt --out-dir WORK --tag full0
    python ingest_hy3d.py --annot-dir WORK                 # or --from-release EditHero/data/captions/hy3d_captions.json

## Assembly chains (`assembly/`)

Chains that start from a host object's core and add the host's own parts back, one to three per turn, each with a
natural-language instruction. In the dataset they are part of family A (additions). Folders are set in `assembly/paths.py`
(`PXFORM_LIBRARY_ROOT`, `PXFORM_ASSEMBLY_WORK`, `PXFORM_ASSEMBLY_HOSTS`); run the scripts from `assembly/`.

| Step | Script / prompt | Model | Output in the working folder |
|---|---|---|---|
| 1 Hosts | `screen_hosts.py`, `host_colour_stats.py`, `select_hosts.py`, `preview_hosts.py --concat` / `--render` | | `candidates.json`, `hosts.txt`, `preview/img/<oid>/` (hosts that do not look like appealing, colourful game-style assets are then removed by eye) |
| 2 Planner input | `export_parts.py` | | `hosts/<oid>/parts.json` |
| 3 Plan | `prompts/plan_assembly.md`, then `check_plan.py <oid>` | LLM agent (Claude) | `plans/<oid>.json`: start state and steps |
| 4 Recipe | `build_recipe.py <oid>` or `build_all.sh` | | `chains/<oid>/`: recipe, `turnNN.glb`, `img/` (template instructions) |
| 5 Brief | `write_briefs.py` | | `chains/<oid>/brief.json` |
| 6 Draft | `qwen_draft.py` | Qwen3.8-27B | `instructions_qwen.json` (+ raw output) |
| 7 Revision | `prompts/revise_instructions.md` | LLM agent (Claude Haiku 4.5) | `instructions_final.json`: front, object's own left/right, part identity checked against coordinates and renders |
| 8 Left/right | `lr_audit.py`, `lr_fix.py` | | `instructions_checked.json`: side words set from coordinates where draft and revision agree on the front |
| 9 Relational wording | `prompts/relational_rewrite.md` | LLM agent (Claude Haiku 4.5) | `instructions_relational.json`: placement by relations to existing parts, no axis words, left/right only when nothing else works |
| 10 Check | `qwen_check.py` | Qwen3.8-27B | `instructions_check.json`: per turn ok / issues / proposed sentence |
| 11 Apply | `apply_instructions.py [--decisions decisions.json]` | | the final instruction in every record of `place_report.json`; `instruction_original` keeps the template |

Turns flagged in step 10 were decided by a person (keep the sentence, take the proposal, or rewrite it). Every
intermediate file of the released chains (plan, brief, draft, revision, checked, relational, check) is in the dataset
under `data/annotations/assembly/`.

## Instructions of the other families

The engine writes a template instruction for every turn (`chain_run.py`, `build_replace_chains.py`). The released
instructions of all families were rewritten from these templates with the rules in `docs/INSTRUCTIONS.md` and reviewed by
us; the template is kept in each record as `instruction_original`.
