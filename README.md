<h1 align="center">EditHero: A Benchmark for Long-Horizon Part-Level 3D Editing and Vibe Modeling</h1>

<!-- <p align="center">
  <a href="https://auroraryan0301.github.io/">Ruihan Yu</a><sup>1,2*</sup>,
  <a href="https://liagm.github.io/">Yu-Ju Tsai</a><sup>3,4*</sup>,
  <a href="https://myniuuu.github.io/">Muyao Niu</a><sup>1,2</sup>,
  <a href="https://lirunyi2001.github.io/">Runyi Li</a><sup>1,2</sup>,
  <a href="https://scholar.google.com/citations?user=CbWYuIEAAAAJ">Lian Fu</a><sup>1,2</sup>,
  <a href="https://openreview.net/profile?id=%7ELiu_Hanqing3">Hanqing Liu</a><sup>2</sup>,
  <a href="https://brian90709.github.io/">Zheng-Hui Huang</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=lwuwuAYAAAAJ&amp;hl=zh-CN">Yonghao Yu</a><sup>1</sup>,
  <a href="https://shokuno5.github.io/">Sho Kuno</a><sup>1,2</sup>,
  <a href="https://faculty.ucmerced.edu/mhyang/">Ming-Hsuan Yang</a><sup>4&dagger;</sup>,
  <a href="https://kpzhang93.github.io/">Kaipeng Zhang</a><sup>1&dagger;</sup>,
  <a href="https://lightchaserx.github.io/">Zhixiang Wang</a><sup>1&dagger;</sup>
</p>

<p align="center">
  <sup>1</sup><a href="https://alayalab.ai/">Alaya Lab</a> &nbsp;&nbsp;
  <sup>2</sup>The University of Tokyo &nbsp;&nbsp;
  <sup>3</sup>Institute of Science Tokyo &nbsp;&nbsp;
  <sup>4</sup>University of California, Merced
</p> -->

<!-- <p align="center"><sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>&dagger;</sup>Corresponding authors</p> -->

<p align="center">
  <a href="https://arxiv.org/pdf/2610.02298"><img src="https://img.shields.io/badge/arXiv-2610.02298-b31b1b"></a>
  <a href="https://alaya-lab.github.io/EditHero/"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
  <a href="https://huggingface.co/datasets/AlayaLab/EditHero"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow"></a>
</p>

<p align="center">
  <img src="assets/teaser.png" width="100%" alt="EditHero teaser">
</p>

## 📰 News

- **[2026-10-08]** Data engine and the full dataset released ([Hugging Face](https://huggingface.co/datasets/AlayaLab/EditHero)).
- **[2026-10-01]** Paper released on [arXiv](https://arxiv.org/pdf/2610.02298).
- **[2026-09-29]** [Project page](https://alaya-lab.github.io/EditHero/) and trailer released.

## 🚀 Release Roadmap

- [x] Project page
- [x] Paper
- [x] Data engine, rebuild and render script
- [x] Annotation pipelines and prompts (part captions, assembly chains), part captions on Hugging Face
- [x] Edit chains on Hugging Face (states, recipes, parts, four-view renders, part masks)

## 📦 Data

457 edit chains, 2,755 edits and 3,212 states on 252 host objects, at
[huggingface.co/datasets/AlayaLab/EditHero](https://huggingface.co/datasets/AlayaLab/EditHero):

```bash
huggingface-cli download AlayaLab/EditHero --repo-type dataset --local-dir EditHero
```

Every chain comes with its instructions, the exact state after every turn (GLB), the recipe that rebuilds it, every part it uses,
renders of every state from a fixed four-view rig (view 0 is the conditioning view, views 1-3 are held out) and 2D masks of the
parts edited in each turn. See the dataset card for the layout and the licenses.

## 🛠️ Code

| Path | Contents |
|---|---|
| `part_assembly/` | Data engine. `chain_run.py` builds chains (slot selection, retrieval, placement and contact checks); `embed_*.py` part retrieval; `restyle_serve.py` + `tex_serve.py` retexturing (Qwen-Image-Edit-2509 restyle, TRELLIS.2 texturing into a non-overlapping UV, kept regions carried over); `blender_serve.py` resident render worker |
| `part_assembly/tools/assemble.py` | Rebuilds every state of a chain from its recipe and the parts, and verifies it against `checks.json` |
| `part_assembly/annotate/` | Annotation with vision-language models and LLM agents: part captions of PartVerse-XL (Qwen3-VL-32B) and HY3D-Bench (Qwen3-VL-8B); the assembly chains of family A (host screening, LLM-agent assembly plans, Qwen draft instructions, agent revision, left/right check, relational rewrite, Qwen check), with every prompt as used. See `part_assembly/annotate/README.md` |
| `docs/INSTRUCTIONS.md` | How the released instructions were written from the engine's template sentences |
| `scripts/render_state.py` | Renders a state (or your own GLB) with a chain's fixed four-view rig, exactly like the renders in the dataset |
| `jobs/rebuild_all.sh` | Rebuilds and verifies every chain of the dataset |

Paths come from environment variables with defaults relative to this repository (`part_assembly/local_paths.py`):
`PXFORM_BLENDER` (Blender 4.2), `PXFORM_RENDER_SCRIPT` (blender_kit's `scripts/render.py`), `PXFORM_LIBRARY_ROOT` (part library
for building new chains), `PXFORM_ENCODER_PYTHON` (TRELLIS.2 environment).

### Captions and annotation

The engine retrieves parts by their captions. The captions are in the dataset; to set up retrieval from them:

```bash
cd part_assembly/annotate
python merge_captions.py --from-release EditHero/data/captions/captions_qwen3.json   # PartVerse-XL, writes the library files
python ingest_hy3d.py --from-release EditHero/data/captions/hy3d_captions.json      # HY3D-Bench parts
python ../embed_library.py                                                           # part embeddings for retrieval
```

`part_assembly/annotate/README.md` describes how the captions were made and every step of the assembly chains
(scripts, prompts, models), and the dataset holds each intermediate file of the released assembly chains
(`data/annotations/assembly/`).

### Rebuild the chains

```bash
bash jobs/rebuild_all.sh EditHero/data out/
# PASS A/1a3267df_s0 (verified vs checks.json) ...
```

### Render a state

```bash
git clone https://github.com/AuroraRyan0301/Blender-Visualization-Skill third_party/blender_kit
python scripts/render_state.py --data EditHero/data --chain A/1a3267df_s0 --turn 1 --out out/A_1a3267df_s0_turn01
# out/.../f0000.png (cond view), f0001-f0003.png (held-out views); pass --glb your_result.glb to render your own edit
```

All renders in EditHero are made with [blender_kit](https://github.com/AuroraRyan0301/Blender-Visualization-Skill), a small
Blender/Cycles toolkit for checking generation and reconstruction results by eye. If it saves you an afternoon of arguing with
Blender, a ⭐ on it would make its author's day.

## 📄 License

Code: MIT (`LICENSE`); third-party components: `THIRD_PARTY_LICENSES.md`. Data: see the dataset card (our annotations CC BY 4.0;
meshes, textures and renders keep the license of their source objects).

## BibTeX
```bibtex
@misc{yu2026editherobenchmarklonghorizonpartlevel,
      title={EditHero: A Benchmark for Long-Horizon Part-Level 3D Editing and Vibe Modeling},
      author={Ruihan Yu and Yu-Ju Tsai and Muyao Niu and Runyi Li and Lian Fu and Hanqing Liu and Zheng-Hui Huang and Yonghao Yu and Sho Kuno and Ming-Hsuan Yang and Kaipeng Zhang and Zhixiang Wang},
      year={2026},
      eprint={2610.02298},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.02298},
}
```
