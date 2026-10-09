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
- [x] Dataset on Hugging Face (states, recipes, parts, four-view renders, part masks, part captions)
- [x] Data engine: chain building, rebuild, rendering, annotation pipelines and prompts

## 📦 Data

457 edit chains, 2,755 edits and 3,212 states on 252 host objects, at
[huggingface.co/datasets/AlayaLab/EditHero](https://huggingface.co/datasets/AlayaLab/EditHero). Every chain has its
instructions, the exact state after every turn (GLB), the recipe that rebuilds it, renders of every state from a fixed
four-view rig (view 0 is the conditioning view, views 1-3 are held out) and 2D masks of the parts edited in each turn.
The dataset card explains the layout and how to download only what you need.

## 🛠️ The data engine

A chain starts from a textured, part-segmented host object; every turn adds, removes or replaces a part, or gives a part
a new material. New parts are retrieved from a part library by their captions and placed on the host's contact surfaces
with five geometric checks; new materials are restyled in an image and re-textured in 3D. The result is a recipe from
which every state is rebuilt and verified exactly.

| Path | Contents |
|---|---|
| `part_assembly/chain_run.py` | Builds a chain: host slots, retrieval, placement and checks, retexturing, rendering (`--stage run / serve / render`) |
| `part_assembly/embed_library.py`, `embed_retrieve.py`, `build_replace_chains.py` | Part library embeddings, retrieval, caption handling |
| `part_assembly/restyle_serve.py`, `tex_serve.py`, `blender_serve.py` | Workers for restyling (Qwen-Image-Edit-2509), 3D texturing (TRELLIS.2) and rendering (Blender) |
| `part_assembly/slot_registry.py`, `contact_graph.py`, `change_meter.py` | Host slots, contact graph, visible-change measure |
| `part_assembly/tools/` | `assemble.py` rebuilds and verifies a chain from its recipe; `finalize_chain.py` turns an engine run into a recipe; `registry_from_manifest.py` takes a dataset host's slots |
| `part_assembly/annotate/` | Part captions (PartVerse-XL, HY3D-Bench) and the assembly chains of family A, with every prompt ([README](part_assembly/annotate/README.md)) |
| `scripts/render_state.py` | Renders a state, or your own GLB, with a chain's fixed four-view rig |
| `jobs/rebuild_all.sh` | Rebuilds and verifies every chain of the dataset |
| `docs/DATA_ENGINE.md` | How to build a new chain, step by step |
| `docs/INSTRUCTIONS.md` | How the instructions are written |

All paths are environment variables (`part_assembly/local_paths.py`).

## ⚡ Quick start

```bash
# data
huggingface-cli download AlayaLab/EditHero --repo-type dataset --local-dir EditHero
cd EditHero && for t in data/*/*.tar; do tar -xf "$t" && rm "$t"; done && cd ..

# rebuild every chain from its recipe and verify it
bash jobs/rebuild_all.sh EditHero/data out/
# PASS A/1a3267df_s0 (verified vs checks.json) ...

# render a state with its chain's rig (Blender 4.2 + blender_kit)
git clone https://github.com/AuroraRyan0301/Blender-Visualization-Skill third_party/blender_kit
python scripts/render_state.py --data EditHero/data --chain A/1a3267df_s0 --turn 1 --out out/A_1a3267df_s0_turn01
# f0000.png is the cond view, f0001-f0003.png the held-out views; --glb your_result.glb renders your own edit
```

## 🧩 Build a new chain

1. **Library.** Point `PXFORM_LIBRARY_ROOT` at the PartVerse-XL parts and set up retrieval from the released captions
   (`annotate/merge_captions.py --from-release`, then `embed_library.py`).
2. **Slots.** Register the host's slots: for a dataset host, `tools/registry_from_manifest.py <its turn00_manifest.json>`;
   for a new host, group its parts with `slot_registry.put`.
3. **Chain.** `chain_run.py --stage run` builds a replacement chain automatically; `chain_run.py --stage serve` builds a
   chain with all four operations, asking a person or an LLM agent for each decision through small text files.
4. **Recipe.** `tools/finalize_chain.py` writes the manifest and checks and verifies the rebuild; then write the
   instructions with `docs/INSTRUCTIONS.md`.

The full walk-through, with the settings, the slot format and the decision protocol, is in
[docs/DATA_ENGINE.md](docs/DATA_ENGINE.md).

All renders in EditHero are made with [blender_kit](https://github.com/AuroraRyan0301/Blender-Visualization-Skill), a small
Blender/Cycles toolkit for checking generation and reconstruction results by eye. If it saves you an afternoon of arguing
with Blender, a ⭐ on it would make its author's day.

## 📄 License

Code: MIT (`LICENSE`); third-party components: `THIRD_PARTY_LICENSES.md`. Data: see the dataset card (our annotations
CC BY 4.0; meshes, textures and renders keep the license of their source objects).

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
