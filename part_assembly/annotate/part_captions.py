"""Part and object captions of the PartVerse-XL library with Qwen3-VL (the captions used for retrieval and instructions).

The composite image follows the FullPart authors (partverse/get_text_caption.py + get_infos.py):

  1. render the whole object from 8 views around it, together with an object-index pass (pixel = part rank + 1);
  2. for every part pick the view where it has the most visible pixels (a part hidden in all 8 views is skipped);
  3. left panel = the whole object in that view with a red box around the part's 2D bounding box;
  4. right panel = the part alone (normalized to fill the frame), same view;
  5. the two panels side by side with a 4-pixel white gap;
  6. the prompt is the authors' Chinese prompt (answer in English), with two rules added (see PART_PROMPT).

The object-level caption is generated first from a 2x2 sheet of four views and given to the part prompt as context, so
the two levels name the object the same way.

Two stages, shardable with --rank/--world, resumable (objects that already have captions.json are skipped):
    python part_captions.py --stage render  --objects oids.txt --work WORK [--rank R --world W]
    python part_captions.py --stage caption --objects oids.txt --work WORK [--rank R --world W]
Output: WORK/<oid>/captions.json = {"object": [brief, detailed], "parts": {"<pid>": [brief, detailed], ...}}.
merge_captions.py turns the per-object files into the library files the engine reads.

Paths (environment variables):
    PXFORM_PARTVERSE_ANNO     PartVerse-XL anno_infos (<oid>/<oid>_info.json, _face2label.json, _segmented.glb)
    PXFORM_LIBRARY_ROOT       part library root: pv_textured/normalized_glbs/<oid>.glb and pv_textured/textured_part_glbs/<oid>/<pid>.glb
    PXFORM_BLENDER, PXFORM_RENDER_SCRIPT   Blender 4.2 and blender_kit's scripts/render.py
"""
import os, sys, json, glob, time, argparse, subprocess, shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from local_paths import DATA_ROOT, BLENDER, RENDER_SCRIPT as KIT

ANNO = os.environ.get("PXFORM_PARTVERSE_ANNO", os.path.join(DATA_ROOT, "partverse_anno", "anno_infos"))
TEX_OBJ = os.path.join(DATA_ROOT, "pv_textured", "normalized_glbs")
TEX_PART = os.path.join(DATA_ROOT, "pv_textured", "textured_part_glbs")
KIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(KIT)))

N_VIEWS = 8
START_AZ = 35.0
ELEVATION = 30.0          # the FullPart authors use 30 (blender_kit's default is 25)
DISTANCE = 1.56           # at the default 2.5 the longest side fills 0.44 of the frame; 1.56 gives about 0.70

# The FullPart authors' Chinese prompt (USER_PROMPT_CHINESE in partverse/get_text_caption.py) with rules 3 and 4 added.
# Rule 3 (describe only the right object itself): without it Qwen3 writes things seen in the left image into the part's
# caption (an empty tray rendered on the right became "containing six red-topped blocks").
# Rule 4 (name the part): with rule 3 alone the captions degrade to pure geometry ("a beige rectangular block with a red
# top surface") and lose what the part is ("the rice base of a sushi piece"), which retrieval by meaning needs.
# The prompt text is kept exactly as used to caption the library.
_PART_PROMPT = (
    "### 假如你是一个图像标注专家，你将根据用户提供的包含左右两个物体（右物体是左物体的一个组件且被红框突出显示），"
    "来解决为右物体生成**英文**文本描述的任务。根据以下规则一步步执行：\n"
    "1. 生成一个简洁描述，仅用一句话。\n"
    "2. 生成一个详细描述，描述其颜色、材料、形状以及属于左物体的哪一部分。\n"
    "3. **只描述右物体本身实际存在的几何和材质。** 左图和红框只用来判断它在整体上的位置和作用；"
    "红框范围内出现、但并不属于右物体的东西，一律不要写进描述。"
    "比如右物体是一个空托盘时，不要写它\"装着某某东西\"；右物体是一根杆子时，不要写与它相连的其他零件。"
    "拿不准某个细节在不在右物体上，就以右图为准。\n"
    "4. **但一定要说清楚这个部件是什么、在整体上起什么作用**，用具体的部件名称来称呼它，"
    "而不是只写形状和颜色。比如写\"寿司的米饭底座\"、\"茶壶的盖子\"、\"椅子的靠背立柱\"，"
    "不要写成\"一个米色的长方块\"、\"一个圆形部件\"。第 3 条管的是不要把别的东西算进来，"
    "不是让你回避说它是什么。简洁描述里必须出现这个部件的名称。\n"
    "请回答问题：用户提供的包含左右两个物体（右物体是左物体的一个组件且被红框突出显示）的场景，为右物体生成文本描述\n"
    '输出：\n要求：\n1 按照**JSON格式**输出\n2 **JSON格式**内容为：{ "brief_caption": "", "detailed_caption": "" }\n'
    '3 **但caption中不要使用"左物体"、"右物体"这种代称，而是使用具体的物体类别名称指代**\n ###'
)


def part_prompt(obj_cap):
    """Prefix the object-level caption, so the part caption names the object the same way (without it the object level
    said "modular structure with stacked blocks" and the part level "sushi rice bases" for the same object)."""
    if not obj_cap or len(obj_cap) < 2:
        return _PART_PROMPT
    return (f"### 已知左边这个整体物体是:{obj_cap[0]}\n{obj_cap[1]}\n"
            f"请在这个前提下作答,部件的称呼要和上面对这个物体的判断一致。\n\n"
            + _PART_PROMPT)


OBJ_PROMPT = (
    "### 这张图是同一个 3D 物体的四个视角。请为这个**整体物体**生成**英文**描述:\n"
    "1. 简洁描述:一句话,说清它是什么(具体类别名称),不超过 15 个词。\n"
    "2. 详细描述:它的类别、整体形状、主要颜色和材质、由哪些主要部件构成。\n"
    "只描述图里实际有的东西,不要猜测看不见的部分。不要提\"四个视角\"、\"这张图\"之类的话。\n"
    '输出:\n1 按照**JSON格式**输出\n2 内容为:{ "brief_caption": "", "detailed_caption": "" }\n ###'
)


def obj_dir(oid):
    return os.path.join(ANNO, oid)


def write_face_ids(oid, dst):
    """Per-face part labels for blender_kit from ordered_faceid (the face indices of every part).

    A few objects (101 of 37,449) contain duplicated faces (the same triangle two or more times, often shared by several
    parts). Blender's glTF importer merges them, so the per-face labels no longer match its face count; deduplicating
    would merge those parts into one. Such objects are skipped (an exception), and no empty select.json is left behind,
    which would mark them as done."""
    d = obj_dir(oid)
    info = json.load(open(os.path.join(d, f"{oid}_info.json")))
    nf = len(json.load(open(os.path.join(d, f"{oid}_face2label.json"))))
    import trimesh
    m = trimesh.load(os.path.join(d, f"{oid}_segmented.glb"), process=False, force="mesh")
    if len(m.faces) == nf:
        key = np.sort(np.asarray(m.faces), axis=1)
        n_uniq = len(np.unique(key, axis=0))
        if n_uniq != nf:
            raise ValueError(f"{nf - n_uniq} duplicated faces; Blender merges them and the per-face labels no longer match")
    ids = np.full(nf, -1, np.int32)
    for i, faces in enumerate(info["ordered_faceid"]):
        ids[np.asarray(faces, np.int64)] = i
    np.save(dst, ids)
    return len(info["ordered_faceid"])


def run_blender(manifest, extra):
    subprocess.run([BLENDER, "-b", "--python", KIT, "--", "--manifest", manifest,
                    "--trajectory", "circle", "--elevation", str(ELEVATION),
                    "--distance", str(DISTANCE), "--continue_on_error"] + extra,
                   check=False, stdout=subprocess.DEVNULL)


# ---------------------------------------------------------------- render

def stage_render(oids, work, res, samples, chunk=25):
    """Textured whole-object views, the object-index pass (only to pick views and boxes) and textured part views.

    The _segmented.glb of anno_infos has no material, so it is used for the segmentation only; the textures come from
    normalized_glbs (whole object) and textured_part_glbs (per part). The two meshes differ in topology but their
    normalized bounding boxes coincide (IoU 1.0), so boxes from the segmentation pass apply to the textured renders."""
    sys.path.insert(0, KIT_DIR)
    from lib import exr_reader

    todo = [o for o in oids if not os.path.isfile(os.path.join(work, o, "select.json"))]   # resume
    print(f"[render] {len(oids)} objects, {len(oids) - len(todo)} done, {len(todo)} to render", flush=True)
    if chunk and len(todo) > chunk:            # batches of 25, so a lost node loses one batch at most
        t0 = time.time()
        for i in range(0, len(todo), chunk):
            stage_render(todo[i:i + chunk], work, res, samples, chunk=0)
            done = min(i + chunk, len(todo)); el = time.time() - t0
            print(f"[render] {done}/{len(todo)}  {el / 60:.1f} min, about {el / done * (len(todo) - done) / 60:.0f} min left", flush=True)
        return
    oids = todo

    # pass 1: textured whole object and segmentation, 8 views each, same cameras
    jobs, ok = [], []
    for oid in oids:
        tex = os.path.join(TEX_OBJ, oid + ".glb")
        if not os.path.isfile(tex):
            print(f"  ! {oid}: no textured whole-object glb", flush=True)
            continue
        od = os.path.join(work, oid); os.makedirs(od, exist_ok=True)
        fid = os.path.join(od, "face_ids.npy")
        try:
            n = write_face_ids(oid, fid)
            lab = json.load(open(os.path.join(obj_dir(oid), f"{oid}_info.json")))["ordered_face_label"]
        except Exception as e:
            print(f"  ! {oid}: face ids failed: {type(e).__name__}: {e}", flush=True)
            continue
        ok.append((oid, n, lab))
        jobs.append(dict(scene="mesh", mesh=tex, material="file_embedded", normalize="whole", outputs="rgb",
                         out_dir=os.path.join(od, "obj")))
        # rgb has to be requested together with mask: with mask alone blender_kit does not write the multilayer EXR
        # that carries indexob. The rgb of this pass (tab20 colours) is not used.
        jobs.append(dict(scene="parts", mesh=os.path.join(obj_dir(oid), f"{oid}_segmented.glb"), face_ids=fid,
                         select_parts="all", normalize="whole", outputs="rgb,mask", out_dir=os.path.join(od, "seg")))
    man1 = os.path.join(work, f"_man1_{os.getpid()}.jsonl")
    open(man1, "w").write("\n".join(json.dumps(j) for j in jobs))
    t = time.time()
    run_blender(man1, ["--frames", str(N_VIEWS), "--start_az", str(START_AZ), "--res", str(res), "--samples", str(samples)])
    print(f"[render] whole objects + segmentation: {len(ok)} objects, {time.time() - t:.1f}s", flush=True)

    # pick per part the view with the most visible pixels and keep its 2D box
    jobs2 = []
    for oid, n, lab in ok:
        od = os.path.join(work, oid); idx = []
        for v in range(N_VIEWS):
            p = os.path.join(od, "seg", f"f{v:04d}", "mask.exr")
            try:
                idx.append(exr_reader.read_multilayer(p, layers=("indexob",))["indexob"])
            except Exception:
                idx.append(None)
        full = sorted(glob.glob(os.path.join(od, "obj", "*.png"))); sel = {}
        for rank in range(n):
            if rank >= len(lab):
                continue
            pid = str(lab[rank])                # captions are keyed by ordered_face_label[rank], not by rank
            part_glb = os.path.join(TEX_PART, oid, pid + ".glb")
            if not os.path.isfile(part_glb):
                continue
            best_v, best_n, best_bb = -1, 0, None
            for v, m in enumerate(idx):
                if m is None:
                    continue
                hit = (np.rint(m) == rank + 1); c = int(hit.sum())
                if c > best_n:
                    ys, xs = np.nonzero(hit)
                    best_v, best_n = v, c
                    best_bb = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            if best_v < 0 or best_v >= len(full):
                continue                        # hidden in all 8 views, or a missing whole-object frame
            sel[pid] = dict(view=best_v, bbox=best_bb, pixels=best_n, rank=rank, full=os.path.basename(full[best_v]))
            jobs2.append(dict(scene="mesh", mesh=part_glb, material="file_embedded", normalize="whole", outputs="rgb",
                              start_az=START_AZ + 360.0 * best_v / N_VIEWS, out_dir=os.path.join(od, f"p{pid}")))
        json.dump(sel, open(os.path.join(od, "select.json"), "w"), indent=1)
        shutil.rmtree(os.path.join(od, "seg"), ignore_errors=True)   # ~12 MB of EXR per object; select.json keeps what is needed

    # pass 2: every part alone, textured, in its own view
    man2 = os.path.join(work, f"_man2_{os.getpid()}.jsonl")
    open(man2, "w").write("\n".join(json.dumps(j) for j in jobs2))
    t = time.time()
    run_blender(man2, ["--frames", "1", "--res", str(res), "--samples", str(samples)])
    print(f"[render] parts: {len(jobs2)}, {time.time() - t:.1f}s", flush=True)


# ---------------------------------------------------------------- caption

def _on_white(path, size):
    """Composite the RGBA render on white (the authors' images have a white background)."""
    from PIL import Image
    im = Image.open(path).convert("RGBA").resize((size, size))
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB")


def composite(full_png, part_png, bbox, size):
    """Left: the whole object with a red box around the part. Right: the part alone. 4-pixel white gap."""
    from PIL import Image, ImageDraw
    a0 = Image.open(full_png); a = _on_white(full_png, size)
    sx, sy = size / a0.width, size / a0.height
    x0, y0, x1, y1 = bbox
    ImageDraw.Draw(a).rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=(255, 0, 0), width=3)
    b = _on_white(part_png, size)
    c = Image.new("RGB", (size * 2 + 4, size), (255, 255, 255))
    c.paste(a, (0, 0)); c.paste(b, (size + 4, 0))
    return c


def object_sheet(pngs, size=320):
    """Four whole-object views as a 2x2 sheet for the object caption (one view often misidentifies the object)."""
    from PIL import Image
    idx = [0, len(pngs) // 4, len(pngs) // 2, 3 * len(pngs) // 4][:max(1, len(pngs))]
    ims = [_on_white(pngs[i], size) for i in idx if i < len(pngs)]
    while len(ims) < 4:
        ims.append(Image.new("RGB", (size, size), (255, 255, 255)))
    g = Image.new("RGB", (size * 2 + 4, size * 2 + 4), (255, 255, 255))
    for k, im in enumerate(ims[:4]):
        g.paste(im, ((k % 2) * (size + 4), (k // 2) * (size + 4)))
    return g


def parse_json(text):
    t = text.strip()
    for p in ("```json", "```"):
        if t.startswith(p):
            t = t[len(p):]
    t = t.removesuffix("```").strip()
    try:
        d = json.loads(t)
    except Exception:
        return None
    b, l = d.get("brief_caption"), d.get("detailed_caption")
    return [b, l] if b and l else None


def stage_caption(oids, work, model_id, batch, max_new, size, dump=0):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(model_id); proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"[caption] model loaded in {time.time() - t0:.1f}s", flush=True)

    def run_prompt(imgs, prompt):
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}] for im in imgs]
        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inp = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        return proc.batch_decode([o[len(i):] for i, o in zip(inp.input_ids, out)], skip_special_tokens=True)

    n_part = n_bad = 0; t0 = time.time()
    for k, oid in enumerate(oids):
        od = os.path.join(work, oid); dst = os.path.join(od, "captions.json")
        if os.path.isfile(dst):
            continue
        sp = os.path.join(od, "select.json")
        if not os.path.isfile(sp):
            continue
        sel = json.load(open(sp)); pids, imgs = [], []
        for pid, s in sorted(sel.items(), key=lambda kv: int(kv[0])):
            full = os.path.join(od, "obj", s["full"]); parts = sorted(glob.glob(os.path.join(od, f"p{pid}", "*.png")))
            if not (os.path.isfile(full) and parts):
                continue
            pids.append(pid); imgs.append(composite(full, parts[0], s["bbox"], size))
        if not imgs:
            continue
        if dump and k < dump:                    # keep a few composites to check the layout by eye
            dd = os.path.join(work, "_composites"); os.makedirs(dd, exist_ok=True)
            for pid, im in list(zip(pids, imgs))[:3]:
                im.save(os.path.join(dd, f"{oid[:8]}_p{pid}.png"))
        obj_cap = None                           # object caption first: one call on the 2x2 sheet
        try:
            pngs = sorted(glob.glob(os.path.join(od, "obj", "*.png")))
            if pngs:
                obj_cap = parse_json(run_prompt([object_sheet(pngs, 320)], OBJ_PROMPT)[0])
        except Exception as e:
            print(f"  ! {oid}: object caption failed: {type(e).__name__}: {e}", flush=True)
        outs = []
        for i in range(0, len(imgs), batch):
            outs += run_prompt(imgs[i:i + batch], part_prompt(obj_cap))
        res = {}
        for pid, o in zip(pids, outs):
            p = parse_json(o)
            if p is None:
                n_bad += 1
            else:
                res[pid] = p
        n_part += len(pids)
        tmp = dst + f".tmp{os.getpid()}"         # atomic write: workers may race on the same object
        json.dump(dict(object=obj_cap, parts=res), open(tmp, "w"), indent=1, ensure_ascii=False)
        os.replace(tmp, dst)
        if (k + 1) % 5 == 0 or k + 1 == len(oids):
            dt = time.time() - t0
            print(f"[caption] {k + 1}/{len(oids)} objects, {n_part} parts, {dt / max(n_part, 1):.2f} s/part, {n_bad} unparsable", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["render", "caption"])
    ap.add_argument("--objects", required=True, help="file with one object id per line")
    ap.add_argument("--work", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--dump", type=int, default=0, help="save the composites of the first N objects")
    a = ap.parse_args()
    oids = [l.strip() for l in open(a.objects) if l.strip()][a.rank::a.world]
    os.makedirs(a.work, exist_ok=True)
    print(f"[{a.stage}] rank {a.rank}/{a.world}: {len(oids)} objects", flush=True)
    if a.stage == "render":
        stage_render(oids, a.work, a.res, a.samples)
    else:
        stage_caption(oids, a.work, a.model, a.batch, a.max_new, a.res, a.dump)


if __name__ == "__main__":
    main()
