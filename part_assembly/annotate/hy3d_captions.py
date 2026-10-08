#!/usr/bin/env python
"""Part captions of the HY3D-Bench parts (the HY3D parts used by the chains), in the brief-caption style of the
PartVerse-XL captions. Inputs: the part-segmented HY3D-Bench meshes (PXFORM_HY3D_MESH_ROOT/<oid>/mesh.glb + meta.json)
and their 42-view renders with per-pixel part masks (PXFORM_HY3D_COND_ROOT/<oid>.npz). Then ingest the captions into the
retrieval library with ../ingest_hy3d.py, or use data/captions/hy3d_captions.json of the dataset.

    python hy3d_captions.py --list oids.txt --out-dir WORK --tag full0 [--gpu 0 --batch 16]


For every part of every object we build a 3-panel strip
(part from its two best viewpoints + the whole object for context),
caption it with Qwen3-VL-8B-Instruct in the PartVerseXL brief-caption
style, and dump the part AABB / face count next to it.

Outputs (append-only, resumable):
  captions_pilot.jsonl    {"oid","pid","cap"}
  parts_meta_pilot.jsonl  {"oid","pid","bbox":[[min],[max]],"n_faces"}
  done_oids.txt           one oid per finished object
  stats_pilot.json        counters + throughput

Part ids are the geometry index k, which is simultaneously
  - the value stored in the render masks (<oid>.npz, NNN_mask.npy),
  - the index into meta.json["geom_order"] (g00000..),
  - the cluster "part_k" of split_mesh.json.
That identity was verified by reprojecting geometry centroids into the
masks.
"""
import argparse
import io
import json
import os
import queue
import sys
import threading
import time
import traceback

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glb_bbox import geom_bboxes  # noqa: E402

MESH_ROOT = os.environ.get("PXFORM_HY3D_MESH_ROOT", "./library/hy3d/meshes")
COND_ROOT = os.environ.get("PXFORM_HY3D_COND_ROOT", "./library/hy3d/cond")
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
N_VIEWS = 42

PROMPT = """The image shows three panels of the SAME 3D object.
Panel 1 and panel 2 show ONE isolated part of that object, rendered from two viewpoints on a white background.
Panel 3 shows the complete object with that same part outlined by a red box, for context only.

Write ONE English sentence describing the isolated part shown in panels 1 and 2.

Rules:
- Begin with what the part is, including its colour, material and shape.
- Then give its role with "serving as ..." or "... of the <whole object>", naming the whole object from panel 3.
- Exactly one sentence, at most 40 words.
- Never mention panels, viewpoints, images, renders, backgrounds, red boxes, "left"/"right".
- Describe only what is visible. Do not invent text, logos, branding or hidden details.

Two examples of the required style:
"A black rubber tire with a gold-colored rim, serving as a wheel for the toy race car."
"The green pointed hat of the low-poly 3D character model, serving as the headwear component."

Reply with the sentence only."""


# --------------------------------------------------------------------------
# image building
# --------------------------------------------------------------------------
def _decode_rgb(npz, view):
    """Decode NNN.webp (RGBA) onto a white background -> uint8 HxWx3."""
    im = Image.open(io.BytesIO(npz["%03d.webp" % view].tobytes()))
    if im.mode != "RGBA":
        return np.asarray(im.convert("RGB"))
    a = np.asarray(im, dtype=np.float32)
    alpha = a[..., 3:4] / 255.0
    rgb = a[..., :3] * alpha + 255.0 * (1.0 - alpha)
    return rgb.astype(np.uint8)


def _square_pad_resize(arr, size):
    h, w = arr.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s, 3), 255, np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = arr
    return np.asarray(Image.fromarray(canvas).resize((size, size), Image.LANCZOS))


def _part_panel(rgb, mask, pid, size, margin=0.12):
    sel = mask == pid
    ys, xs = np.nonzero(sel)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = int(round(margin * max(y1 - y0, x1 - x0))) + 4
    H, W = mask.shape
    y0, y1 = max(0, y0 - m), min(H, y1 + m)
    x0, x1 = max(0, x0 - m), min(W, x1 + m)
    out = np.full_like(rgb, 255)
    out[sel] = rgb[sel]
    return _square_pad_resize(out[y0:y1, x0:x1], size)


def _context_panel(rgb, mask, pid, size, box_w=3):
    """Whole object cropped to its silhouette, with the part outlined in red."""
    fg = mask >= 0
    ys, xs = np.nonzero(fg)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = int(round(0.06 * max(y1 - y0, x1 - x0))) + 4
    H, W = mask.shape
    y0, y1 = max(0, y0 - m), min(H, y1 + m)
    x0, x1 = max(0, x0 - m), min(W, x1 + m)
    img = rgb.copy()
    pys, pxs = np.nonzero(mask == pid)
    if len(pxs):
        b = 3
        py0, py1 = max(0, pys.min() - b), min(H, pys.max() + 1 + b)
        px0, px1 = max(0, pxs.min() - b), min(W, pxs.max() + 1 + b)
        red = np.array([220, 30, 30], np.uint8)
        img[py0:py0 + box_w, px0:px1] = red
        img[max(py0, py1 - box_w):py1, px0:px1] = red
        img[py0:py1, px0:px0 + box_w] = red
        img[py0:py1, max(px0, px1 - box_w):px1] = red
    return _square_pad_resize(img[y0:y1, x0:x1], size)


def _strip(panels, gap=6):
    h = panels[0].shape[0]
    parts = []
    for i, p in enumerate(panels):
        if i:
            parts.append(np.full((h, gap, 3), 210, np.uint8))
        parts.append(p)
    return Image.fromarray(np.concatenate(parts, axis=1))


# --------------------------------------------------------------------------
# per-object CPU preparation
# --------------------------------------------------------------------------
def prepare_object(oid, min_area, panel, max_parts):
    """-> dict(oid=..., items=[{pid,image,bbox,n_faces}], stats={...}) or raises."""
    st = dict(n_parts_meta=0, n_visible=0, n_tiny=0, n_invisible=0, n_nobbox=0)
    meta_p = os.path.join(MESH_ROOT, oid, "meta.json")
    glb_p = os.path.join(MESH_ROOT, oid, "mesh.glb")
    npz_p = os.path.join(COND_ROOT, oid + ".npz")
    meta = json.load(open(meta_p))
    geom_order = meta["geom_order"]
    st["n_parts_meta"] = len(geom_order)

    with np.load(npz_p) as npz:
        masks = [npz["%03d_mask.npy" % v] for v in range(N_VIEWS)]
        nb = len(geom_order)
        area = np.zeros((nb, N_VIEWS), np.int64)
        for v, m in enumerate(masks):
            ids, cnt = np.unique(m, return_counts=True)
            keep = (ids >= 0) & (ids < nb)
            area[ids[keep], v] = cnt[keep]
        if area.sum() == 0:
            raise RuntimeError("empty_mask")

        best = area.max(1)
        order = np.argsort(-best)
        pids = []
        for pid in order:
            if best[pid] == 0:
                st["n_invisible"] += 1
            elif best[pid] < min_area:
                st["n_tiny"] += 1
            else:
                pids.append(int(pid))
        st["n_visible"] = len(pids)
        if max_parts and len(pids) > max_parts:
            pids = pids[:max_parts]

        # decode only the views we actually need
        need = set()
        top2 = {}
        for pid in pids:
            v = np.argsort(-area[pid])[:2]
            v = [int(x) for x in v if area[pid, x] > 0]
            if len(v) == 1:
                v = v * 2
            top2[pid] = v
            need.update(v)
        rgb = {v: _decode_rgb(npz, v) for v in sorted(need)}

    boxes = geom_bboxes(glb_p)
    items = []
    for pid in pids:
        name = geom_order[pid]
        b = boxes.get(name)
        if b is None:
            st["n_nobbox"] += 1
            continue
        v1, v2 = top2[pid]
        panels = [_part_panel(rgb[v1], masks[v1], pid, panel),
                  _part_panel(rgb[v2], masks[v2], pid, panel),
                  _context_panel(rgb[v1], masks[v1], pid, panel)]
        items.append(dict(pid=str(pid), image=_strip(panels),
                          bbox=b["bbox"], n_faces=b["n_faces"]))
    return dict(oid=oid, items=items, stats=st)


# --------------------------------------------------------------------------
# caption post-processing
# --------------------------------------------------------------------------
def clean_caption(text):
    t = " ".join(text.strip().split())
    for pre in ('"', "'", "*"):
        t = t.strip(pre).strip()
    if t.lower().startswith("sentence:"):
        t = t[9:].strip()
    # keep the first sentence only
    cut = -1
    for i in range(len(t) - 1):
        if t[i] == "." and (t[i + 1] == " " or i == len(t) - 2):
            cut = i + 1
            break
    if cut > 0:
        t = t[:cut]
    if t and not t.endswith("."):
        t += "."
    return t


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="pilot_oids.txt")
    ap.add_argument("--out-dir", default="./work/hy3d_captions")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--panel", type=int, default=384)
    ap.add_argument("--min-area", type=int, default=400,
                    help="skip a part whose largest mask over all 42 views is smaller than this (px)")
    ap.add_argument("--max-parts", type=int, default=0, help="0 = no cap")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--prefetch", type=int, default=6, help="CPU prep worker threads")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump-strips", type=int, default=0,
                    help="save this many composed strips to <out>/strips_<tag>/ for eyeballing")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    cap_p = os.path.join(out, "captions_%s.jsonl" % args.tag)
    met_p = os.path.join(out, "parts_meta_%s.jsonl" % args.tag)
    don_p = os.path.join(out, "done_oids_%s.txt" % args.tag)
    err_p = os.path.join(out, "errors_%s.jsonl" % args.tag)
    stat_p = os.path.join(out, "stats_%s.json" % args.tag)

    done = set()
    if os.path.exists(don_p):
        done = {l.strip() for l in open(don_p) if l.strip()}
    oids = [l.strip() for l in open(args.list) if l.strip()]
    todo = [o for o in oids if o not in done]
    if args.limit:
        todo = todo[:args.limit]
    print("[info] %d oids, %d already done, %d to do" % (len(oids), len(done), len(todo)), flush=True)
    if not todo:
        return

    strip_dir = os.path.join(out, "strips_%s" % args.tag)
    if args.dump_strips:
        os.makedirs(strip_dir, exist_ok=True)

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    if proc.tokenizer.padding_side != "left":
        proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    print("[info] model loaded in %.1fs" % (time.time() - t0), flush=True)

    # ---- CPU prep threads feeding a bounded queue -------------------------
    q = queue.Queue(maxsize=args.prefetch * 2)
    idx = [0]
    lk = threading.Lock()

    def worker():
        while True:
            with lk:
                i = idx[0]
                idx[0] += 1
            if i >= len(todo):
                q.put(None)
                return
            oid = todo[i]
            try:
                q.put(prepare_object(oid, args.min_area, args.panel, args.max_parts))
            except Exception as e:
                q.put(dict(oid=oid, error="%s: %s" % (type(e).__name__, e),
                           trace=traceback.format_exc()[-800:]))

    nthr = max(1, args.prefetch)
    for _ in range(nthr):
        threading.Thread(target=worker, daemon=True).start()

    S = dict(objects=0, objects_failed=0, parts=0, parts_captioned=0,
             parts_tiny=0, parts_invisible=0, parts_nobbox=0, parts_failed=0,
             gpu_seconds=0.0)
    fail_kinds = {}
    fcap = open(cap_p, "a")
    fmet = open(met_p, "a")
    fdon = open(don_p, "a")
    ferr = open(err_p, "a")
    dumped = 0
    finished = 0
    t_start = time.time()

    while finished < nthr:
        job = q.get()
        if job is None:
            finished += 1
            continue
        oid = job["oid"]
        if "error" in job:
            S["objects_failed"] += 1
            kind = job["error"].split(":")[0] if "RuntimeError" not in job["error"] else job["error"].split(": ")[-1]
            fail_kinds[kind] = fail_kinds.get(kind, 0) + 1
            ferr.write(json.dumps(dict(oid=oid, error=job["error"])) + "\n")
            ferr.flush()
            fdon.write(oid + "\n")
            fdon.flush()
            continue

        st = job["stats"]
        S["parts_tiny"] += st["n_tiny"]
        S["parts_invisible"] += st["n_invisible"]
        S["parts_nobbox"] += st["n_nobbox"]
        items = job["items"]
        S["parts"] += len(items)

        caps = []
        for b0 in range(0, len(items), args.batch):
            chunk = items[b0:b0 + args.batch]
            msgs = [[{"role": "user", "content": [
                {"type": "image", "image": it["image"]},
                {"type": "text", "text": PROMPT}]}] for it in chunk]
            tg = time.time()
            try:
                inputs = proc.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt", padding=True).to("cuda:0")
                with torch.inference_mode():
                    o = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                       do_sample=False)
                txt = proc.batch_decode(o[:, inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                txt = []
                for it in chunk:  # retry one by one
                    try:
                        inp = proc.apply_chat_template(
                            [[{"role": "user", "content": [
                                {"type": "image", "image": it["image"]},
                                {"type": "text", "text": PROMPT}]}]],
                            tokenize=True, add_generation_prompt=True,
                            return_dict=True, return_tensors="pt", padding=True).to("cuda:0")
                        with torch.inference_mode():
                            oo = model.generate(**inp, max_new_tokens=args.max_new_tokens,
                                                do_sample=False)
                        txt.append(proc.batch_decode(
                            oo[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0])
                    except Exception:
                        txt.append("")
            S["gpu_seconds"] += time.time() - tg
            caps.extend(txt)

        for it, raw in zip(items, caps):
            c = clean_caption(raw)
            if len(c) < 12:
                S["parts_failed"] += 1
                fail_kinds["empty_caption"] = fail_kinds.get("empty_caption", 0) + 1
                continue
            S["parts_captioned"] += 1
            fcap.write(json.dumps(dict(oid=oid, pid=it["pid"], cap=c)) + "\n")
            fmet.write(json.dumps(dict(oid=oid, pid=it["pid"],
                                       bbox=it["bbox"], n_faces=it["n_faces"])) + "\n")
            if dumped < args.dump_strips:
                it["image"].save(os.path.join(strip_dir, "%s_%s.jpg" % (oid, it["pid"])), quality=88)
                dumped += 1

        fcap.flush()
        fmet.flush()
        fdon.write(oid + "\n")
        fdon.flush()
        S["objects"] += 1
        if S["objects"] % 10 == 0:
            el = time.time() - t_start
            print("[prog] obj %d/%d parts %d  %.2fs/part  %.1fs/obj  elapsed %.0fs"
                  % (S["objects"], len(todo), S["parts_captioned"],
                     el / max(1, S["parts_captioned"]), el / S["objects"], el), flush=True)
            S["wall_seconds"] = el
            S["fail_kinds"] = fail_kinds
            json.dump(S, open(stat_p, "w"), indent=1)

    S["wall_seconds"] = time.time() - t_start
    S["fail_kinds"] = fail_kinds
    S["sec_per_part"] = S["wall_seconds"] / max(1, S["parts_captioned"])
    S["sec_per_object"] = S["wall_seconds"] / max(1, S["objects"])
    json.dump(S, open(stat_p, "w"), indent=1)
    print("[done]", json.dumps(S), flush=True)


if __name__ == "__main__":
    main()
