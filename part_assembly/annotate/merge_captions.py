"""Merge the per-object captions of part_captions.py into the library files the engine reads, and encode the object captions.

Input: WORK/<oid>/captions.json = {"object": [brief, detailed], "parts": {"<pid>": [brief, detailed], ...}}.
Output, in the part library root (PXFORM_LIBRARY_ROOT):
    captions_qwen3.json        {oid: {object: [...], parts: {...}}}    full structure (the file released with the dataset)
    text_captions_qwen3.json   {oid: {pid: [...]}}                     what retrieval reads (build_replace_chains.captions_path)
    objects_qwen3.json         {oid: [brief, detailed]}                object captions
    obj_vecs.npy, obj_meta.json                                        object-caption embeddings (unless --merge-only)
Afterwards run `python embed_library.py` to rebuild the part embeddings from text_captions_qwen3.json.

    python merge_captions.py --work WORK [--merge-only]
The released dataset already contains captions_qwen3.json; to start from it instead of captioning again:
    python merge_captions.py --from-release EditHero/data/captions/captions_qwen3.json
"""
import os, sys, json, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from local_paths import DATA_ROOT as D

OUT_FULL = os.path.join(D, "captions_qwen3.json")
OUT_PARTS = os.path.join(D, "text_captions_qwen3.json")
OUT_OBJS = os.path.join(D, "objects_qwen3.json")
OBJ_VECS = os.path.join(D, "obj_vecs.npy")
OBJ_META = os.path.join(D, "obj_meta.json")


def merge(work):
    full, bad = {}, 0
    for p in glob.glob(os.path.join(work, "*", "captions.json")):
        oid = os.path.basename(os.path.dirname(p))
        try:
            d = json.load(open(p))
        except Exception:
            bad += 1
            continue
        if d.get("parts"):
            full[oid] = dict(object=d.get("object"), parts=d["parts"])
    return full, bad


def split(full):
    parts = {o: v["parts"] for o, v in full.items()}
    objs = {o: v["object"] for o, v in full.items() if v.get("object")}
    return parts, objs


def encode_objects(objs, device="auto"):
    """Object captions are embedded separately from the part captions (different retrieval granularity)."""
    from embed_library import encoder
    oids = sorted(objs)
    texts = [(objs[o][0] if isinstance(objs[o], list) else str(objs[o])).strip().rstrip(".") for o in oids]
    vec = encoder(device)(texts)
    np.save(OBJ_VECS, vec.astype(np.float32))
    json.dump([dict(oid=o, cap=t) for o, t in zip(oids, texts)], open(OBJ_META, "w"), ensure_ascii=False)
    return vec.shape


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--work", help="output folder of part_captions.py")
    g.add_argument("--from-release", help="captions_qwen3.json of the released dataset")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if a.work:
        full, bad = merge(a.work)
    else:
        full, bad = json.load(open(a.from_release)), 0
    parts, objs = split(full)
    print(f"{len(full)} objects / {sum(len(v) for v in parts.values())} parts; {len(objs)} object captions"
          + (f"; {bad} unreadable files" if bad else ""), flush=True)
    os.makedirs(D, exist_ok=True)
    for p, obj in ((OUT_FULL, full), (OUT_PARTS, parts), (OUT_OBJS, objs)):
        json.dump(obj, open(p, "w"), ensure_ascii=False)
        print(f"  {p}  {os.path.getsize(p) / 2 ** 20:.1f} MB")
    if a.merge_only:
        return
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("encoding object captions ...", flush=True)
    print("  ", encode_objects(objs, a.device))
    print("next: python embed_library.py (rebuilds the part embeddings from text_captions_qwen3.json)")


if __name__ == "__main__":
    main()
