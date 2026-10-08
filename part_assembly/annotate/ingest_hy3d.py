"""Add HY3D-Bench part captions (hy3d_captions.py) to the retrieval library, incrementally.

Input (the output folder of hy3d_captions.py, appendable):
    captions*.jsonl     {"oid", "pid", "cap"}
    parts_meta*.jsonl   {"oid", "pid", "bbox": [[min], [max]], ...}
Output (atomic rewrite; embed_retrieve.py loads them next to the PartVerse-XL library):
    <library>/lib_meta_hy3d.json   entries {"oid", "pid", "cap", "bbox", "sig", "sym", "ds": "hy3d"}
    <library>/lib_vecs_hy3d.npy    MiniLM vectors (384-d, normalized)
Parts already in the library are not encoded again. Objects whose texture bake failed to align with the masks
(status align_fail in the bake reports) are left out, since their captions would be attached to the wrong geometry.

    python ingest_hy3d.py --annot-dir WORK [--exclude-align-fail BAKE_LOG_DIR]
    python ingest_hy3d.py --from-release EditHero/data/captions/hy3d_captions.json
"""
import os, sys, json, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_replace_chains import shape_signature, axisymmetry
from embed_retrieve import VECS_HY3D, META_HY3D


def read_annot(annot_dir):
    caps, meta = {}, {}
    for f in sorted(glob.glob(os.path.join(annot_dir, "captions*.jsonl"))):
        for line in open(f):
            try:
                e = json.loads(line); caps[(e["oid"], str(e["pid"]))] = e["cap"].strip()
            except Exception:
                continue
    for f in sorted(glob.glob(os.path.join(annot_dir, "parts_meta*.jsonl"))):
        for line in open(f):
            try:
                e = json.loads(line); meta[(e["oid"], str(e["pid"]))] = e["bbox"]
            except Exception:
                continue
    return caps, meta


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--annot-dir", help="output folder of hy3d_captions.py")
    g.add_argument("--from-release", help="hy3d_captions.json of the released dataset ({oid: {pid: {cap, bbox}}})")
    ap.add_argument("--exclude-align-fail", default=None, help="folder of the bake reports report_g*.jsonl")
    a = ap.parse_args()
    bad = set()
    if a.exclude_align_fail:
        for f in glob.glob(os.path.join(a.exclude_align_fail, "report_g*.jsonl")):
            for line in open(f):
                try:
                    e = json.loads(line)
                    if e.get("status") == "align_fail":
                        bad.add(e["oid"])
                except Exception:
                    continue
    if a.annot_dir:
        caps, meta = read_annot(a.annot_dir)
    else:
        rel = json.load(open(a.from_release))
        caps = {(o, p): v["cap"] for o, ps in rel.items() for p, v in ps.items()}
        meta = {(o, p): v["bbox"] for o, ps in rel.items() for p, v in ps.items()}
    old = json.load(open(META_HY3D)) if os.path.isfile(META_HY3D) else []
    old_v = np.load(VECS_HY3D) if os.path.isfile(VECS_HY3D) else np.zeros((0, 384), np.float32)
    have = {(i["oid"], str(i["pid"])) for i in old}
    new = []
    for k, cap in caps.items():
        if k in have or k not in meta or not cap or k[0] in bad:
            continue
        bb = meta[k]
        new.append(dict(oid=k[0], pid=k[1], cap=cap, bbox=bb, sig=[float(x) for x in shape_signature(bb)],
                        sym=float(axisymmetry(bb)), ds="hy3d"))
    print(f"captions {len(caps)}  meta {len(meta)}  already {len(old)}  new {len(new)}", flush=True)
    if not new:
        return
    from embed_library import encoder
    from build_replace_chains import head_clause
    texts = [head_clause(i["cap"]).strip().rstrip(".") for i in new]   # the library encodes the head clause
    vecs = encoder("cuda")(texts)
    out_m = old + new; out_v = np.concatenate([old_v, vecs.astype(np.float32)])
    assert len(out_m) == len(out_v)
    json.dump(out_m, open(META_HY3D + ".tmp", "w"), ensure_ascii=False)
    np.save(VECS_HY3D + ".tmp.npy", out_v)
    os.replace(META_HY3D + ".tmp", META_HY3D); os.replace(VECS_HY3D + ".tmp.npy", VECS_HY3D)
    print(f"library now {len(out_m)} HY3D parts -> {META_HY3D}", flush=True)


if __name__ == "__main__":
    main()
