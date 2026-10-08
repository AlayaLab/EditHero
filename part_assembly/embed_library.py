"""Encode part descriptions into vectors for retrieval.

Replaces the original lexical set intersection. Lexical matching has hard ceiling:
if a slot's `accepts` says "helmet", but the description says "reptilian cranium"
for a crocodile head, it will **not match a single word and be eliminated directly** —
not a low score, but cannot enter the candidate pool at all. Synonyms and
hypernyms/hyponyms are not recognized at all.

Uses `all-MiniLM-L6-v2` (locally cached, 384 dimensions). Encodes **only the first half** —
description is like "A black Santa hat, which is a component of the sleigh.",
the second half says which parent object it came from, not what it is itself;
encoding it will pull parts "from the same parent class" together, which is not
what we want.

Output:
    lib_vecs.npy    [N, 384] normalized vectors
    lib_meta.json   [N] each vector's {oid, pid, cap, bbox, sig, sym}

At retrieval time, the slot encodes the same way: each word in `accepts`
gets encoded to a vector, take **maximum similarity** (matching any concept counts),
not average — average would blur "helmet/mask/animal head" into a center
that doesn't look like any of them.
"""
import os, sys, json, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_replace_chains import (head_clause, shape_signature, axisymmetry,
                                  BAD, SCENE, TABOO, is_generic, captions_path,
                                  ANNO, TEX_PART, D)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT_VECS = os.path.join(D, "lib_vecs.npy")
OUT_META = os.path.join(D, "lib_meta.json")


def encoder(device="auto"):
    import torch
    from transformers import AutoTokenizer, AutoModel
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoder running on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = AutoModel.from_pretrained(MODEL).to(device).eval()

    @torch.no_grad()
    def enc(texts, batch=256):
        out = []
        for i in range(0, len(texts), batch):
            b = tok(texts[i:i + batch], padding=True, truncation=True,
                    max_length=64, return_tensors="pt").to(device)
            h = mod(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)      # mean pooling
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 384), np.float32)
    return enc


def collect(caps_path=None, limit=None):
    """Use the same filtering as PartLibrary to ensure candidate pools are consistent."""
    p = caps_path or captions_path()
    print(f"Part descriptions from {p}", flush=True)
    caps = json.load(open(p))
    items = []
    for oid, parts in caps.items():
        ip = os.path.join(ANNO, oid, f"{oid}_info.json")
        if not os.path.isfile(ip):
            continue
        try:
            info = json.load(open(ip))
        except Exception:
            continue
        lab = info.get("ordered_face_label") or []
        bbs = info.get("bboxes") or []
        pid2rank = {str(x): r for r, x in enumerate(lab)}
        for pid, v in parts.items():
            if (not v or BAD.search(v[0]) or SCENE.search(v[0])
                    or TABOO.search(v[0]) or is_generic(v[0])):
                continue
            r = pid2rank.get(pid)
            if r is None or r >= len(bbs):
                continue
            if not os.path.isfile(os.path.join(TEX_PART, oid, f"{pid}.glb")):
                continue
            items.append(dict(oid=oid, pid=pid, cap=v[0], bbox=bbs[r],
                              sig=shape_signature(bbs[r]).tolist(),
                              sym=axisymmetry(bbs[r])))
        if limit and len(items) >= limit:
            break
    return items


QVECS = os.path.join(D, "query_vecs.npz")


def refresh_queries(device="auto"):
    """Pre-encode and save all slot example queries (and reject fallback queries) from the registry.

    The placement step runs in the `articraft` environment, **which has no torch** —
    it only has the mini_articraft suite. At retrieval time, encoding on the fly
    would directly raise ImportError. Fortunately, queries are a fixed small set
    (three or four per slot), compute them in advance and save, at retrieval time
    only a dot product remains, pure numpy is enough.

    After modifying the registry's `queries`, rerun this, otherwise new queries
    won't be found.
    """
    import slot_registry as SR
    texts = set()
    for items in SR._load().values():
        for d in items:
            for r in (d.get("roles") or {}).values():
                texts.update(r.get("queries") or [])
                texts.update(f"a {w}" if " " not in w else w
                             for w in (r.get("rejects") or []))
    texts = sorted(texts)
    if not texts:
        print("No example queries in registry, skipping")
        return
    vec = encoder(device)(texts)
    np.savez(QVECS, texts=np.array(texts, dtype=str), vecs=vec.astype(np.float32))   # Unicode array: loads without pickle
    print(f"Cached {len(texts)} queries -> {QVECS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--queries-only", action="store_true",
                    help="Only refresh vector cache for registry example queries, do not re-encode entire library")
    a = ap.parse_args()

    if a.queries_only:
        refresh_queries(a.device)
        return

    items = collect(limit=a.limit)
    print(f"Candidate pool {len(items)} parts (same filtering as PartLibrary)", flush=True)
    texts = [head_clause(x["cap"]).strip().rstrip(".") for x in items]
    enc = encoder(a.device)
    import time
    t = time.time()
    vecs = enc(texts)
    print(f"Encoding complete {vecs.shape}, {time.time()-t:.1f}s elapsed", flush=True)
    np.save(OUT_VECS, vecs.astype(np.float32))
    json.dump(items, open(OUT_META, "w"), ensure_ascii=False)
    print(f"  {OUT_VECS}  {os.path.getsize(OUT_VECS)/2**20:.1f} MB")
    print(f"  {OUT_META}  {os.path.getsize(OUT_META)/2**20:.1f} MB")
    refresh_queries(a.device)


if __name__ == "__main__":
    main()
