"""Replace parts using embedding nearest-neighbor retrieval, replacing the
original lexical set intersection approach.

## What was replaced

In the original `rank_replacements`, the semantic layer worked like this:

    hit = len(accepts & ck)          # slot vocabulary ∩ candidate description vocabulary
    if accepts and not hit: continue

Two problems:

1. **Recognizing characters, not meaning.** When a slot is labeled `helmet` but
   the description is "reptilian cranium" (a crocodile head), no words match at all,
   so it gets **eliminated outright**, not just downscored. To patch this, I manually
   wrote 37 words for the "head" slot in the registry—that's not describing the slot,
   that's making the encoder memorize a synonym table.
2. **Only coverage 5.5%.** Lexical matching can't be batched, so we randomly sample
   3000 times from the library (54,145 candidates), then select the top 8 from hits.
   The "best fit" is really "best among the 3000 we happened to sample."

Vector retrieval solves both: one matrix multiplication scores the **entire library**
in tens of milliseconds. Synonyms and hypernyms (cranium/skull/head) already cluster
together in vector space.

## Scoring method

The slot provides a few **example queries** (`queries`); candidates contribute the
first part of their description. After encoding each, we compute cosine similarity.
For each slot, we take the **maximum** across example queries, not the average—a head
slot accepts both helmets and animal heads, and averaging would yield a midpoint that
looks like neither; the max score means "matches any of the concepts."

    score = 10.0 · semantic_similarity (saturates after 0.58)   # heaviest, user-prioritized
          -  1.5 · symmetry_diff
          -  1.0 · shape_diff

Geometry uses hard gates as before (`d_sig ≤ 0.6`, `d_sym ≤ 0.4`);
vectors only control the semantic layer.

**Semantic scores must saturate.** The first version used `10 · sim` directly; the
result was that the top 12 head-slot candidates were all helmets, arm-slot candidates
all "A human arm and hand"—the highest-similarity item is always the original's synonym,
so swapping it is like not swapping at all, hitting the "pixel change <1% after edits"
pitfall again. Semantic priority means **reasonable**, not **identical**; once
reasonable, other slots in the ranking should go to symmetry, shape, and diversity.

## Where lexical matching remains

`rejects` keeps literal matching. These are words like wall / floor / ceiling that
should be rejected on the surface; they're already precise, so vector replacement might
harm roundish things. Vectors add only one high-threshold fallback.

## Returns are not pure top-k

Taking the top 8 by score yields 8 nearly identical items (all helmets). When placement
fails and we need to retry with the next candidate, we get the same thing—no progress.
So **each source object contributes at most one candidate**, then we take the top 8.
"""
import os, re, json, functools
import numpy as np

from local_paths import DATA_ROOT as D
VECS = os.path.join(D, "lib_vecs.npy")
META = os.path.join(D, "lib_meta.json")
# HY3D part library (equal standing with PartVerseXL,
# annotations incorporated as they are produced). ingest_hy3d.py incrementally
# rewrites these files; if they exist, concatenate and load, with entries tagged ds="hy3d".
VECS_HY3D = os.path.join(D, "lib_vecs_hy3d.npy")
META_HY3D = os.path.join(D, "lib_meta_hy3d.json")

SIM_MIN = 0.30        # Below this, candidate doesn't count as "acceptable for this slot"
SIM_SAT = 0.58        # Above this similarity, score saturates—considered "good enough",
                      # no need for higher scores (see module docstring)
SIM_REJECT = 0.72     # Similarity to reject examples above this → eliminate outright
D_SIG_MAX = 0.60      # Shape prototype difference threshold (aspect ratios, stable across runs)
D_SYM_MAX = 0.40      # Rotational symmetry difference threshold (stable across runs)
MMR_LAMBDA = 6.0      # Deduplication strength between candidates; see _diversify()
MMR_DUP = 0.86        # Similarity above this → treat as identical, discard


QVECS = os.path.join(D, "query_vecs.npz")
QVECS_EXTRA = os.path.join(D, "query_vecs_extra.npz")     # Ad-hoc encoded queries (for add operations)
from local_paths import ENCODER_PYTHON as ENCODER_PY


@functools.lru_cache(maxsize=1)
def _cached_queries():
    """Pre-encoded query embeddings.

    Placement runs in the `articraft` environment, which **lacks torch**, so
    on-the-fly encoding would fail with ImportError. Queries are a small fixed set,
    pre-computed by `embed_library.py --queries-only`; we only read from disk here.
    """
    out = {}
    for path in (QVECS, QVECS_EXTRA):
        if os.path.isfile(path):
            z = _load_qvecs(path)
            out.update({str(t): v for t, v in zip(z["texts"], z["vecs"])})
    return out


def _load_qvecs(path):
    """Query caches hold a Unicode text array and a float array; they are read without pickle. Caches written by older versions
    (object arrays) are rejected: delete them and rebuild with `embed_library.py --queries-only`."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return dict(texts=np.asarray(z["texts"]), vecs=np.asarray(z["vecs"]))
    except ValueError as e:
        raise RuntimeError(f"{path} is in an old format that needs pickle to load; delete it and rebuild it with "
                           "embed_library.py --queries-only") from e


def _encode_subprocess(texts):
    """When torch is unavailable in the current environment (articraft), spawn a
    subprocess in the trellis2 environment to encode, and append results to
    query_vecs_extra.npz (add-operation queries are written on-the-fly by agents
    and cannot be pre-computed)."""
    import subprocess, tempfile, json as _json
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory(dir=D) as td:
        tin, tout = os.path.join(td, "q.json"), os.path.join(td, "v.npy")
        _json.dump(list(texts), open(tin, "w"), ensure_ascii=False)
        code = ("import sys, json, numpy as np; sys.path.insert(0, sys.argv[1]); "
                "from embed_library import encoder; enc = encoder('cpu'); "
                "np.save(sys.argv[3], enc(json.load(open(sys.argv[2]))))")
        env = dict(os.environ); env.pop("LD_PRELOAD", None)
        subprocess.run([ENCODER_PY, "-c", code, here, tin, tout], check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
        vecs = np.load(tout)
    # Read-and-write must happen inside a file lock: when multiple servers encode
    # concurrently, whoever calls os.replace last drops all others' vectors, and the
    # other server later gets KeyError when trying to fetch its query from cache (W1-G incident).
    import fcntl
    with open(QVECS_EXTRA + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        old_t, old_v = [], np.zeros((0, vecs.shape[1]), np.float32)
        if os.path.isfile(QVECS_EXTRA):
            z = _load_qvecs(QVECS_EXTRA)
            old_t, old_v = list(z["texts"]), z["vecs"]
        seen = {str(t) for t in old_t}
        add = [(t, v) for t, v in zip(texts, vecs) if str(t) not in seen]
        if add:
            tmp = QVECS_EXTRA + f".tmp{os.getpid()}.npz"
            np.savez(tmp, texts=np.array([str(t) for t in old_t] + [str(t) for t, _ in add], dtype=str),
                     vecs=np.concatenate([old_v] + [np.asarray([v], np.float32) for _, v in add]))
            os.replace(tmp, QVECS_EXTRA)
    _cached_queries.cache_clear()
    return vecs


@functools.lru_cache(maxsize=1)
def _encoder():
    from embed_library import encoder
    return encoder("auto")


def encode(texts):
    """If all queries are in cache, skip loading the model—environments without
    torch depend on this fallback."""
    texts = list(texts)
    cache = _cached_queries()
    if all(t in cache for t in texts):
        return np.stack([cache[t] for t in texts])
    miss = [t for t in texts if t not in cache]
    try:
        enc = _encoder()
        return enc(texts)
    except ImportError:
        pass
    try:
        _encode_subprocess(miss)
    except Exception as e:
        raise RuntimeError(
            f"These queries are not pre-encoded, current environment has no encoder, "
            f"and subprocess encoding also failed ({e}): {miss[:3]}\n"
            f"After modifying queries in the registry, run once in the partflow environment:\n"
            f"  python embed_library.py --queries-only") from e
    cache = _cached_queries()
    return np.stack([cache[t] for t in texts])


def _as_phrase(w):
    """Registry 'accepts' are bare words (`helmet`); wrapping as noun phrases
    gives more stable encoding."""
    w = w.strip()
    return w if " " in w else f"a {w}"


class EmbedLibrary:
    """Interface compatible with PartLibrary (pick_replacement / rank_replacements),
    using embeddings internally."""

    def __init__(self, vecs=VECS, meta=META):
        self.V = np.load(vecs)                       # [N, 384], normalized
        self.items = json.load(open(meta))
        for i in self.items:
            i.setdefault("ds", "pv")
        if os.path.isfile(VECS_HY3D) and os.path.isfile(META_HY3D):
            v2 = np.load(VECS_HY3D)
            m2 = json.load(open(META_HY3D))
            if len(v2) == len(m2) and len(m2):
                for i in m2:
                    i.setdefault("ds", "hy3d")
                self.V = np.concatenate([self.V, v2.astype(self.V.dtype)])
                self.items = self.items + m2
                print(f"[embed_library] Merged {len(m2)} HY3D parts", flush=True)
        self.sig = np.array([i["sig"] for i in self.items], np.float32)   # [N, 3]
        self.sym = np.array([i["sym"] for i in self.items], np.float32)   # [N]
        self.oid = np.array([i["oid"] for i in self.items])
        self.cap = [i["cap"] for i in self.items]
        # Same (oid, pid) may appear in both pv and hy3d: must store index list.
        # Storing a single value would make veto only mask the later-merged copy
        # (W3-2 incident: after veto, the replacement that comes back is still it).
        self._pos = {}
        for k, i in enumerate(self.items):
            self._pos.setdefault((i["oid"], i["pid"]), []).append(k)
        self.ds = [i.get("ds", "pv") for i in self.items]
        self._qcache = {}
        print(f"[embed_library] {len(self.items)} parts × {self.V.shape[1]} dims "
              f"(from {len(set(self.oid))} objects)")

    def _q(self, texts):
        """Cache encoded query results—the same slot is looked up repeatedly
        throughout a chain."""
        key = tuple(texts)
        if key not in self._qcache:
            self._qcache[key] = encode(texts)
        return self._qcache[key]

    def rank_replacements(self, cur, rng=None, exclude_oid=None, tries=None, topk=8,
                          exclude=(), temperature=0.35, sig_free=False):
        """Score the entire library, return topk candidates (at most one per source
        object, mutually distinct).

        `cur` requires: caption, bbox; optionally queries / accepts / rejects.
        `tries` parameter kept only for compatibility with PartLibrary call sites—
        here we score the full library, no sampling.

        `exclude` is a set of (oid, pid) pairs: parts **already used for this slot
        in this chain**. Without exclusion, after 20 rounds the left leg might use the
        same "A human foot" five times over—library scoring is deterministic, the slot's
        queries and bbox are fixed per round, so it returns the same top-1 each time.
        The original sampled 3000 candidates randomly, which accidentally provided
        variation; embeddings removed that randomness, so we must restore it explicitly.

        `temperature` controls the final sampling step: after MMR selection, we don't
        sort strictly by score but sample via exp(score/T). T=0 degenerates to pure sorting.
        """
        from build_replace_chains import shape_signature, axisymmetry, head_clause, keywords

        sig = shape_signature(cur["bbox"]).astype(np.float32)
        sym = float(axisymmetry(cur["bbox"]))

        # --- Geometry hard gates: shrink candidate pool first (add operations have no
        # original to compare against; skip when sig_free) ---
        if sig_free:
            ok = np.ones(len(self.items), bool)
        else:
            ok = ((np.abs(self.sig - sig).sum(1) <= D_SIG_MAX)
                  & (np.abs(self.sym - sym) <= D_SYM_MAX))
        if exclude_oid is not None:
            ok &= (self.oid != exclude_oid)
        if exclude:
            ex = {(o, p) for o, p in exclude}
            for k in ex:
                for j in self._pos.get(k, ()):
                    ok[j] = False
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            return None

        # --- rejects: literal match → eliminate (wall / floor type words that should be rejected outright) ---
        rej = [w.lower() for w in (cur.get("rejects") or [])]
        if rej:
            pat = re.compile(r"\b(" + "|".join(map(re.escape, rej)) + r")\b", re.I)
            idx = idx[[not pat.search(head_clause(self.cap[i])) for i in idx]]
            if idx.size == 0:
                return None

        # --- Semantics: one matrix multiplication over the entire library ---
        qtexts = list(cur.get("queries") or [_as_phrase(w) for w in (cur.get("accepts") or [])])
        if not qtexts:
            # Slot has no specified requirements (e.g., unregistered object), so use
            # **the original's own description** as the query. The intent is "swap for
            # something roughly in the same category." Saturation + diversity deduplication
            # prevents falling back to picking only synonyms.
            qtexts = [head_clause(cur["caption"]).strip().rstrip(".")]
        Vs = self.V[idx]
        s_acc = (Vs @ self._q(qtexts).T).max(1)            # Take max over example queries, not average
        keep = s_acc >= SIM_MIN

        if rej and qtexts:                                  # Embedding fallback: also eliminate if too similar to reject items
            s_rej = (Vs @ self._q([_as_phrase(w) for w in rej]).T).max(1)
            keep &= (s_rej < SIM_REJECT) & (s_acc > s_rej)
        if not keep.any():
            return None
        idx, s_acc, Vs = idx[keep], s_acc[keep], Vs[keep]

        d_sig = np.abs(self.sig[idx] - sig).sum(1) * (0.0 if sig_free else 1.0)
        d_sym = np.abs(self.sym[idx] - sym) * (0.0 if sig_free else 1.0)

        # Semantic scores **saturate above the threshold**. Using `10·sim` directly,
        # 0.78 similarity would overwhelm 0.60, and the selection would always be the
        # original's synonyms—head slot top 12 all helmets, arm slot top 10 all
        # "A human arm and hand." That's like not swapping at all, the exact "pixel
        # change <1%" pitfall again. Semantic priority means **reasonable**, not
        # **identical**: once good enough, the remaining slots go to symmetry, shape,
        # and diversity. The part above SIM_SAT is weighted at one-quarter strength.
        s_eff = np.minimum(s_acc, SIM_SAT) + 0.25 * np.maximum(s_acc - SIM_SAT, 0.0)
        score = 10.0 * s_eff - 1.5 * d_sym - 1.0 * d_sig

        keep = self._diversify(idx, score, Vs, topk)
        if rng is not None and temperature > 0 and len(keep) > 1:
            keep = self._sample_order(keep, score, rng, temperature)
        out = []
        for k in keep:
            c = dict(self.items[int(idx[k])])
            c.setdefault("ds", "pv")
            c["sig"] = np.asarray(c["sig"], np.float32)
            c["kw"] = keywords(c["cap"])
            c["score"] = float(score[k])
            c["sim"] = float(s_acc[k])
            out.append(c)
        return out

    @staticmethod
    def _sample_order(keep, score, rng, T):
        """Sample without replacement via exp(score/T) distribution, not strict score order.

        Placement always tries the first item in the list, so "rank first" equals
        "selected." Library scoring is deterministic; without this step, the same slot
        each round would return the same candidate.
        """
        s = np.array([score[k] for k in keep], float)
        w = np.exp((s - s.max()) / max(T, 1e-6))
        order, pool = [], list(range(len(keep)))
        while pool:
            p = w[pool] / w[pool].sum()
            j = pool[min(int(np.searchsorted(np.cumsum(p), rng.random())), len(pool) - 1)]
            order.append(keep[j]); pool.remove(j)
        return order

    def _diversify(self, idx, score, Vs, topk):
        """Greedily select topk items, **mutually distinct** (maximum marginal relevance).

        Selecting top k by score alone yields k synonyms. When placement fails and we
        retry the next candidate, we get the same item—retry is wasted. So for each
        selection, we suppress similar ones: marginal_score = score - MMR_LAMBDA ·
        (max similarity to already-selected).

        Also: each source object contributes at most one—left and right hands from the
        same object would be split into two candidates but are the same thing.
        """
        order = np.argsort(-score)[:2000]        # Pick only from top-scoring to avoid pairwise similarity over full library
        chosen, seen_oid = [], set()
        simsel = np.full(len(order), -1.0, np.float32)   # Max similarity of each candidate to already-selected
        pool = list(range(len(order)))
        while pool and len(chosen) < topk:
            best, best_v = None, -1e9
            for pi in pool:
                v = score[order[pi]] - MMR_LAMBDA * max(simsel[pi], 0.0)
                if v > best_v:
                    best, best_v = pi, v
            k = order[best]
            chosen.append(int(k))
            seen_oid.add(self.oid[int(idx[k])])
            s = Vs[order[pool]] @ Vs[k]
            for j, pi in enumerate(pool):
                simsel[pi] = max(simsel[pi], float(s[j]))
            pool = [pi for pi, sv in zip(pool, s)
                    if pi != best and sv < MMR_DUP
                    and self.oid[int(idx[order[pi]])] not in seen_oid]
        return chosen

    def pick_replacement(self, cur, rng, exclude_oid, tries=None, topk=8):
        out = self.rank_replacements(cur, rng, exclude_oid, topk=topk)
        return rng.choice(out) if out else None


def main():
    import argparse, random
    ap = argparse.ArgumentParser(description="Query self-check: given a slot, see what ranks at the top")
    ap.add_argument("--caption", default="A robot's head.")
    ap.add_argument("--queries", nargs="*", default=None, help="Example queries; if omitted, use --accepts")
    ap.add_argument("--accepts", nargs="*", default=["head", "helmet", "mask"])
    ap.add_argument("--rejects", nargs="*", default=["wall", "floor", "wheel"])
    ap.add_argument("--bbox", nargs=6, type=float,   # lo xyz + hi xyz, combined below as [lo, hi]
                    default=[-0.1, -0.1, -0.1, 0.1, 0.1, 0.1])
    ap.add_argument("--topk", type=int, default=15)
    a = ap.parse_args()

    lib = EmbedLibrary()
    bb = [a.bbox[:3], a.bbox[3:]]
    cur = dict(caption=a.caption, bbox=bb, accepts=a.accepts,
               rejects=a.rejects, queries=a.queries)
    out = lib.rank_replacements(cur, random.Random(0), exclude_oid=None, topk=a.topk)
    print(f"\nSlot: {a.caption}   Queries: {a.queries or a.accepts}")
    for c in out or []:
        print(f"  {c['score']:6.2f}  sim={c['sim']:.3f}  {c['cap'][:88]}")


if __name__ == "__main__":
    main()
