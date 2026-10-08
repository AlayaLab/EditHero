"""Build long pure-replacement chains: each turn replaces one part on an object with another semantically different part.

Why build pure-replacement chains separately:

  1. **Part count is constant.** In mixed add-delete chains, the two degenerate behaviors—"do nothing" and "delete all"—
     must be separated by splitting implementation scores (a pitfall we hit twice before). Under pure replacement,
     both score zero each turn, so no workaround is needed.
  2. **Placement is automatic.** New parts are positioned at the replaced part's location, scaled and aligned by its bounding box.
     No need for the agent to decide "where should the hat go", because slot positions are fixed—this is the simplest aspect of pure replacement.
  3. **The 2D side lacks this too.** Currently, the longest multi-turn image editing benchmark is 5 turns (MSE-Bench);
     EdiVal and MICE-Bench are both 3 turns and use mixed instruction types, with no pure-replacement long chains alone.

The ground truth for each turn is **the rendered image after assembly**, not the generated image—assembly is deterministic;
identical inputs always produce identical outputs and depend on no model.

Chain state is a "slot → current occupant" mapping:

    turn 0: {0: original0, 1: original1, ..., n: originaln}
    turn 1: {0: original0, 1: replacedA,  ..., n: originaln}
    turn 2: {0: replacedB,  1: replacedA,  ..., n: originaln}
    ...

The same slot can be replaced multiple times, so chain length can exceed part count—an object with median 9 parts can run 20 turns.
"""
import os, sys, json, glob, random, argparse, re
import numpy as np

from local_paths import DATA_ROOT as D
ANNO = f"{D}/partverse_anno/anno_infos"
TEX_OBJ = f"{D}/pv_textured/normalized_glbs"
TEX_PART = f"{D}/pv_textured/textured_part_glbs"

# These terms indicate the description refers to a region on the parent object, not a removable, placeable part.
# Descriptions like "close-up"/"view of" describe **how to view it**, not **what it is**; retrieving by them is pointless.
# Previously only matched "a close-?up of", missing "a detailed close-up of ..." with words in between,
# so "A detailed close-up of a wolf's head" ranked second for the head slot. Changed to not require adjacency.
BAD = re.compile(r"\b(close-?up (of|view)|detailed view of|a section of|"
                 r"(view|shot|render) of)\b", re.I)

# "A segment of the cylindrical body panel …"—describes a small piece on a surface;
# removing and placing it makes no sense. Only 23 in the whole library, but one can ruin an entire turn (the chain drew it on turn 1).
FRAGMENT = re.compile(r"^(a|an|the)\s+(segment|portion|fragment|sliver|strip|patch)\s+of\b", re.I)

# Scene elements, not things that can be picked up and placed.
SCENE = re.compile(r"\b(floor|wall|ground|ceiling|backdrop|tile[ds]? surface|"
                   r"base plate|platform|terrain|grass patch)\b", re.I)

# Things no one would mount on another object. The criterion is not "unclean" but **placing it makes no sense**:
# a toilet seat only works laid flat; mounting it vertically on a vase is neither stable nor something anyone would do to a figurine.
TABOO = re.compile(r"\b(toilet|urinal|bidet|sewage|septic|trash|garbage|waste|"
                   r"dumpster|feces|manure|litter box)\b", re.I)
# Vague descriptions—"a rectangular component", "a black piece".
# When replaced, such candidates become featureless gray blocks, but we want things like "crocodile head/dog head/cube head"
# that are recognizable at a glance. These make up 23.5% of the dataset and are filtered directly.
GENERIC_HEAD = re.compile(
    r"^(a|an|the)\s+[\w\s,'-]*?\b(component|piece|element|object|part|section|"
    r"block|shape|structure|unit|item|fragment)\b\s*$", re.I)


def captions_path():
    """Where to read part descriptions. **Qwen3's version takes priority**; fall back to PartVerse's original if absent.

    The PartVerse-XL release annotations lack part semantics; `text_captions.json` is labeled by Qwen2.5
    with coarse granularity and many errors (e.g., a cross-shaped bracket was labeled "a wheel from the vehicle",
    retrieval accepted it uncritically). After Qwen3 re-labels, use the new version.

    The `PXFORM_CAPTIONS` environment variable can force a choice for control experiments.
    """
    env = os.environ.get("PXFORM_CAPTIONS")
    if env and os.path.isfile(env):
        return env
    q3 = os.path.join(D, "text_captions_qwen3.json")
    if os.path.isfile(q3):
        return q3
    return glob.glob(f"{D}/../hf_cache/**/text_captions.json", recursive=True)[0]


def is_generic(caption):
    h = head_clause(caption).strip().rstrip(".")
    return bool(GENERIC_HEAD.match(h) or FRAGMENT.match(h))


STOP = set("a an the of from is are this that it its with and or in on to for which was were "
           "be been part component object image shows appears extracted detached removed "
           "isolated close up left right red box larger main small large".split())


# Clauses after commas starting with these words say "where it originally came from, what it does, what details",
# not "what it is". Retrieval and instruction generation only need the first half.
# The first four are Qwen2.5's patterns (", which is a component of the sleigh");
# featuring / serving / designed / consisting are Qwen3's—its descriptions are whole long sentences, and without truncation,
# the generated instruction reads
#   "Replace A gray helmet with a faceted, dome-shaped design and a side cheek guard,
#    serving as head protection for the warrior figure. with the side shield of ..."
# which is both long and ungrammatical. In practice, the most common connectives after commas in Qwen3 descriptions are featuring(1899),
# serving(1346), designed(689).
_TAIL = re.compile(
    r",\s+(which\s+(is|was)|a component|part of|from the|serving\s|located\s|"
    r"featuring\s|designed\s|used\s|consisting\s|forming\s|providing\s|"
    r"including\s|attached\s|connected\s)", re.I)


# Opening phrase. Qwen3 has 478 descriptions that start this way; keeping them produces instructions like
# "Replace the muzzle … with **the object is** a magazine release button".
_LEAD = re.compile(r"^\s*(the object is|this is|it is|the image shows)\s+", re.I)


# Copula sentences: "The round dining table top is a circular, flat surface that serves…"
# must be compressed to "the round dining table top". Without truncation, the generated instruction is ungrammatical and **propagates along the chain**
# (a bad description becomes "the replaced side" next turn; Agent C's oven chain has bad instructions for 4 turns because of this).
# Only truncate when the subject side has meaningful words (>2 words), to avoid cutting "It is a hat" into an empty shell
# —those opening phrases are handled by _LEAD.
_COPULA = re.compile(r"^((?:the|a|an)\s+(?:[\w'-]+\s+){1,7}[\w'-]+)\s+(?:is|are)\s+(?:a|an|the)\s", re.I)


def head_clause(c):
    """Extract only the "what it is" part: drop opening phrases and clauses about origin or detail."""
    c = _LEAD.sub("", str(c))
    m2 = _COPULA.match(c)
    if m2:
        c = m2.group(1)
    m = _TAIL.search(c)
    return c[:m.start()] if m and m.start() > 0 else c


def keywords(c):
    ws = re.sub(r"[^a-z ]", " ", head_clause(c).lower()).split()
    return {w for w in ws if w not in STOP and len(w) > 2}


def bbox_size(bb):
    lo, hi = np.array(bb[0], float), np.array(bb[1], float)
    return np.abs(hi - lo)


def aspect(bb):
    d = np.sort(bbox_size(bb))[::-1]
    return float(d[0] / max(d[2], 1e-6))


def shape_signature(bb):
    """Shape prototype: sort three dimensions from long to short, normalize to (1, b, c).

    Swapping a human head for a crocodile head works because both are "a thick cylinder";
    swapping a rose for a sunflower works because both are "a round bulb on a thin stem".
    Whether a swap can work depends on whether this shape prototype matches, **not** semantic relatedness—
    a previous version required "keywords must not overlap", which actively sought the most unrelated things
    and ended up pairing a vase with a toilet seat.
    """
    d = np.sort(bbox_size(bb))[::-1]
    d = d / max(d[0], 1e-9)
    return np.array([1.0, d[1], d[2]])


def axisymmetry(bb):
    """Axial symmetry about the longest axis: the closer the other two dimensions are, the more it resembles a solid of revolution (cylinder/cone/sphere).

    This is the other half of why swapping a human head for a crocodile head works—both revolve roughly around one axis.
    Replacing with a flat plate fails, even if bounding box volumes are similar.
    """
    d = np.sort(bbox_size(bb))[::-1]
    return float(min(d[1], d[2]) / max(d[1], d[2], 1e-9))


class PartLibrary:
    """Collect parts with captions into a searchable library."""

    def __init__(self, caps, limit=None):
        self.items = []          # (oid, pid, caption, keywords, bbox, aspect ratio)
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
            pid2rank = {str(p): r for r, p in enumerate(lab)}
            for pid, v in parts.items():
                if (not v or BAD.search(v[0]) or SCENE.search(v[0])
                        or TABOO.search(v[0]) or is_generic(v[0])):
                    continue
                r = pid2rank.get(pid)
                if r is None or r >= len(bbs):
                    continue
                if not os.path.isfile(os.path.join(TEX_PART, oid, f"{pid}.glb")):
                    continue
                self.items.append(dict(oid=oid, pid=pid, cap=v[0], kw=keywords(v[0]),
                                       bbox=bbs[r], asp=aspect(bbs[r]),
                                       sig=shape_signature(bbs[r]), sym=axisymmetry(bbs[r])))
            if limit and len(self.items) >= limit:
                break
        print(f"[library] {len(self.items)} usable parts"
              f"(from {len(set(i['oid'] for i in self.items))} objects)")

    def pick_replacement(self, cur, rng, exclude_oid, tries=3000, topk=8):
        out = self.rank_replacements(cur, rng, exclude_oid, tries, topk)
        return rng.choice(out) if out else None

    def rank_replacements(self, cur, rng, exclude_oid, tries=3000, topk=8,
                          exclude=(), temperature=None):
        """Rank candidates in three priority tiers, **semantics first**.

        Priorities align with kitbash/shape-language design guidelines:

          1. **Semantics**—what this slot can accept is specified per-object in a registry (`accepts`/`rejects`).
             Heads can swap for helmets, masks, animal heads, spheres, lanterns; **cannot swap for wall panels, wheels**.
             Legs can swap for wheels (robots already have wheel legs), but arms cannot—whether a candidate suits different slots differs,
             so criteria attach to slots, not candidates.
          2. **Symmetry**—axial revolution quality must be close. Head-to-crocodile-head works, flat plate does not.
          3. **Shape/silhouette**—three dimension ratios are similar. Design guidelines say readability depends on silhouette,
             so this tier ranks below "same category", but cannot differ too much.

        Returns the topk highest-scoring candidates, **sorted by score**. When placement fails (snap_to rejects, size spirals,
        assembly falls apart), we must fall back to the next option, so this returns a list, not a single candidate.

        `temperature` is a vector-retrieval-side parameter; here only used for signature consistency—
        keyword matching is already random sampling with inherent variation.
        """
        ex = {(o, p) for o, p in exclude}
        sig = shape_signature(cur["bbox"]); sym = axisymmetry(cur["bbox"])
        accepts = {w.lower() for w in (cur.get("accepts") or [])}
        rejects = {w.lower() for w in (cur.get("rejects") or [])}
        kw = keywords(cur["caption"])
        scored = []
        for _ in range(tries):
            c = self.items[rng.randrange(len(self.items))]
            if c["oid"] == exclude_oid or (c["oid"], c["pid"]) in ex:
                continue
            ck = c["kw"]
            if rejects & ck:                      # Explicitly prohibited from this slot; eliminate immediately.
                continue
            d_sig = float(np.abs(sig - c["sig"]).sum())
            d_sym = abs(sym - c["sym"])
            if d_sig > 0.6 or d_sym > 0.4:
                continue
            hit = len(accepts & ck)
            if accepts and not hit:               # If slot specifies accepts, must match.
                continue
            score = (6.0 * min(hit, 2)             # Semantics: highest weight
                     - 1.5 * d_sym                 # Symmetry: next
                     - 1.0 * d_sig                 # Shape: next
                     + 0.3 * len(kw & ck))
            scored.append((score, c))
        if not scored:
            return None
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:topk]]


class PoolLibrary:
    """Fixed candidate pool: per-slot list of (oid, pid) hand-selected by LLM.

    Used for control experiments comparing embedding retrieval vs. LLM nomination—placement, gating, seed all unchanged;
    the only variable is candidate source. Selection follows three hard criteria from REPLACE_CHAIN_SPEC
    (shape prototype matches / symmetry consistent / gravitationally stable).

    Pool file format: {slot_name: [[oid, pid], ...]}. Part descriptions/bounding boxes are queried from lib_meta.json,
    same data as embedding side, ensuring fairness.
    """

    def __init__(self, pool_path):
        pool = json.load(open(pool_path))
        meta = json.load(open(os.path.join(D, "lib_meta.json")))
        idx = {(m["oid"], m["pid"]): m for m in meta}
        self.by_slot = {}
        miss = []
        for slot, lst in pool.items():
            items = []
            for oid, pid in lst:
                m = idx.get((oid, pid))
                if m is None:
                    miss.append((slot, oid[:8], pid)); continue
                items.append(dict(oid=m["oid"], pid=m["pid"], cap=m["cap"],
                                  kw=keywords(m["cap"]), bbox=m["bbox"],
                                  sig=np.asarray(m["sig"], np.float32),
                                  sym=float(m["sym"])))
            self.by_slot[slot] = items
        n = sum(len(v) for v in self.by_slot.values())
        print(f"[nomination pool] {len(self.by_slot)} slots / {n} candidates"
              + (f", {len(miss)} dropped (not in library): {miss[:4]}" if miss else ""))

    def rank_replacements(self, cur, rng, exclude_oid=None, tries=None, topk=8,
                          exclude=(), temperature=None):
        slot = cur.get("slot")
        items = self.by_slot.get(slot)
        if not items:
            return None
        ex = {(o, p) for o, p in exclude}
        avail = [c for c in items
                 if c["oid"] != exclude_oid and (c["oid"], c["pid"]) not in ex]
        if not avail:
            return None
        rng.shuffle(avail)          # Pool is an unordered set; variety across turns comes from shuffling + exclude.
        return avail[:topk]

    def pick_replacement(self, cur, rng, exclude_oid, tries=None, topk=8):
        out = self.rank_replacements(cur, rng, exclude_oid, topk=topk)
        return rng.choice(out) if out else None


def make_library(caps=None, limit=None, force_keyword=False):
    """Build a retrieval library. **Defaults to vector embedding**; falls back to keyword matching if vector file is absent.

    The vector library is generated once by `embed_library.py` (whole library in 7 seconds); both sides use identical filtering,
    and the candidate pool draws from the same parts. The keyword path is kept only as a fallback when encoding hasn't run—
    it recognizes words but not meaning; a description like "reptilian cranium" is unrelated to "helmet" in its eyes.
    """
    if not force_keyword:
        try:
            from embed_retrieve import EmbedLibrary, VECS
            if os.path.isfile(VECS):
                return EmbedLibrary()
            print("[library] vector file not found, falling back to keyword matching (run embed_library.py first)")
        except Exception as e:
            print(f"[library] vector library load failed ({e}), falling back to keyword matching")
    if caps is None:
        caps = json.load(open(captions_path()))
    return PartLibrary(caps, limit=limit)


def build_chain(oid, info, caps_o, lib, rng, turns):
    """Return a chain: which slot to replace each turn and with what."""
    lab = info.get("ordered_face_label") or []
    bbs = info.get("bboxes") or []
    slots = {}                                  # slot (= original part id) -> current occupant
    for r, p in enumerate(lab):
        pid = str(p)
        if r < len(bbs) and pid in caps_o and caps_o[pid] and not BAD.search(caps_o[pid][0]):
            slots[pid] = dict(kind="orig", oid=oid, pid=pid, caption=caps_o[pid][0],
                              bbox=bbs[r], rank=r)
    if len(slots) < 2:
        return None

    order = list(slots)
    chain = []
    for t in range(turns):
        # Cycle through slots, one lap at a time—ensures each slot is touched, then start the second cycle.
        slot = order[t % len(order)]
        cur = slots[slot]
        cand = lib.pick_replacement(cur, rng, exclude_oid=oid)
        if cand is None:
            continue
        new = dict(kind="repl", oid=cand["oid"], pid=cand["pid"], caption=cand["cap"],
                   bbox=cand["bbox"], rank=None)
        chain.append(dict(
            turn=t + 1, slot=slot,
            instruction=f"Replace the {head_clause(cur['caption']).lower().lstrip('a ').lstrip('an ')} "
                        f"with {head_clause(cand['cap']).lower()}",
            removed=dict(oid=cur["oid"], pid=cur["pid"], caption=cur["caption"]),
            added=dict(oid=cand["oid"], pid=cand["pid"], caption=cand["cap"],
                       glb=f"textured_part_glbs/{cand['oid']}/{cand['pid']}.glb",
                       shape_dist=float(np.abs(shape_signature(cur["bbox"]) - cand["sig"]).sum()),
                       sym_dist=float(abs(axisymmetry(cur["bbox"]) - cand["sym"]))),
            # 3D bounding box of the slot—new parts scale and align by it; placement needs no judgment.
            slot_bbox=slots[slot]["bbox"] if cur["kind"] == "orig" else cur["bbox"],
            state={k: dict(oid=v["oid"], pid=v["pid"]) for k, v in slots.items()},
        ))
        slots[slot] = new
        # Slot geometry is always the original; incoming parts scale into this box.
        slots[slot]["bbox"] = chain[-1]["slot_bbox"]
    return chain if len(chain) >= turns * 0.8 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-chains", type=int, default=100)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lib-limit", type=int, default=None)
    a = ap.parse_args()

    caps = json.load(open(captions_path()))
    lib = PartLibrary(caps, limit=a.lib_limit)
    rng = random.Random(a.seed)

    oids = sorted(caps)
    rng.shuffle(oids)
    out, tried = {}, 0
    for oid in oids:
        if len(out) >= a.n_chains:
            break
        tried += 1
        ip = os.path.join(ANNO, oid, f"{oid}_info.json")
        if not (os.path.isfile(ip) and os.path.isfile(os.path.join(TEX_OBJ, oid + ".glb"))):
            continue
        ch = build_chain(oid, json.load(open(ip)), caps[oid], lib, rng, a.turns)
        if ch:
            out[oid] = dict(source_glb=f"normalized_glbs/{oid}.glb",
                            n_slots=len(ch[0]["state"]), turns=ch)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    nt = sum(len(v["turns"]) for v in out.values())
    print(f"[chains] {len(out)} chains (tried {tried} objects), {nt} total turns,"
          f"avg {nt/max(len(out),1):.1f} turns/chain, avg {np.mean([v['n_slots'] for v in out.values()]):.1f} slots")
    print("wrote", a.out)
    k = next(iter(out))
    print(f"\nExample {k[:10]}:")
    for t in out[k]["turns"][:3]:
        print(f"  Turn {t['turn']} slot {t['slot']}: {t['instruction'][:110]}")


if __name__ == "__main__":
    main()
