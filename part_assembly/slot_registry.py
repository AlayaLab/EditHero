"""Object slot division registry -- once divided, store it; don't make me review it again.

Unpacking an object's structure, reading descriptions, viewing renders, making adjustments back and forth --
this entire process is done by me (LLM) personally. Each object takes several rounds.
**Must be stored after completion**; next time for replacement or retrieval, look up the table directly.
Only walk through the full process if not found.

One object can have **multiple divisions** with different granularities:

    coarse   head / left arm / right arm / left leg / right leg          -- swap an entire arm
    fine     head / left upper arm / left forearm / left hand / ...      -- swap one hand
    finer    ... / left thumb / left index finger / ...                  -- swap one finger

When the same object is retrieved by different tasks, the required granularity differs, so index by
(object, granularity), not just one division per object.

Each division comes with `verified`: only set to True if rendering confirms the grouping is correct.
Unconfirmed ones should not be used directly.

The registry lives in `<dataset>/slot_registry.json`, format:

    { "<object_id>": [ {granularity, groups, anchors, verified, preview, method, notes, ts}, ... ] }
"""
import os, json, argparse, datetime

from local_paths import DATA_ROOT as D
PATH = os.path.join(D, "slot_registry.json")


def _load():
    return json.load(open(PATH)) if os.path.isfile(PATH) else {}


def _save(reg):
    tmp = PATH + ".tmp"
    json.dump(reg, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, PATH)


def get(oid, granularity=None, verified_only=True):
    """Get one object's slot divisions. When granularity is None, return all entries meeting the conditions."""
    items = _load().get(oid, [])
    out = [d for d in items
           if (granularity is None or d.get("granularity") == granularity)
           and (not verified_only or d.get("verified"))]
    return out


def get_one(oid, granularity=None, verified_only=True):
    """Get one set. If granularity is given, get that set; otherwise get the coarsest one (fewest slots)."""
    out = get(oid, granularity, verified_only)
    if not out:
        return None
    return min(out, key=lambda d: len(d.get("groups", {})))


def _put_unlocked(oid, granularity, groups, anchors=None, verified=False,
        preview=None, method="llm+render", notes="", roles=None):
    """Store one slot division. The same (object, granularity) will be overwritten -- re-dividing means updating it."""
    reg = _load()
    items = [d for d in reg.get(oid, []) if d.get("granularity") != granularity]
    items.append(dict(
        granularity=granularity,
        groups={k: list(v) for k, v in groups.items()},
        anchors=sorted(anchors or [k for k in groups if k.startswith("anchor")]),
        # roles: {slot: {queries: [...], accepts: [...], rejects: [...]}} -- what this slot can accept.
        # Criteria must be attached to the **slot** itself, not the candidates: placing a wheel on a leg is reasonable
        # (robots have wheels for feet anyway), but placing it on an arm is absurd.
        #
        # `queries` is several **example sentences**, encoded to vectors during retrieval for comparison;
        # this is what's actually used now.
        # `accepts` is the old vocabulary, kept as a fallback when vector encoding hasn't been run -- it recognizes text,
        # not meaning, so originally the "head" slot had to manually list 37 synonyms; now 4 sentences suffice.
        # `rejects` still does literal matching: contains words like wall / floor / ceiling that should literally be rejected.
        roles=roles or {},
        verified=bool(verified),
        preview=preview, method=method, notes=notes,
        n_slots=len([k for k in groups if not k.startswith("anchor")]),
        ts=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    ))
    reg[oid] = items
    _save(reg)
    return items[-1]


def mark_verified(oid, granularity, preview=None, notes=""):
    """Call this after confirming the groups are correct via rendering. Unconfirmed divisions cannot be used directly in chains."""
    import fcntl
    with open(PATH + ".lock", "w") as _lk:
        fcntl.flock(_lk, fcntl.LOCK_EX)
        return _mark_verified_unlocked(oid, granularity, preview, notes)


def _mark_verified_unlocked(oid, granularity, preview=None, notes=""):
    reg = _load()
    for d in reg.get(oid, []):
        if d.get("granularity") == granularity:
            d["verified"] = True
            if preview:
                d["preview"] = preview
            if notes:
                d["notes"] = (d.get("notes", "") + " | " + notes).strip(" |")
            _save(reg)
            return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "show", "add", "verify"])
    ap.add_argument("--oid")
    ap.add_argument("--granularity", default="coarse")
    ap.add_argument("--groups", help="add: JSON path for slot groups")
    ap.add_argument("--preview")
    ap.add_argument("--notes", default="")
    ap.add_argument("--verified", action="store_true")
    a = ap.parse_args()

    if a.cmd == "list":
        reg = _load()
        print(f"{len(reg)} objects, {sum(len(v) for v in reg.values())} divisions")
        for oid, items in reg.items():
            for d in items:
                mark = "✓" if d["verified"] else "·"
                print(f"  {mark} {oid[:12]}  {d['granularity']:10s} {d['n_slots']} slots  "
                      f"{','.join(k for k in d['groups'] if k not in d['anchors'])[:60]}")
    elif a.cmd == "show":
        for d in get(a.oid, None, verified_only=False):
            print(json.dumps(d, ensure_ascii=False, indent=1))
    elif a.cmd == "add":
        g = json.load(open(a.groups))
        d = put(a.oid, a.granularity, g, verified=a.verified,
                preview=a.preview, notes=a.notes)
        print(f"Stored {a.oid[:12]} / {a.granularity}: {d['n_slots']} slots, "
              f"anchors {d['anchors']}, verified={d['verified']}")
    else:
        d = mark_verified(a.oid, a.granularity, a.preview, a.notes)
        print("Marked as verified" if d else "Division not found")


def put(*args, **kwargs):
    """Locked version of put: load-modify-save is not locked, multiple concurrent registry agents can
    overwrite each other (reported by W1-REG1). Re-execute the original logic inside exclusive flock; the original logic
    loads itself so the data inside the lock is fresh."""
    import fcntl
    lock_path = PATH + ".lock" if "PATH" in globals() else __file__ + ".lock"
    with open(lock_path, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            return _put_unlocked(*args, **kwargs)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


if __name__ == "__main__":   # Must be after put definition: CLI add once had NameError due to definition order (review #13)
    main()
