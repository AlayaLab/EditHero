"""Visible change metric: quantifies pixel-level changes from a replacement operation.

Primary quality requirement: **visible, human-recognizable changes on screen.**
Previous batches had cases like "wooden spoke replaced with wooden spoke" or "wagon bed blocked by chest"—
changes that looked the same after the operation. Manual review was tedious and subjective.
This module automates that gate:

    metric = max_over_views( changed_pixels / object_pixels )

  - Compare turn(k-1) and turn(k) at the same camera position (RGBA composited to white background);
  - Changed pixels: any channel diff > PIX_TAU (filters antialiasing and shadow noise);
  - Object pixels: union of alpha masks from both frames (avoids using full image as denominator
    when object occupies a small screen region);
  - Max over 4 views: when reviewing a filmstrip, all 4 views are visible; if any view shows change,
    the operation was not pointless. Only when all 4 views show no change is it truly imperceptible.

Threshold VIS_TAU is not arbitrary: calibrate mode re-computes all turns from existing chains,
sorts by value, and displays side-by-side comparisons for human (or VLM) threshold selection.
See PIPELINE.md, "visible change gate" section.

Known limitation (observed in D2 evaluation): rendering normalizes by bounding box of the entire object.
When adding a new part, the bbox expands, causing the whole object to rescale and shift on screen.
Every pixel changes—even a thin black wire can score 0.93. Therefore: **use this metric as a lower bound only.**
Low scores guarantee invisibility; high scores do not guarantee visibility. High-scoring turns still need manual review.

Calibration: each turn uses sphere-based framing (chain_run.sphere_frame)
with view cone tangent to sphere, not bbox normalization. Comparisons re-render the prior turn's state
in the current turn's camera (img/turnKK_prev); both images now use the same camera, so the metric measures
only the replacement itself. Object pixels still form the denominator; hard gate is 10%, soft gate is 30%.
Previous thresholds 0.06/0.12 were calibrated on re-normalized images where bbox changes were counted
as pixel changes, making those thresholds artificially low.
    python change_meter.py calibrate --dirs <chain_dirs...> --out <dir>   # define thresholds
    python change_meter.py one --dir <chain_dir> --turn K                # single turn
"""
import os, json, glob, argparse
import numpy as np
from PIL import Image

PIX_TAU = 14      # per-pixel channel diff threshold; filters render noise
# Calibrated on 144 turns across 2 batches (lowest boundaries checked by eye):
#   < 0.06  imperceptible to human eye (all identical wheel swaps in racing chains fall here) → auto veto
#   0.06–0.12 visible if you look for it → flag as LOW VIS in review UI, veto by default,
#            pass only if visually clear despite low score (high contrast, silhouette change, etc.)
#   > 0.12  immediately visible
VIS_TAU = 0.10    # hard gate: < 10% of object pixels changed → auto veto (same-camera basis)
VIS_SOFT = 0.30   # soft gate: < 30% → flag as LOW VIS in review UI, veto by default, pass if visually clear
VIEWS = 4


def _load(p):
    im = np.asarray(Image.open(p).convert("RGBA"), np.float32)
    a = im[..., 3:] / 255.0
    rgb = im[..., :3] * a + 255.0 * (1 - a)      # composite to white background
    return rgb, (im[..., 3] > 8)


def change_frac(chain_dir, turn, img_dir=None):
    """Returns (metric, per-view details). Returns (None, []) if images are missing."""
    img = img_dir or os.path.join(chain_dir, "img")
    fracs = []
    for v in range(VIEWS):
        # Prefer "prior turn state, current camera" render (turnKK_prev): same camera in both images.
        # Fall back to prior turn's own render for older chains.
        prev_dir = os.path.join(img, f"turn{turn:02d}_prev")
        if not os.path.isdir(prev_dir):
            prev_dir = os.path.join(img, f"turn{turn-1:02d}")
        pa = os.path.join(prev_dir, f"f{v:04d}.png")
        pb = os.path.join(img, f"turn{turn:02d}", f"f{v:04d}.png")
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            return None, []
        if os.path.getsize(pa) < 1024 or os.path.getsize(pb) < 1024:
            return None, []          # Incomplete PNG (worker crash); prevent serve from crashing (review #9)
        try:
            ra, ma = _load(pa)
            rb, mb = _load(pb)
        except OSError:
            return None, []
        changed = (np.abs(ra - rb).max(-1) > PIX_TAU)
        denom = max(int((ma | mb).sum()), 1)
        fracs.append(float(changed.sum()) / denom)
    return max(fracs), [round(f, 4) for f in fracs]


def cmd_calibrate(dirs, out, img_sub="img"):
    os.makedirs(out, exist_ok=True)
    rows = []
    for d in dirs:
        rp = os.path.join(d, "place_report.json")
        if not os.path.isfile(rp):
            continue
        for r in json.load(open(rp)):
            m, per = change_frac(d, r["turn"], img_dir=os.path.join(d, img_sub))
            if m is None:
                continue
            rows.append(dict(dir=d, turn=r["turn"], metric=round(m, 4), per=per,
                             slot=r["slot"], added=r["added"][:60]))
    rows.sort(key=lambda r: r["metric"])
    json.dump(rows, open(os.path.join(out, "metrics.json"), "w"),
              ensure_ascii=False, indent=1)
    for r in rows:
        print(f"{r['metric']:.4f}  turn{r['turn']:02d}  {os.path.basename(r['dir'])[:8]}"
              f"  [{r['slot']}] {r['added'][:50]}")

    # Percentile overview + composite comparison of lowest-scoring turns for manual threshold selection
    ms = [r["metric"] for r in rows]
    for q in (0, 5, 10, 25, 50, 90):
        print(f"  p{q:02d} = {np.percentile(ms, q):.4f}")
    S = 220
    low = rows[:14]
    sh = Image.new("RGB", (S * 4 + 12, (S + 34) * len(low)), (255, 255, 255))
    from PIL import ImageDraw
    dr = ImageDraw.Draw(sh)
    for i, r in enumerate(low):
        y = i * (S + 34)
        dr.text((4, y + 2), f"metric={r['metric']:.4f}  turn{r['turn']:02d} "
                            f"{os.path.basename(r['dir'])[:8]} [{r['slot']}]", fill=(0, 0, 0))
        best_v = int(np.argmax(r["per"]))
        for j, (t, v) in enumerate([(r["turn"] - 1, 0), (r["turn"], 0),
                                    (r["turn"] - 1, best_v), (r["turn"], best_v)]):
            p = os.path.join(r["dir"], img_sub, f"turn{t:02d}", f"f{v:04d}.png")
            im = Image.open(p).convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            sh.paste(Image.alpha_composite(bg, im).convert("RGB").resize((S, S)),
                     (j * S + (12 if j >= 2 else 0), y + 18))
    p = os.path.join(out, "lowest.png")
    sh.save(p)
    print(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "one"])
    ap.add_argument("--dirs", nargs="*", default=[])
    ap.add_argument("--dir")
    ap.add_argument("--turn", type=int)
    ap.add_argument("--out", default="calib")
    ap.add_argument("--img-sub", default="img", help="rendering subdirectory under chain dir (use img_fixed for fixed-camera re-renders)")
    a = ap.parse_args()
    if a.cmd == "one":
        m, per = change_frac(a.dir, a.turn, img_dir=os.path.join(a.dir, a.img_sub))
        print(json.dumps(dict(metric=m, per_view=per)))
    else:
        dirs = []
        for d in a.dirs:
            dirs += sorted(glob.glob(d)) if any(c in d for c in "*?") else [d]
        cmd_calibrate(dirs, a.out, img_sub=a.img_sub)


if __name__ == "__main__":
    main()
