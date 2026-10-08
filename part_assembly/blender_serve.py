"""Resident blender rendering worker. Runs inside blender, monitors spool directory for jobs, process never exits.

Why needed: per-round review (serve / step) requires rendering each round. Blender cold start + OptiX
initialization takes ~5 seconds; 12 rounds = 1 minute pure startup overhead. After becoming resident,
each job only pays for "load glb + render 4 frames".

Reuses all rendering logic from blender_kit (`run_one_job` from `render.py`); this is just a dispatch loop
—— run_batch already runs multiple jobs sequentially in one process, proving run_one_job is safely reentrant.

    blender -b --python blender_serve.py -- <render.py base parameters...>
    environment variable BK_SPOOL=<spool directory>

Protocol (all file-based, no network):
    <spool>/job_*.json     one override set from render.py (mesh/out_dir/...)
    <spool>/job_*.done     rendered receipt (content ok / error text)
    <spool>/STOP           appears → exit
"""
import os, sys, json, glob, time, traceback, importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_paths import RENDER_SCRIPT as KIT


def load_render_module():
    spec = importlib.util.spec_from_file_location("bk_render", KIT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["bk_render"] = m
    spec.loader.exec_module(m)
    return m


def main():
    spool = os.environ["BK_SPOOL"]
    os.makedirs(spool, exist_ok=True)
    R = load_render_module()
    import argparse
    ap = argparse.ArgumentParser()
    R.add_args(ap)
    base = ap.parse_args(R.parse_blender_argv())
    print(f"[serve] spool={spool} ready", flush=True)
    idle = 0.0
    while True:
        if os.path.exists(os.path.join(spool, "STOP")):
            print("[serve] STOP, exit", flush=True)
            return
        todo = [p for p in sorted(glob.glob(os.path.join(spool, "job_*.json")))
                if not os.path.exists(p + ".done")]
        if not todo:
            time.sleep(0.2)
            idle += 0.2
            if idle > 3600:                     # no jobs for 1 hour, self-terminate to free GPU overnight
                print("[serve] idle timeout, exit", flush=True)
                return
            continue
        idle = 0.0
        for p in todo:
            try:
                overrides = json.load(open(p))
                # focus: closeup rendering. switches camera target from "entire scene" to "bounding box of one part"
                # (normalized coordinates computed by chain_run._focus). kit's camera always faces
                # scene.center and scales by scene.diag, so just temporarily replace these two values.
                focus = overrides.pop("focus", None)
                t0 = time.time()
                if focus:
                    orig = R.resolve_scene
                    def patched(a, _f=focus, _o=orig):
                        s = _o(a)
                        s.center = tuple(_f["center"])
                        s.diag = float(_f["diag"])
                        return s
                    R.resolve_scene = patched
                    try:
                        R.run_one_job(R._merge(base, overrides))
                    finally:
                        R.resolve_scene = orig
                else:
                    R.run_one_job(R._merge(base, overrides))
                msg = f"ok {time.time()-t0:.1f}s"
            except (SystemExit, Exception) as e:
                traceback.print_exc(limit=2)
                msg = f"error {type(e).__name__}: {e}"
            tmp = p + ".done.tmp"
            open(tmp, "w").write(msg)
            os.replace(tmp, p + ".done")        # atomic write, receiving side won't read partial data
            print(f"[serve] {os.path.basename(p)} -> {msg}", flush=True)


if __name__ == "__main__":
    main()
