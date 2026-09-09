"""openmp/exercises/08-heat-io -- fix stale periodic I/O snapshots.

`u` lives in a `target data` region for the whole time loop and is only
synced back to host on region exit; a plain `write_array(..., u, ...)`
inside the loop (the stub) writes whatever `u` had on *entry* to the region
(i.e. byte-identical to u_initial.bin) at every checkpoint, since nothing
ever refreshes the host copy mid-loop. The fix adds a `target update from`
right before writing. Rather than deriving a golden per-iteration reference,
this checks that behavioral signature directly: is u_001000.bin still
byte-identical to u_initial.bin, or did it actually change?

Uses niter=1500 (not the exercise's various defaults) so exactly one
intermediate checkpoint (it=1000) fires before the run ends, distinct from
the final write.
"""

import hashlib
import os
import lib

EXERCISE_PATH = "openmp/exercises/08-heat-io"
FILES = ["heat.c", "heat_helper_functions.h"]
SOLUTION_FILE_RENAMES = {"heat.c": "heat-1.c"}


def build(workdir):
    out = os.path.join(workdir, "heat")
    path, _stderr = lib.compile_nvc([os.path.join(workdir, "heat.c")], out)
    return {"gpu": path}


def run(built):
    # cwd=dirname(binary): the binary writes relative-path .bin snapshots,
    # which must land in the workdir (also where the binary itself lives)
    # so grade() can read them back, not wherever the harness process runs from.
    cwd = os.path.dirname(built["gpu"])
    rc, out, err = lib.run_gpu_nvhpc(built["gpu"], args=["1024", "1500", "1"], timeout=200, cwd=cwd)
    return {"gpu": (rc, out, err)}


def _md5(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def grade(results, workdir):
    rc, out, err = results["gpu"]
    if rc != 0:
        return {"A": (False, f"run exited {rc}: {err.strip()[:200]}"),
                 "B": (False, "skipped, run failed")}

    initial_hash = _md5(os.path.join(workdir, "u_initial.bin"))
    checkpoint_hash = _md5(os.path.join(workdir, "u_001000.bin"))
    a_ok = initial_hash is not None and checkpoint_hash is not None and initial_hash != checkpoint_hash
    a_detail = (f"u_initial.bin md5={initial_hash}, u_001000.bin md5={checkpoint_hash} -- "
                + ("captured fresh data at checkpoint" if a_ok else
                   "IDENTICAL to initial snapshot -- classic stale-host-read bug"))

    src = lib.read(os.path.join(workdir, "heat.c"))
    update_count = lib.grep_count(src, r'#pragma\s+omp\s+target\s+update\s+from\s*\(\s*u\b')
    b_ok = update_count >= 1
    b_detail = f"'target update from(u...)' occurrences={update_count} (need >=1)"

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
