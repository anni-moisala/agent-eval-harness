"""openmp/exercises/06-heat-reduction -- fix stale-host-read bug in per-quadrant averages.

Same physics/parameters as openmp_04_heat, so the final u[511,511] value is
checkable against that same reference. The quadrant-average bug is specific
though: `u` lives in a `target data` region for the whole time loop, so a
plain host-side loop over `u[]` (the stub) reads a copy that's never synced
back until the region exits -- it just reprints the same stale numbers every
checkpoint. Rather than needing a separately-cached golden reference for the
per-iteration averages, this checks the behavioral signature of that bug
directly: are the printed quadrant averages actually changing over time?
"""

import os
import re
import lib

EXERCISE_PATH = "openmp/exercises/06-heat-reduction"
FILES = ["heat.c", "heat_helper_functions.h"]

EXPECTED_U = -2.522502
TOL = 5e-3


def build(workdir):
    out = os.path.join(workdir, "heat")
    path, _stderr = lib.compile_nvc([os.path.join(workdir, "heat.c")], out)
    return {"gpu": path}


def run(built):
    cwd = os.path.dirname(built["gpu"])  # binary writes relative-path .bin snapshots
    rc, out, err = lib.run_gpu_nvhpc(built["gpu"], args=["1024", "500", "1"], timeout=200, cwd=cwd)
    return {"gpu": (rc, out, err)}


def grade(results, workdir):
    rc, out, err = results["gpu"]
    if rc != 0:
        return {"A": (False, f"run exited {rc}: {err.strip()[:200]}"),
                 "B": (False, "skipped, run failed")}

    m = re.search(r'u\[\d+,\d+\]\s*=\s*([-\d.eE+]+)', out)
    u_val = float(m.group(1)) if m else None
    u_ok = u_val is not None and lib.close(u_val, EXPECTED_U, tol=TOL)

    avg_tuples = [tuple(float(x) for x in row)
                  for row in re.findall(r'^\d{6}:\s+([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)',
                                         out, re.MULTILINE)]
    varying_ok = len(avg_tuples) >= 2 and len(set(avg_tuples)) > 1

    a_ok = u_ok and varying_ok
    a_detail = (f"u[511,511]={u_val!r} (expect {EXPECTED_U}, tol {TOL}); "
                f"quadrant-average checkpoints={avg_tuples} "
                f"({'vary over time' if varying_ok else 'STALE -- identical every checkpoint, classic bug signature'})")

    src = lib.read(os.path.join(workdir, "heat.c"))
    reduction_count = lib.grep_count(src, r'reduction\s*\(\s*\+\s*:\s*avg\[\d\]\s*\)')
    b_ok = reduction_count >= 4
    b_detail = f"'reduction(+:avg[k])' occurrences={reduction_count} (need >=4, one per quadrant)"

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
