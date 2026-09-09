"""openmp/exercises/05-reduction-sum -- fix a race condition with `reduction(+:total)`.

The bug is a data race (many GPU threads incrementing `total` unsynchronized),
so a single run can occasionally land close to the right answer by luck.
The harness runs the binary twice and requires *both* runs to match the
independently-computed reference sum(sin(i)) for i in [0, n) -- a race
losing most updates reliably misses by far more than floating-point
reduction-order noise would.
"""

import os
import re
import math
import lib

EXERCISE_PATH = "openmp/exercises/05-reduction-sum"
FILES = ["sum.c"]

N = 100000
EXPECTED_SUM = math.fsum(math.sin(float(i)) for i in range(N))
TOL = 1e-3


def build(workdir):
    out = os.path.join(workdir, "sum")
    path, _stderr = lib.compile_nvc([os.path.join(workdir, "sum.c")], out)
    return {"gpu": path}


def run(built):
    r1 = lib.run_gpu_nvhpc(built["gpu"])
    r2 = lib.run_gpu_nvhpc(built["gpu"])
    return {"run1": r1, "run2": r2}


def _parse_sum(out):
    m = re.search(r'Sum:\s*([-\d.eE+]+)', out)
    return float(m.group(1)) if m else None


def grade(results, workdir):
    vals = []
    for key in ("run1", "run2"):
        rc, out, err = results[key]
        if rc != 0:
            return {"A": (False, f"{key} exited {rc}: {err.strip()[:200]}"),
                     "B": (False, "skipped, run failed")}
        vals.append(_parse_sum(out))

    a_ok = all(v is not None and lib.close(v, EXPECTED_SUM, tol=TOL) for v in vals)
    a_detail = f"two independent runs gave {vals} (expect both within {TOL} of {EXPECTED_SUM!r})"

    src = lib.read(os.path.join(workdir, "sum.c"))
    reduction_count = lib.grep_count(src, r'reduction\s*\(\s*\+\s*:\s*total\s*\)')
    b_ok = reduction_count >= 1
    b_detail = f"'reduction(+:total)' occurrences={reduction_count}"

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
