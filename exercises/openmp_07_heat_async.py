"""openmp/exercises/07-heat-async -- make kernel launches async with nowait+depend, same result.

Stub here is 06-heat-reduction's fixed solution (reduction bug already
fixed). The only diff to the real solution is `nowait`/`depend` clauses on
the target pragmas -- by design these must NOT change the numerical result
(that's the whole point of "async but still correct"), so Tier A can't
distinguish "added async" from "did nothing"; Tier B checks for the clauses
directly.
"""

import os
import re
import lib

EXERCISE_PATH = "openmp/exercises/07-heat-async"
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
    a_ok = u_val is not None and lib.close(u_val, EXPECTED_U, tol=TOL)
    a_detail = f"u[511,511]={u_val!r} (expect {EXPECTED_U}, tol {TOL}) -- must be unchanged by going async"

    src = lib.read(os.path.join(workdir, "heat.c"))
    nowait_count = lib.grep_count(src, r'\bnowait\b')
    depend_count = lib.grep_count(src, r'\bdepend\s*\(')
    b_ok = nowait_count >= 1 and depend_count >= 1
    b_detail = f"nowait={nowait_count} (need >=1), depend(...)={depend_count} (need >=1)"

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
