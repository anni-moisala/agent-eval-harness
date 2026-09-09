"""openmp/exercises/04-heat -- offload the heat-equation stencil to GPU with a persistent data region.

solution/ has several variants (heat-1, heat-2, heat-2a, heat-2b); heat-2 is
the one the README calls "fastest" (persistent target data region around
the whole time loop), so it's the canonical reference here, renamed to
heat.c to match the stub's filename.

Same gaming pattern as openmp_02/03_axpy: the CPU-only loop is numerically
identical, so Tier A alone can't tell offloaded from not.
"""

import os
import re
import lib

EXERCISE_PATH = "openmp/exercises/04-heat"
FILES = ["heat.c", "heat_helper_functions.h"]
SOLUTION_FILE_RENAMES = {"heat.c": "heat-2.c"}

EXPECTED_U = -2.522502
TOL = 5e-3


def build(workdir):
    out = os.path.join(workdir, "heat")
    path, _stderr = lib.compile_nvc([os.path.join(workdir, "heat.c")], out)
    return {"gpu": path}


def run(built):
    # n=1024 niter=500 nrep=1 -- matches the README's quoted reference run.
    # cwd: the binary writes relative-path u_initial.bin/u_final.bin.
    cwd = os.path.dirname(built["gpu"])
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
    a_detail = f"u[511,511]={u_val!r} (expect {EXPECTED_U}, tol {TOL})"

    src = lib.read(os.path.join(workdir, "heat.c"))
    target_data_count = lib.grep_count(src, r'#pragma\s+omp\s+target\s+data')
    target_count = lib.grep_count(src, r'#pragma\s+omp\s+target\b(?!\s+data)')
    teams_count = lib.grep_count(src, r'#pragma\s+omp\s+(target\s+)?teams\s+distribute\s+parallel\s+for')
    b_ok = target_data_count >= 1 and target_count >= 1 and teams_count >= 1
    b_detail = (f"target data={target_data_count} (need >=1), "
                f"target={target_count} (need >=1), "
                f"teams distribute parallel for={teams_count} (need >=1)")

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
