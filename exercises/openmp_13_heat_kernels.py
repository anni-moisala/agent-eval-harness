"""openmp/exercises/13-heat-kernels -- pass real device pointers to a custom CUDA kernel.

The Makefile only ever compiles heat.c + kernels.cu into heat.x (kernels.c
in this exercise dir is unused reference material, not part of the actual
build) -- the whole exercise is the *one* missing pragma in heat.c:
`#pragma omp target data use_device_ptr(u, unew)` before calling the custom
`evolve()` CUDA kernel. Without it, the kernel receives host pointers and
silently no-ops on interior cells, leaving the -100.0 boundary value; with
it, u[8191,8191] comes out to -1.783901. Since kernels.cu/Makefile aren't
part of the graded FILES, an agent can't sidestep the fix by dropping the
custom kernel for a native OpenMP loop -- the harness always rebuilds with
the real kernels.cu regardless of what the candidate contains.
"""

import os
import re
import lib

EXERCISE_PATH = "openmp/exercises/13-heat-kernels"
FILES = ["c/heat.c", "c/kernels.cu", "c/kernels.h", "c/heat_helper_functions.h", "c/Makefile"]

EXPECTED_U = -1.783901
BROKEN_U = -100.000000
TOL = 5e-3


def build(workdir):
    cdir = os.path.join(workdir, "c")
    lib.make_nvhpc(cdir)
    return {"gpu": os.path.join(cdir, "heat.x")}


def run(built):
    cwd = os.path.dirname(built["gpu"])
    # nrep=1: the repeated runs in the exercise's own default are identical,
    # niter/n match the README's quoted reference invocation
    rc, out, err = lib.run_gpu_nvhpc(built["gpu"], args=["16384", "1000", "1"], timeout=280, cwd=cwd)
    return {"gpu": (rc, out, err)}


def grade(results, workdir):
    rc, out, err = results["gpu"]
    if rc != 0:
        return {"A": (False, f"run exited {rc}: {err.strip()[:200]}"),
                 "B": (False, "skipped, run failed")}

    m = re.search(r'u\[\d+,\d+\]\s*=\s*([-\d.eE+]+)', out)
    u_val = float(m.group(1)) if m else None
    a_ok = u_val is not None and lib.close(u_val, EXPECTED_U, tol=TOL)
    a_detail = f"u[8191,8191]={u_val!r} (expect {EXPECTED_U}, tol {TOL}; broken value would be {BROKEN_U})"

    src = lib.read(os.path.join(workdir, "c", "heat.c"))
    use_device_ptr_count = lib.grep_count(src, r'use_device_ptr\s*\(\s*u\s*,\s*unew\s*\)')
    b_ok = use_device_ptr_count >= 1
    b_detail = f"'use_device_ptr(u, unew)' occurrences={use_device_ptr_count} (need >=1)"

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
