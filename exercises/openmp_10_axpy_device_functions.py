"""openmp/exercises/10-axpy-device-functions -- fix a link error by adding `declare target`.

kernels.c defines `axpy()` in a separate compilation unit from main.c, which
calls it inside a `target` region. Without `#pragma omp declare target`
around it, linking fails with an undefined symbol at link time (the device
side never got a compiled version of the function). Grading is two-phase:
the unfixed stub must fail to LINK with that specific error signature; the
fixed version must build+run and reproduce the standard axpy output.

Tier B additionally guards against sidestepping the actual lesson by
inlining axpy() into main.c to dodge the cross-compilation-unit problem
entirely -- it requires kernels.c to still define the function.
"""

import os
import re
import lib

EXERCISE_PATH = "openmp/exercises/10-axpy-device-functions"
FILES = ["c/kernels.c", "c/kernels.h", "c/main.c", "c/axpy_helper_functions.h", "c/Makefile"]

N = 102400


def _y_final(i, frac):
    return i * frac * 100 + 3.0 * (i * frac)


def _expected():
    frac = 1.0 / (N - 1)
    idxs = [0, 1, 2, 3, N - 4, N - 3, N - 2, N - 1]
    return [_y_final(i, frac) for i in idxs]


def build(workdir):
    cdir = os.path.join(workdir, "c")
    lib.make_nvhpc(cdir)  # raises BuildError (with the linker output) if it fails
    return {"gpu": os.path.join(cdir, "main.x")}


def run(built):
    cwd = os.path.dirname(built["gpu"])
    rc, out, err = lib.run_gpu_nvhpc(built["gpu"], cwd=cwd)
    return {"gpu": (rc, out, err)}


def grade(results, workdir):
    rc, out, err = results["gpu"]
    if rc != 0:
        return {"A": (False, f"run exited {rc}: {err.strip()[:200]}"),
                 "B": (False, "skipped, run failed")}

    y_lines = [l for l in out.splitlines() if l.strip().startswith("y =")]
    a_ok = False
    a_detail = f"found {len(y_lines)} 'y = ...' lines (need 2: Input echo + Output)"
    if len(y_lines) >= 2:
        nums = [float(n) for n in re.findall(r'-?\d+\.\d+', y_lines[1])]
        expected = _expected()
        a_ok = len(nums) == 8 and all(lib.close(a, b, tol=5e-4) for a, b in zip(nums, expected))
        a_detail = f"Output y values={nums} (expect {expected}, tol 5e-4)"

    kernels_src = lib.read(os.path.join(workdir, "c", "kernels.c"))
    declare_target_count = lib.grep_count(kernels_src, r'#pragma\s+omp\s+declare\s+target')
    axpy_def_count = lib.grep_count(kernels_src, r'\bdouble\s+axpy\s*\(')
    b_ok = declare_target_count >= 1 and axpy_def_count >= 1
    b_detail = (f"kernels.c: 'declare target'={declare_target_count} (need >=1), "
                f"axpy() still defined here={axpy_def_count} (need >=1, "
                f"guards against inlining it into main.c to dodge the actual fix)")

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
