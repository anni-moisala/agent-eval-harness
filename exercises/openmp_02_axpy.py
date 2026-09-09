"""openmp/exercises/02-axpy -- add OpenMP GPU-offload directives to axpy.

The axpy math is elementwise and identical whether or not it's offloaded, so
the un-offloaded stub already prints byte-identical output to the solution.
Tier B (an `omp target ... parallel for` pragma actually present) is
load-bearing here, same reasoning as the kokkos "convert serial to parallel"
family.
"""

import os
import re
import lib

EXERCISE_PATH = "openmp/exercises/02-axpy"
FILES = ["axpy.c", "axpy_helper_functions.h"]

N = 102400


def _expected_line(label, values_fn):
    frac = 1.0 / (N - 1)
    idxs = [0, 1, 2, 3, N - 4, N - 3, N - 2, N - 1]
    vals = [values_fn(i, frac) for i in idxs]
    return vals


def _y_final(i, frac):
    x = i * frac
    y0 = i * frac * 100
    return y0 + 3.0 * x


def build(workdir):
    out = os.path.join(workdir, "axpy")
    path, _stderr = lib.compile_nvc([os.path.join(workdir, "axpy.c")], out)
    return {"gpu": path}


def run(built):
    rc, out, err = lib.run_gpu_nvhpc(built["gpu"])
    return {"gpu": (rc, out, err)}


def grade(results, workdir):
    rc, out, err = results["gpu"]
    if rc != 0:
        return {"A": (False, f"run exited {rc}: {err.strip()[:200]}"),
                 "B": (False, "skipped, run failed")}

    # second "y = ..." block is the post-axpy Output section
    y_lines = [l for l in out.splitlines() if l.strip().startswith("y =")]
    a_ok = False
    a_detail = f"found {len(y_lines)} 'y = ...' lines (need 2: Input echo + Output)"
    if len(y_lines) >= 2:
        nums = [float(n) for n in re.findall(r'-?\d+\.\d+', y_lines[1])]
        expected = _expected_line("y", _y_final)
        a_ok = len(nums) == 8 and all(lib.close(a, b, tol=5e-4) for a, b in zip(nums, expected))
        a_detail = f"Output y values={nums} (expect {expected}, tol 5e-4)"

    src = lib.read(os.path.join(workdir, "axpy.c"))
    target_count = lib.grep_count(src, r'#pragma\s+omp\s+target\s+teams\s+distribute\s+parallel\s+for')
    b_ok = target_count >= 1
    b_detail = f"'omp target teams distribute parallel for' occurrences={target_count}"

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
