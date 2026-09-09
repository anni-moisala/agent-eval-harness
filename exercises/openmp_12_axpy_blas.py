"""openmp/exercises/12-axpy-blas -- fix cuBLAS device-pointer interop bug.

Same axpy math/output as openmp_02, but computed via a cuBLAS `daxpy` call
instead of a hand-written kernel. The bug is passing host pointers straight
to the BLAS call instead of the device pointers OpenMP is managing; the fix
is `#pragma omp target data use_device_ptr(x, y)` around the call. The
broken version either gives wrong numbers or crashes -- Tier A already
discriminates that -- but an agent could also "fix" it by dropping the BLAS
call entirely and hand-rolling axpy on GPU, which would give correct output
without actually fixing the interop bug being taught here. Tier B guards
against that by requiring the BLAS call and the fix to still be present.
"""

import os
import re
import lib

EXERCISE_PATH = "openmp/exercises/12-axpy-blas"
FILES = ["axpy.c", "axpy_helper_functions.h", "gpublas.h"]

N = 102400


def _y_final(i, frac):
    return i * frac * 100 + 3.0 * (i * frac)


def _expected():
    frac = 1.0 / (N - 1)
    idxs = [0, 1, 2, 3, N - 4, N - 3, N - 2, N - 1]
    return [_y_final(i, frac) for i in idxs]


def build(workdir):
    out = os.path.join(workdir, "axpy")
    path, _stderr = lib.compile_nvc(
        [os.path.join(workdir, "axpy.c")], out,
        extra_flags=["-mp=gpu", "-O3", "-gpu=cc90", "-Wall", "-lcublas"])
    return {"gpu": path}


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
    a_detail = f"found {len(y_lines)} 'y = ...' lines (need 2)"
    if len(y_lines) >= 2:
        nums = [float(n) for n in re.findall(r'-?\d+\.\d+', y_lines[1])]
        expected = _expected()
        a_ok = len(nums) == 8 and all(lib.close(a, b, tol=5e-4) for a, b in zip(nums, expected))
        a_detail = f"Output y values={nums} (expect {expected}, tol 5e-4)"

    src = lib.read(os.path.join(workdir, "axpy.c"))
    use_device_ptr_count = lib.grep_count(src, r'use_device_ptr\s*\(')
    blas_call_count = lib.grep_count(src, r'blas_daxpy\s*\(')
    b_ok = use_device_ptr_count >= 1 and blas_call_count >= 1
    b_detail = (f"use_device_ptr(...)={use_device_ptr_count} (need >=1), "
                f"blas_daxpy(...) call still present={blas_call_count} (need >=1, "
                f"guards against dropping BLAS entirely to dodge the interop fix)")

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
