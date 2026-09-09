"""openmp/exercises/03-axpy-data -- wrap init+axpy in one structured `target data` region.

Builds on 02-axpy: the stub here IS exercise 02's solution (init on host,
axpy on GPU, no data region). Task: offload init too, and wrap both kernels
in a single `target data` region with `target update from` so host prints
still see fresh data. Correctness/output is unchanged from exercise 02 (same
math) -- this exercise is purely about *how* data moves, so Tier B checks
for the specific pragmas rather than relying on output alone.
"""

import os
import re
import lib

EXERCISE_PATH = "openmp/exercises/03-axpy-data"
FILES = ["axpy.c", "axpy_helper_functions.h"]

N = 102400


def _expected(fn):
    frac = 1.0 / (N - 1)
    idxs = [0, 1, 2, 3, N - 4, N - 3, N - 2, N - 1]
    return [fn(i, frac) for i in idxs]


_EXPECTED_X = _expected(lambda i, frac: i * frac)
_EXPECTED_Y_INIT = _expected(lambda i, frac: i * frac * 100)
_EXPECTED_Y_FINAL = _expected(lambda i, frac: i * frac * 100 + 3.0 * i * frac)


def build(workdir):
    out = os.path.join(workdir, "axpy")
    path, _stderr = lib.compile_nvc([os.path.join(workdir, "axpy.c")], out)
    return {"gpu": path}


def run(built):
    rc, out, err = lib.run_gpu_nvhpc(built["gpu"])
    return {"gpu": (rc, out, err)}


def _nums(line):
    return [float(n) for n in re.findall(r'-?\d+\.\d+', line)]


def grade(results, workdir):
    rc, out, err = results["gpu"]
    if rc != 0:
        return {"A": (False, f"run exited {rc}: {err.strip()[:200]}"),
                 "B": (False, "skipped, run failed")}

    x_lines = [l for l in out.splitlines() if l.strip().startswith("x =")]
    y_lines = [l for l in out.splitlines() if l.strip().startswith("y =")]
    a_ok = False
    a_detail = f"found {len(x_lines)} 'x =' line(s) (need 1), {len(y_lines)} 'y =' line(s) (need 2)"
    if len(x_lines) == 1 and len(y_lines) == 2:
        x_vals, y_init_vals, y_final_vals = _nums(x_lines[0]), _nums(y_lines[0]), _nums(y_lines[1])
        checks = [
            len(x_vals) == 8 and all(lib.close(a, b, 5e-4) for a, b in zip(x_vals, _EXPECTED_X)),
            len(y_init_vals) == 8 and all(lib.close(a, b, 5e-4) for a, b in zip(y_init_vals, _EXPECTED_Y_INIT)),
            len(y_final_vals) == 8 and all(lib.close(a, b, 5e-4) for a, b in zip(y_final_vals, _EXPECTED_Y_FINAL)),
        ]
        a_ok = all(checks)
        a_detail = (f"x(Input)={x_vals} (expect {_EXPECTED_X}), "
                    f"y(Input)={y_init_vals} (expect {_EXPECTED_Y_INIT}), "
                    f"y(Output)={y_final_vals} (expect {_EXPECTED_Y_FINAL}), tol 5e-4")

    src = lib.read(os.path.join(workdir, "axpy.c"))
    target_data_count = lib.grep_count(src, r'#pragma\s+omp\s+target\s+data')
    target_teams_count = lib.grep_count(src, r'#pragma\s+omp\s+target\s+teams\s+distribute\s+parallel\s+for')
    target_update_count = lib.grep_count(src, r'#pragma\s+omp\s+target\s+update')
    b_ok = target_data_count >= 1 and target_teams_count >= 2 and target_update_count >= 1
    b_detail = (f"target data={target_data_count} (need >=1), "
                f"target teams distribute parallel for={target_teams_count} (need >=2: init+axpy), "
                f"target update={target_update_count} (need >=1)")

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
