"""openmp/exercises/16-heat-multi-gpu -- per-rank device assignment + halo exchange.

Domain-decomposed heat equation across 2 MPI ranks/GPUs. The stub has
scatter/gather already wired up but two TODOs missing: (1) each rank always
uses the default GPU (device 0) instead of one GPU per rank, and (2) there's
no halo exchange between ranks' subdomains at all, so the stencil reads
uninitialized/stale boundary rows. Halo exchange correctly implemented
should reproduce the exact same global result as the single-GPU heat
exercises (domain decomposition doesn't change the physics), so the same
u[511,511] reference applies.

Tier B additionally guards against a partial fix: an agent could get the
numeric result right while still hardcoding every rank onto GPU 0 (halo
exchange alone is sufficient for correctness; multi-GPU utilization is a
separate, not-automatically-implied claim) -- checked separately from
correctness.
"""

import os
import re
import lib

EXERCISE_PATH = "openmp/exercises/16-heat-multi-gpu"
FILES = ["heat.c", "heat_helper_functions.h"]

EXPECTED_U = -2.522502
TOL = 5e-3


def build(workdir):
    out = os.path.join(workdir, "heat")
    lib.compile_mpi_nvc([os.path.join(workdir, "heat.c")], out)
    return {"gpu": out}


def run(built):
    cwd = os.path.dirname(built["gpu"])
    rc, out, err = lib.run_mpi_gpu_nvhpc(built["gpu"], args=["1024", "500", "1"], ngpus=2, timeout=200, cwd=cwd)
    return {"gpu": (rc, out, err)}


def grade(results, workdir):
    rc, out, err = results["gpu"]
    if rc != 0:
        return {"A": (False, f"run exited {rc}: {err.strip()[:200]}"),
                 "B": (False, "skipped, run failed")}

    m = re.search(r'u\[\d+,\d+\]\s*=\s*([-\d.eE+]+)', out)
    u_val = float(m.group(1)) if m else None
    a_ok = u_val is not None and lib.close(u_val, EXPECTED_U, tol=TOL)
    a_detail = f"u[511,511]={u_val!r} (expect {EXPECTED_U}, tol {TOL}; should match single-GPU heat exercises)"

    src = lib.read(os.path.join(workdir, "heat.c"))
    sendrecv_count = lib.grep_count(src, r'MPI_Sendrecv\s*\(')
    device_assign_count = lib.grep_count(src, r'omp_set_default_device\s*\(\s*rank\s*%')
    b_ok = sendrecv_count >= 1 and device_assign_count >= 1
    b_detail = (f"MPI_Sendrecv(...) in halo exchange={sendrecv_count} (need >=1), "
                f"omp_set_default_device(rank % ...) present={device_assign_count} (need >=1, "
                f"not hardcoded to a single device)")

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
