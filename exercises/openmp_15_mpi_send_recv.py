"""openmp/exercises/15-mpi-send-recv -- no code changes; GPU-aware MPI send/recv already correct.

README states explicitly "no code changes needed" -- the provided code is
already the answer, demonstrating use_device_ptr with MPI_Send/Recv plus
per-rank device assignment. Grading just confirms it actually runs correctly
with 2 GPU-bound ranks: rank 0 fills a buffer with 42.0 and sends it,
rank 1 receives it. Print order between ranks is non-deterministic, so each
rank's own line is matched independently rather than assuming an order.
"""

import os
import re
import lib

EXERCISE_PATH = "openmp/exercises/15-mpi-send-recv"
FILES = ["mpi_send_and_recv.c"]
SOLUTION_FILE_SOURCES = {"mpi_send_and_recv.c": "openmp/exercises/15-mpi-send-recv/mpi_send_and_recv.c"}
NO_UNSOLVED_STUB = True


def build(workdir):
    out = os.path.join(workdir, "mpi_send_and_recv")
    lib.compile_mpi_nvc([os.path.join(workdir, "mpi_send_and_recv.c")], out)
    return {"gpu": out}


def run(built):
    cwd = os.path.dirname(built["gpu"])
    rc, out, err = lib.run_mpi_gpu_nvhpc(built["gpu"], ngpus=2, cwd=cwd)
    return {"gpu": (rc, out, err)}


def grade(results, workdir):
    rc, out, err = results["gpu"]
    if rc != 0:
        return {"A": (False, f"run exited {rc}: {err.strip()[:200]}")}

    sent = re.search(r'Rank\s+0\s+sent\s+([\d.]+)\s*\.\.\s*([\d.]+)', out)
    received = re.search(r'Rank\s+1\s+received\s+([\d.]+)\s*\.\.\s*([\d.]+)', out)

    a_ok = (sent is not None and received is not None and
             all(lib.close(float(v), 42.0, tol=1e-6) for v in sent.groups() + received.groups()))
    a_detail = (f"rank0 sent={sent.groups() if sent else None}, "
                f"rank1 received={received.groups() if received else None} (expect all 42.0)")

    return {"A": (a_ok, a_detail)}
