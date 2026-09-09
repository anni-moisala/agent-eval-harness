"""kokkos/exercises/09-mpi-message-exchange -- port a raw-buffer MPI exchange to Kokkos::View.

2 MPI ranks exchange a message via MPI_Sendrecv; message[i] is always the
sender's own rank, so each rank's first received element must equal its
`src` rank ((rank-1+ntasks)%ntasks). With ntasks=2: rank 0 must receive
first=1, rank 1 must receive first=0. Print order between the two ranks is
non-deterministic, so each rank's own line is matched independently.
"""

import os
import re
import lib

EXERCISE_PATH = "kokkos/exercises/09-mpi-message-exchange"
FILES = ["exchange.cpp"]

_CMAKELISTS = """\
cmake_minimum_required(VERSION 3.20)
project(EXCHANGE LANGUAGES CXX)
find_package(Kokkos REQUIRED CONFIG)
add_executable(exchange exchange.cpp)
target_link_libraries(exchange PRIVATE Kokkos::kokkos)
"""


def build(workdir):
    with open(os.path.join(workdir, "CMakeLists.txt"), "w") as f:
        f.write(_CMAKELISTS)
    build_dir = lib.cmake_build_mpi_cuda(workdir, os.path.join(workdir, "build-cuda"))
    return {"cuda": os.path.join(build_dir, "exchange")}


def run(built):
    rc, out, err = lib.run_mpi_gpu(built["cuda"])
    return {"cuda": (rc, out, err)}


def grade(results, workdir):
    rc, out, err = results["cuda"]
    if rc != 0:
        return {"A": (False, f"run exited {rc}: {err.strip()[:200]}"),
                 "B": (False, "skipped, run failed")}

    per_rank_first = {}
    for m in re.finditer(r'Rank\s+(\d+)\s+received\s+\d+\s+elements,\s+first\s+(-?\d+)', out):
        per_rank_first[int(m.group(1))] = int(m.group(2))

    expected = {0: 1, 1: 0}  # rank r's src = (r-1+ntasks) % ntasks, ntasks=2
    a_ok = per_rank_first == expected
    a_detail = f"per-rank 'first' values={per_rank_first} (expect {expected})"

    src = lib.read(os.path.join(workdir, "exchange.cpp"))
    view_count = lib.grep_count(src, r'Kokkos::View\s*<')
    vector_count = lib.grep_count(src, r'std::vector\s*<\s*int\s*>')
    parallel_for_count = lib.grep_count(src, r'Kokkos::parallel_for')
    b_ok = view_count >= 2 and vector_count == 0 and parallel_for_count >= 1
    b_detail = (f"Kokkos::View={view_count} (need >=2: message+receiveBuffer), "
                f"std::vector<int>={vector_count} (need 0), parallel_for={parallel_for_count}")

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
