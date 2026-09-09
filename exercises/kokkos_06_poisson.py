"""kokkos/exercises/06-poisson -- port a serial Jacobi Poisson solver to Kokkos.

The stub is a plain std::vector-based serial solver -- it's the literal
source of the reference numbers (README quotes them from running this exact
code), so an unported recompile reproduces u[512,512]/Mean u exactly. Tier B
(Kokkos::View + parallel_for/parallel_reduce actually used) is load-bearing
here, same reasoning as kokkos_01/02.
"""

import os
import re
import lib

EXERCISE_PATH = "kokkos/exercises/06-poisson"
FILES = ["poisson.cpp"]

EXPECTED_U = -120.808781
EXPECTED_MEAN = -0.041238
TOL = 5e-3

_CMAKELISTS = """\
cmake_minimum_required(VERSION 3.20)
project(POISSON LANGUAGES CXX)
find_package(Kokkos REQUIRED CONFIG)
add_executable(poisson poisson.cpp)
target_link_libraries(poisson PRIVATE Kokkos::kokkos)
"""


def build(workdir):
    with open(os.path.join(workdir, "CMakeLists.txt"), "w") as f:
        f.write(_CMAKELISTS)
    build_dir = lib.cmake_build_cuda(workdir, os.path.join(workdir, "build-cuda"))
    return {"cuda": os.path.join(build_dir, "poisson")}


def run(built):
    # cwd: the binary writes a relative-path u.bin, which should land in the
    # workdir, not wherever the harness process happens to be running from.
    cwd = os.path.dirname(built["cuda"])
    rc, out, err = lib.run_gpu(built["cuda"], args=["1024", "500"], timeout=280, cwd=cwd)
    return {"cuda": (rc, out, err)}


def grade(results, workdir):
    rc, out, err = results["cuda"]
    if rc != 0:
        return {"A": (False, f"run exited {rc}: {err.strip()[:200]}"),
                 "B": (False, "skipped, run failed")}

    u_match = re.search(r'u\[\d+,\d+\]\s*=\s*([-\d.eE+]+)', out)
    mean_match = re.search(r'Mean u\s*=\s*([-\d.eE+]+)', out)
    u_val = float(u_match.group(1)) if u_match else None
    mean_val = float(mean_match.group(1)) if mean_match else None

    a_ok = (u_val is not None and lib.close(u_val, EXPECTED_U, tol=TOL) and
             mean_val is not None and lib.close(mean_val, EXPECTED_MEAN, tol=TOL))
    a_detail = (f"u[512,512]={u_val!r} (expect {EXPECTED_U}), "
                f"Mean u={mean_val!r} (expect {EXPECTED_MEAN}), tol={TOL}")

    src = lib.read(os.path.join(workdir, "poisson.cpp"))
    view_count = lib.grep_count(src, r'Kokkos::View\s*<')
    parallel_for_count = lib.grep_count(src, r'Kokkos::parallel_for')
    parallel_reduce_count = lib.grep_count(src, r'Kokkos::parallel_reduce')
    b_ok = view_count >= 1 and parallel_for_count >= 2 and parallel_reduce_count >= 1
    b_detail = (f"Kokkos::View={view_count}, parallel_for={parallel_for_count} (need >=2: init+jacobi), "
                f"parallel_reduce={parallel_reduce_count} (need >=1: mean)")

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
