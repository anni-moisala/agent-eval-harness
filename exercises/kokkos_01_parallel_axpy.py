"""kokkos/exercises/01-parallel-axpy -- convert serial axpy loops to Kokkos::parallel_for.

The stub's CMakeLists.txt also builds a bonus `axpy-sharedspace` target that
depends on a file only present in solution/, so this module supplies its own
minimal CMakeLists rather than copying the repo's, and only ever grades the
primary `axpy` target.
"""

import os
import lib

EXERCISE_PATH = "kokkos/exercises/01-parallel-axpy"
FILES = ["axpy.cpp"]

_CMAKELISTS = """\
cmake_minimum_required(VERSION 3.20)
project(AXPY LANGUAGES CXX)
find_package(Kokkos REQUIRED CONFIG)
add_executable(axpy axpy.cpp)
target_link_libraries(axpy PRIVATE Kokkos::kokkos)
"""


def build(workdir):
    with open(os.path.join(workdir, "CMakeLists.txt"), "w") as f:
        f.write(_CMAKELISTS)
    build_dir = lib.cmake_build_cuda(workdir, os.path.join(workdir, "build-cuda"))
    return {"cuda": os.path.join(build_dir, "axpy")}


def run(built):
    rc, out, err = lib.run_gpu(built["cuda"])
    return {"cuda": (rc, out, err)}


def grade(results, workdir):
    rc, out, err = results["cuda"]
    if rc != 0:
        return {"A": (False, f"run exited {rc}: {err.strip()[:200]}"),
                 "B": (False, "skipped, run failed")}

    line = lib.line_after(out, "First and last element (both should be zero):")
    a_ok = line is not None and line.strip() == "0,0"
    a_detail = f"result line = {line!r} (expect '0,0')"

    src = lib.read(os.path.join(workdir, "axpy.cpp"))
    parallel_for_count = lib.grep_count(src, r'Kokkos::parallel_for')
    b_ok = parallel_for_count >= 1
    b_detail = f"Kokkos::parallel_for occurrences={parallel_for_count} (stub has 0 -- plain serial C++, no Kokkos at all)"

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
