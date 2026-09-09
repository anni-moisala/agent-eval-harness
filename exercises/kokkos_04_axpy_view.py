"""kokkos/exercises/04-axpy-view -- axpy using Kokkos::View instead of raw pointers.

This exercise has no starter stub of its own -- its README says to start from
01-parallel-axpy's axpy.cpp. So the "stub" variant borrows that file: same
math, but raw malloc'd arrays and no Kokkos at all, which correctly fails the
Tier B check here (must use Kokkos::View) even though Tier A's numeric
check alone would pass it (same reasoning as kokkos_01_parallel_axpy).
"""

import os
import lib

EXERCISE_PATH = "kokkos/exercises/04-axpy-view"
FILES = ["axpy.cpp"]
STUB_FILE_SOURCES = {"axpy.cpp": "kokkos/exercises/01-parallel-axpy/axpy.cpp"}

_CMAKELISTS = """\
cmake_minimum_required(VERSION 3.20)
project(AXPYVIEW LANGUAGES CXX)
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
    view_count = lib.grep_count(src, r'Kokkos::View\s*<')
    malloc_count = lib.grep_count(src, r'\bmalloc\s*\(')
    b_ok = view_count >= 1 and malloc_count == 0
    b_detail = f"Kokkos::View declarations={view_count}, raw malloc() calls={malloc_count}"

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
