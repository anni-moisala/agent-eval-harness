"""kokkos/exercises/02-parallel-dot-product -- convert serial dot-product to Kokkos::parallel_reduce."""

import os
import lib

EXERCISE_PATH = "kokkos/exercises/02-parallel-dot-product"
FILES = ["dot-product.cpp"]

_CMAKELISTS = """\
cmake_minimum_required(VERSION 3.20)
project(DOTPRODUCT LANGUAGES CXX)
find_package(Kokkos REQUIRED CONFIG)
add_executable(dot-product dot-product.cpp)
target_link_libraries(dot-product PRIVATE Kokkos::kokkos)
"""


def build(workdir):
    with open(os.path.join(workdir, "CMakeLists.txt"), "w") as f:
        f.write(_CMAKELISTS)
    build_dir = lib.cmake_build_cuda(workdir, os.path.join(workdir, "build-cuda"))
    return {"cuda": os.path.join(build_dir, "dot-product")}


def run(built):
    rc, out, err = lib.run_gpu(built["cuda"])
    return {"cuda": (rc, out, err)}


def grade(results, workdir):
    rc, out, err = results["cuda"]
    if rc != 0:
        return {"A": (False, f"run exited {rc}: {err.strip()[:200]}"),
                 "B": (False, "skipped, run failed")}

    marker_line = None
    for l in out.splitlines():
        if l.strip().startswith("Result (should be 0):"):
            marker_line = l
            break
    value = None
    if marker_line is not None:
        try:
            value = float(marker_line.split(":", 1)[1].strip())
        except (ValueError, IndexError):
            value = None
    a_ok = value is not None and lib.close(value, 0.0, tol=1e-9)
    a_detail = f"result={value!r} (tolerance 1e-9)"

    src = lib.read(os.path.join(workdir, "dot-product.cpp"))
    parallel_reduce_count = lib.grep_count(src, r'Kokkos::parallel_reduce')
    b_ok = parallel_reduce_count >= 1
    b_detail = f"Kokkos::parallel_reduce occurrences={parallel_reduce_count} (stub has 0 -- plain serial C++, no Kokkos at all)"

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
