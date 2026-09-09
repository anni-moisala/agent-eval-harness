"""kokkos/exercises/00-hello -- complete CMakeLists.txt to build+run under Kokkos/CUDA."""

import os
import lib

EXERCISE_PATH = "kokkos/exercises/00-hello"
FILES = ["CMakeLists.txt", "hello.cpp"]
# The working-tree CMakeLists.txt here was already solved earlier this
# session, so the "stub" variant must be read from git HEAD instead.
GIT_HEAD_STUB_FILES = ["CMakeLists.txt"]


def build(workdir):
    build_dir = lib.cmake_build_cuda(workdir, os.path.join(workdir, "build-cuda"))
    return {"cuda": os.path.join(build_dir, "hello")}


def run(built):
    rc, out, err = lib.run_gpu(built["cuda"], args=["--kokkos-print-configuration"])
    return {"cuda": (rc, out, err)}


def grade(results, workdir):
    rc, out, err = results["cuda"]
    if rc != 0:
        return {"A": (False, f"run exited {rc}: {err.strip()[:200]}")}

    a_ok = "Default Device: Cuda" in out
    a_detail = "found 'Default Device: Cuda' in --kokkos-print-configuration output" if a_ok \
        else "did not find 'Default Device: Cuda' in output"

    return {"A": (a_ok, a_detail)}
