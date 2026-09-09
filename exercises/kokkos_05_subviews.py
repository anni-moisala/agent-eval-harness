"""kokkos/exercises/05-subviews -- boundary subviews of a 2D View, exact integer result."""

import os
import lib

EXERCISE_PATH = "kokkos/exercises/05-subviews"
FILES = ["subviews.cpp", "CMakeLists.txt"]

_EXPECTED = {
    "Top boundary": -18,
    "Bottom boundary": -36,
    "Left boundary": -90,
    "Rigth boundary": -108,  # typo is in the actual solution output, preserved on purpose
}


def build(workdir):
    build_dir = lib.cmake_build_cuda(workdir, os.path.join(workdir, "build-cuda"))
    return {"cuda": os.path.join(build_dir, "subviews")}


def run(built):
    rc, out, err = lib.run_gpu(built["cuda"])
    return {"cuda": (rc, out, err)}


def grade(results, workdir):
    rc, out, err = results["cuda"]
    if rc != 0:
        return {"A": (False, f"run exited {rc}: {err.strip()[:200]}"),
                 "B": (False, "skipped, run failed")}

    found = {}
    for l in out.splitlines():
        for label in _EXPECTED:
            if l.strip().startswith(label):
                try:
                    found[label] = int(l.strip().rsplit(None, 1)[1])
                except (ValueError, IndexError):
                    pass

    mismatches = {k: (v, found.get(k)) for k, v in _EXPECTED.items() if found.get(k) != v}
    a_ok = not mismatches
    a_detail = "all 4 boundary sums match exactly" if a_ok else f"mismatches (expected, got): {mismatches}"

    src = lib.read(os.path.join(workdir, "subviews.cpp"))
    subview_count = lib.grep_count(src, r'Kokkos::subview')
    b_ok = subview_count >= 4
    b_detail = f"Kokkos::subview occurrences={subview_count} (need >=4, one per boundary)"

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
