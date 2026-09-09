"""cpp/exercises/01-templates -- templated axpy for double/float/int."""

import os
import lib

EXERCISE_PATH = "cpp/exercises/01-templates"
FILES = ["axpy.cpp"]


def build(workdir):
    out = os.path.join(workdir, "axpy")
    lib.compile_gxx([os.path.join(workdir, "axpy.cpp")], out)
    return {"host": out}


def run(built):
    rc, out, err = lib.run_local(built["host"])
    return {"host": (rc, out, err)}


def grade(results, workdir):
    rc, out, err = results["host"]
    if rc != 0:
        return {"A": (False, f"program exited {rc}: {err.strip()[:200]}"),
                 "B": (False, "skipped, run failed")}

    zero_lines = [l for l in out.splitlines() if l.strip() == "0,0"]
    a_ok = len(zero_lines) == 3
    a_detail = f"{len(zero_lines)}/3 result lines equal '0,0' (double/float/int)"

    src = lib.read(os.path.join(workdir, "axpy.cpp"))
    template_count = lib.grep_count(src, r'^\s*template\s*<')
    concrete_overloads = lib.grep_count(src, r'\bvoid\s+axpy\s*\(\s*(double|float|int)\s*\*')
    b_ok = template_count >= 1 and concrete_overloads == 0
    b_detail = f"template decls={template_count}, concrete-type axpy overloads={concrete_overloads}"

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
