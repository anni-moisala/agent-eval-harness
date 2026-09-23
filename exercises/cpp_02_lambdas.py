"""cpp/exercises/02-lambdas -- lambda-bodied init + dot-product reduction."""

import os
import lib

EXERCISE_PATH = "cpp/exercises/02-lambdas"
FILES = ["dot-product.cpp"]


def build(workdir):
    out = os.path.join(workdir, "dot-product")
    lib.compile_gxx([os.path.join(workdir, "dot-product.cpp")], out)
    return {"host": out}


def run(built):
    rc, out, err = lib.run_local(built["host"])
    return {"host": (rc, out, err)}


def grade(results, workdir):
    rc, out, err = results["host"]
    if rc != 0:
        return {"A": (False, f"program exited {rc}: {err.strip()[:200]}"),
                 "B": (False, "skipped, run failed")}

    value = None
    for l in out.splitlines():
        if l.strip().startswith("Result (should be 0):"):
            try:
                value = float(l.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                pass
            break
    a_ok = value is not None and lib.close(value, 0.0, tol=1e-9)
    a_detail = f"result={value!r} (tolerance 1e-9)"

    src = lib.read(os.path.join(workdir, "dot-product.cpp"))
    # Any capture list, not just [=...]/[&...] -- naming captured variables
    # explicitly (e.g. `[x, y](...)`) is equally valid and idiomatic.
    lambda_count = lib.grep_count(src, r'\[[^\]]*\]\s*\(')
    b_ok = lambda_count >= 2
    b_detail = f"lambda expressions found={lambda_count} (need >=2: init + reduction)"

    return {"A": (a_ok, a_detail), "B": (b_ok, b_detail)}
