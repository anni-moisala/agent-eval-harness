"""openmp/exercises/11-interop -- no code changes; compile+run for GPU target, then CPU target.

The source never changes (solution/ has only a README, no code -- the stub
is already the answer); the actual task is running it two ways and getting
the right *pattern* of address (in)equality. Literal addresses are
non-deterministic per run/machine, so grading checks the relationships, not
exact values: GPU target -> host addr != device addr (and the host-reported
device addr matches what the device itself reports); CPU target -> all three
addresses are equal (no real offload happens).
"""

import os
import re
import lib

EXERCISE_PATH = "openmp/exercises/11-interop"
FILES = ["pointers.c"]
SOLUTION_FILE_SOURCES = {"pointers.c": "openmp/exercises/11-interop/pointers.c"}
NO_UNSOLVED_STUB = True


def build(workdir):
    src = os.path.join(workdir, "pointers.c")
    gpu_out = os.path.join(workdir, "pointers_gpu")
    cpu_out = os.path.join(workdir, "pointers_cpu")
    lib.compile_nvc([src], gpu_out, extra_flags=["-mp=gpu", "-O3", "-gpu=cc90", "-Wall"])
    lib.compile_nvc([src], cpu_out, extra_flags=["-mp", "-O3", "-Wall"])
    return {"gpu": gpu_out, "cpu": cpu_out}


def run(built):
    cwd = os.path.dirname(built["gpu"])
    gpu_result = lib.run_gpu_nvhpc(built["gpu"], cwd=cwd)
    cpu_result = lib.run_nvhpc_local(built["cpu"], cwd=cwd)
    return {"gpu": gpu_result, "cpu": cpu_result}


def _addresses(out):
    host = re.search(r'from host the address of x in host:\s*(0x[0-9a-fA-F]+)', out)
    host_dev = re.search(r'from host the address of x in dev:\s*(0x[0-9a-fA-F]+)', out)
    dev_dev = re.search(r'from dev\s+the address of x in dev:\s*(0x[0-9a-fA-F]+)', out)
    return (host.group(1) if host else None,
            host_dev.group(1) if host_dev else None,
            dev_dev.group(1) if dev_dev else None)


def grade(results, workdir):
    checks = {}
    for label, key in (("gpu-target", "gpu"), ("cpu-target", "cpu")):
        rc, out, err = results[key]
        if rc != 0:
            checks[label] = (False, f"run exited {rc}: {err.strip()[:200]}")
            continue
        host, host_dev, dev_dev = _addresses(out)
        if None in (host, host_dev, dev_dev):
            checks[label] = (False, f"could not parse all 3 addresses from output: {out!r}")
            continue
        if label == "gpu-target":
            ok = host != host_dev and host_dev == dev_dev
            checks[label] = (ok, f"host={host}, host-reported-dev={host_dev}, dev-reported-dev={dev_dev} "
                                  f"(expect host!=dev, dev addrs equal)")
        else:
            ok = host == host_dev == dev_dev
            checks[label] = (ok, f"host={host}, host-reported-dev={host_dev}, dev-reported-dev={dev_dev} "
                                  f"(expect all 3 equal -- no real offload on CPU target)")

    return {"A": checks["gpu-target"], "B": checks["cpu-target"]}
