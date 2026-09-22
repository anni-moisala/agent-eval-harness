"""Shared helpers for the exercise grading harness. See run_eval.py for the CLI.

Compiling runs directly (normal, lightweight login-node work, same as any
interactive `module load && make`). Every function that actually *executes*
a compiled exercise binary goes through `srun` on an allocated compute
node -- this process itself runs on the Roihu login node, and running a
binary directly there instead would be the thing to avoid.

(cmake's own compiler-detection step hangs indefinitely when wrapped in
srun regardless of partition/GPU reservation -- some interaction with how
srun forwards stdio to cmake's nested try_compile subprocesses. Plain
compiler/linker invocations (nvcc_wrapper, nvc, mpicc) don't have this
problem, but cmake specifically does, which is the other reason builds stay
unwrapped here.)
"""

import os
import re
import shutil
import subprocess

# This harness only targets Roihu, so the toolchain paths below are hardcoded
# for this cluster rather than configurable per-cluster. What's still meant
# to vary per user is: SRUN_ACCOUNT (required, no sane default), REPO_ROOT
# (a --repo-root CLI flag, see run_eval.py -- only self-test/solution/stub
# need it), and RUNS_ROOT/KOKKOS_ROOT_CUDA (env-overridable, but with
# defaults that should just work without any setup).
REPO_ROOT = "/scratch/dac/amoisala/portable-gpu-programming"
RUNS_ROOT = os.environ.get("EVAL_HARNESS_RUNS_ROOT",
                            os.path.join(os.getcwd(), "eval-harness-runs"))

# `module load kokkos` sets KOKKOS_INSTROOT to a ready-made Kokkos-CUDA
# install (bin/nvcc_wrapper + lib64/cmake/Kokkos/KokkosConfig.cmake) --
# no per-user Kokkos build needed. EVAL_HARNESS_KOKKOS_ROOT_CUDA still wins
# if set, for anyone who does want their own build.
KOKKOS_ROOT_CUDA = (os.environ.get("EVAL_HARNESS_KOKKOS_ROOT_CUDA")
                     or os.environ.get("KOKKOS_INSTROOT")
                     or "/scratch/dac/amoisala/kokkos/kokkos-cuda")
NVCC_WRAPPER = os.path.join(KOKKOS_ROOT_CUDA, "bin", "nvcc_wrapper")

MPICXX = "/appl/soft/spack/core/v2026_03/aarch64/g14cu129_eg/install_dir/neoverse_v2/gcc-14.3.0/openmpi-5.0.10-hrdnxd/bin/mpicxx"
MPICC = "/appl/soft/spack/core/v2026_03/aarch64/g14cu129_eg/install_dir/neoverse_v2/gcc-14.3.0/openmpi-5.0.10-hrdnxd/bin/mpicc"

SRUN_ACCOUNT = os.environ.get("ACCOUNT")
SRUN_CPU_PARTITION = "small"      # x86_64 nodes -- NOT usable for this harness's
                                   # binaries: nvc/g++/mpicc all target aarch64
                                   # (the login node's arch), and this cluster's
                                   # only aarch64 hardware is the GPU racks. Kept
                                   # only as srun_run's gpu=False default; nothing
                                   # calls it (see run_nvhpc_local's comment).
SRUN_GPU_PARTITION = os.environ.get("EVAL_HARNESS_SRUN_GPU_PARTITION", "gputest")
                                # fast dedicated test nodes -- see note in srun_run
SRUN_CPUS_PER_TASK = 4            # matches run_roihu.sh's --cpus-per-task=4
SRUN_DEFAULT_SLURM_TIME = "00:10:00"  # matches run_roihu.sh's --time=00:10:00


class BuildError(Exception):
    pass


def srun_run(cmd, cwd=None, timeout=180, env=None, gpu=False, ngpus=1, ntasks=1,
             slurm_time=SRUN_DEFAULT_SLURM_TIME):
    """Run `cmd` via srun on an allocated compute node -- never directly on
    the login node. gpu=True requests GPU(s) on the gputest partition (fast
    dedicated test nodes, better suited to the harness's many short
    validation runs than the busier production gpumedium partition the
    exercises' own run_roihu.sh scripts use); otherwise it's a plain CPU
    allocation on the small partition. `--cpus-per-task` and `--time`
    still follow the run_roihu.sh convention. `timeout` is the Python-side
    subprocess wait; `slurm_time` is the Slurm `--time` budget and should
    be >= timeout."""
    if not SRUN_ACCOUNT:
        raise BuildError("ACCOUNT is not set -- export it to "
                          "your Slurm account before running the harness.")
    prefix = ["srun", f"--account={SRUN_ACCOUNT}", f"--cpus-per-task={SRUN_CPUS_PER_TASK}"]
    if gpu:
        prefix += [f"--partition={SRUN_GPU_PARTITION}", f"--gres=gpu:gh200:{ngpus}"]
    else:
        prefix += [f"--partition={SRUN_CPU_PARTITION}"]
    if ntasks != 1:
        prefix += [f"--ntasks={ntasks}"]
    prefix += [f"--time={slurm_time}"]
    p = subprocess.run(prefix + list(cmd), cwd=cwd, capture_output=True,
                        text=True, env=env, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def sh(cmd, cwd=None, timeout=180):
    """Lightweight local exec for pure I/O (reading a git object, etc.) --
    not for anything that compiles or runs exercise code."""
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def compile_gxx(src_files, out_path, std="c++17", extra_flags=None):
    cmd = ["g++", "-O2", "-Wall", f"-std={std}"] + (extra_flags or []) + list(src_files) + ["-o", out_path]
    rc, out, err = sh(cmd)
    if rc != 0:
        raise BuildError(f"g++ failed:\n{err}")
    return out_path


def cmake_build_cuda(src_dir, build_dir):
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    rc, out, err = sh(["cmake", "-B", build_dir, "-S", src_dir,
                        f"-DCMAKE_CXX_COMPILER={NVCC_WRAPPER}",
                        f"-DKokkos_ROOT={KOKKOS_ROOT_CUDA}"])
    if rc != 0:
        raise BuildError(f"cmake configure failed:\n{out}\n{err}")
    rc, out, err = sh(["cmake", "--build", build_dir])
    if rc != 0:
        raise BuildError(f"cmake build failed:\n{out}\n{err}")
    return build_dir


def cmake_build_mpi_cuda(src_dir, build_dir):
    """Like cmake_build_cuda, but the compiler is the MPI wrapper (mpicxx),
    itself pointed at nvcc_wrapper via OMPI_CXX so Kokkos CUDA lambdas still
    compile."""
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    env = dict(os.environ, OMPI_CXX=NVCC_WRAPPER)
    p = subprocess.run(["cmake", "-B", build_dir, "-S", src_dir,
                         f"-DCMAKE_CXX_COMPILER={MPICXX}",
                         f"-DKokkos_ROOT={KOKKOS_ROOT_CUDA}"],
                        capture_output=True, text=True, env=env, timeout=180)
    if p.returncode != 0:
        raise BuildError(f"cmake configure failed:\n{p.stdout}\n{p.stderr}")
    p = subprocess.run(["cmake", "--build", build_dir],
                        capture_output=True, text=True, env=env, timeout=180)
    if p.returncode != 0:
        raise BuildError(f"cmake build failed:\n{p.stdout}\n{p.stderr}")
    return build_dir


def run_mpi_gpu(binary, args=None, timeout=120, cwd=None):
    return srun_run([binary] + (args or []), cwd=cwd, timeout=timeout, gpu=True, ngpus=2, ntasks=2)


_NVHPC_ENV = None


def nvhpc_env():
    """Environment with `module load nvhpc/26.3` applied, memoized. This
    just queries module state (no compute), so it's fine to run directly."""
    global _NVHPC_ENV
    if _NVHPC_ENV is None:
        p = subprocess.run(
            ["bash", "-lc", "module purge >/dev/null 2>&1; module load nvhpc/26.3 && env -0"],
            capture_output=True, timeout=60)
        if p.returncode != 0:
            raise BuildError(f"failed to load nvhpc/26.3 module:\n{p.stderr.decode()}")
        env = {}
        for kv in p.stdout.decode().split("\0"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k] = v
        _NVHPC_ENV = env
    return _NVHPC_ENV


def compile_nvc(src_files, out_path, extra_flags=None):
    """Compile C source(s) with nvc for OpenMP GPU offload (module nvhpc/26.3)."""
    flags = extra_flags if extra_flags is not None else ["-mp=gpu", "-O3", "-gpu=cc90", "-Wall"]
    cmd = ["nvc"] + flags + list(src_files) + ["-o", out_path]
    p = subprocess.run(cmd, capture_output=True, text=True, env=nvhpc_env(), timeout=120)
    if p.returncode != 0:
        raise BuildError(f"nvc failed:\n{p.stderr}")
    return out_path, p.stderr  # stderr carries -Minfo=mp offload diagnostics if requested


def compile_mpi_nvc(src_files, out_path, extra_flags=None):
    """Compile C source(s) with the MPI wrapper (mpicc), itself pointed at
    nvc via OMPI_CC so OpenMP GPU offload still works."""
    flags = extra_flags if extra_flags is not None else ["-O3", "-mp=gpu", "-gpu=cc90"]
    env = dict(nvhpc_env(), OMPI_CC="nvc")
    cmd = [MPICC] + flags + list(src_files) + ["-o", out_path]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    if p.returncode != 0:
        raise BuildError(f"mpicc failed:\n{p.stderr}")
    return out_path


def run_mpi_gpu_nvhpc(binary, args=None, ngpus=2, timeout=120, cwd=None):
    """Run an MPI+OpenMP-offload binary with 2 ranks via srun, with the
    nvhpc runtime environment. ngpus=2 gives each rank its own GPU;
    ngpus=1 makes both ranks share one (also a documented supported mode)."""
    env = dict(nvhpc_env())
    return srun_run([binary] + (args or []), cwd=cwd, env=env, timeout=timeout,
                     gpu=True, ngpus=ngpus, ntasks=2)


def make_nvhpc(build_dir, target=None, timeout=60):
    """Run `make` (optionally a specific target) in build_dir with the
    nvhpc/26.3 environment applied, for Makefile-based exercises."""
    cmd = ["make"] + ([target] if target else [])
    p = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True,
                        env=nvhpc_env(), timeout=timeout)
    if p.returncode != 0:
        raise BuildError(f"make failed:\n{p.stdout}\n{p.stderr}")
    return p.stdout, p.stderr


def run_nvhpc_local(binary, args=None, cwd=None, timeout=60):
    """Run an nvc-built CPU-target binary via srun (still never runs directly
    on the login node despite the name). Despite being a CPU-only build,
    this still requests a GPU-partition node (gpu=True, ngpus=1): this
    cluster's CPU-only Slurm partitions (SRUN_CPU_PARTITION) are x86_64,
    while the login node -- and every compiler this harness uses -- targets
    aarch64, and the GPU racks are the only aarch64 nodes Slurm has. Routing
    an aarch64 binary to the x86_64 "small" partition doesn't queue or time
    out, it fails execve() outright ("Exec format error") on every attempt --
    confirmed by rerunning the same binary against both partitions directly.
    No actual GPU offload happens here; the allocation is just how you get
    an aarch64 node."""
    return srun_run([binary] + (args or []), cwd=cwd, env=nvhpc_env(), gpu=True, ngpus=1, timeout=timeout)


def run_gpu_nvhpc(binary, args=None, extra_env=None, timeout=120, cwd=None):
    """Run a binary built with nvc, via srun on a GPU node, with the nvhpc
    runtime environment (needed for the OpenMP-offload runtime libs).
    Pass cwd=<workdir> for binaries that read/write relative-path files --
    otherwise they'd inherit the harness process's cwd, not the workdir."""
    env = dict(nvhpc_env())
    if extra_env:
        env.update(extra_env)
    return srun_run([binary] + (args or []), cwd=cwd, env=env, timeout=timeout, gpu=True, ngpus=1)


def run_local(binary, args=None, timeout=60, cwd=None):
    """Run a plain host binary directly, no srun. Only for exercises whose
    own instructions never mention Slurm at all (e.g. cpp/01-templates,
    cpp/02-lambdas -- trivial host C++ with no cluster/GPU angle); every
    exercise that's actually part of the GPU/HPC tracks goes through srun
    (see run_gpu, run_gpu_nvhpc, run_nvhpc_local, etc.) because that's the
    convention their own READMEs and run_roihu.sh scripts use, regardless
    of how lightweight the binary is."""
    return sh([binary] + (args or []), cwd=cwd, timeout=timeout)


def run_gpu(binary, args=None, timeout=120, cwd=None):
    return srun_run([binary] + (args or []), cwd=cwd, timeout=timeout, gpu=True, ngpus=1)


def line_after(output, marker):
    """Return the line right after the first line containing `marker`."""
    lines = output.splitlines()
    for i, l in enumerate(lines):
        if marker in l:
            return lines[i + 1] if i + 1 < len(lines) else None
    return None


def parse_floats(line):
    if line is None:
        return None
    try:
        return tuple(float(p) for p in line.split(","))
    except ValueError:
        return None


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def grep_count(text, pattern):
    return len(re.findall(pattern, text, re.MULTILINE))


def read(path):
    with open(path) as f:
        return f.read()


def fresh_workdir(exercise_id, variant):
    d = os.path.join(RUNS_ROOT, f"{exercise_id}-{variant}")
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    return d


def stub_file_from_git(relpath):
    """Read a file's pristine (git HEAD) content, bypassing local edits.
    Pure I/O against the git object store -- no compute, fine to run directly."""
    rc, out, err = sh(["git", "-C", REPO_ROOT, "show", f"HEAD:{relpath}"])
    if rc != 0:
        raise BuildError(f"could not read {relpath} from git HEAD:\n{err}")
    return out
