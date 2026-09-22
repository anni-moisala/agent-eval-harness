# Agent Eval Harness

Grading harness for CSC's [Portable GPU Programming](https://github.com/csc-training/portable-gpu-programming) exercises. It evaluates coding agents on HPC tasks involving C++, Kokkos, OpenMP, and MPI.

⚠️ **Warning:** The harness compiles and executes code directly on the Roihu cluster via `srun`, under your own account, with no sandboxing. Never run it on unreviewed or untrusted code.

## How it works

The harness evaluates each exercise in two tiers:

- **Tier A — Behavioral:** does the program produce the expected result?
- **Tier B — Structural:** does it use the construct the exercise is teaching, e.g. `Kokkos::parallel_for`, `omp target`, or `MPI_Sendrecv`?

Tier B is needed because an unmodified stub can sometimes produce the correct answer without implementing the intended parallel construct.

## Configuration

This harness only targets Roihu, so its toolchain paths are hardcoded. Only
one setting is required; the rest have working defaults:

| Variable | Required? | Default | Description |
|---|---|---|---|
| `ACCOUNT` | **yes** | none | Slurm account jobs are billed to |
| `EVAL_HARNESS_RUNS_ROOT` | no | `./eval-harness-runs` (relative to cwd) | Scratch directory for builds/runs |
| `EVAL_HARNESS_KOKKOS_ROOT_CUDA` | no | `$KOKKOS_INSTROOT` if set (i.e. after `module load kokkos`), else a fallback build | Kokkos CUDA installation |
| `EVAL_HARNESS_SRUN_GPU_PARTITION` | no | `gputest` | Slurm GPU partition |

So the only setup step is:

```
export ACCOUNT=<your Slurm account>
module load kokkos   # so Kokkos exercises build without a manual install
```

## Quick start

1. **Create a clean copy**

   ```
   cp -r repo-no-solutions my-agent-attempt
   ```

   Keep `repo-no-solutions/` untouched so it can be reused for other agents.

2. **Have the agent complete the exercises**

   Point the agent at `my-agent-attempt/` and have it follow each exercise's README.md.

   For example:

   > Complete the exercises in this training repository following the instructions in the READMEs. The solutions have been removed. Just complete the code; do not run anything.

3. **Grade the attempt**

   Grade all exercises:

   ```
   python3 run_eval.py my-agent-attempt
   ```

   Or grade one exercise:

   ```
   python3 run_eval.py <exercise_id> <candidate-dir>
   ```

   For example:

   ```
   python3 run_eval.py cpp_01_templates \
       my-agent-attempt/cpp/exercises/01-templates
   ```

   Grading uses a scratch work directory, so it does not modify the candidate repository.

## Output

Single exercise:

```
A: FAIL (2/3 result lines equal '0,0' (double/float/int))
B: PASS (template decls=1, concrete-type axpy overloads=0)
score: 1/2 tiers passed
workdir: /scratch/.../cpp_01_templates-candidate
```

A full-repository run reports each exercise and a final summary:

```
=== kokkos_04_axpy_view ===
  SKIPPED: missing ['axpy.cpp'] under my-agent-attempt/kokkos/exercises/04-axpy-view

overall: 11/32 tiers passed across 21 graded exercises
skipped (1): kokkos_04_axpy_view
```

SKIPPED means a required file was missing.

## Layout

```
run_eval.py         # CLI entry point
lib.py               # build/run and Slurm helpers
exercises/           # grading logic for each exercise
repo-no-solutions/   # clean, unsolved exercise set
```

## Self-testing the harness

To test the harness itself, pass `--repo-root` pointing at a clone of the
*original* training repository (with solutions, unlike `repo-no-solutions/`):

```
python3 run_eval.py --self-test --repo-root /path/to/portable-gpu-programming
```

Or compare an individual exercise against its known solution or stub:

```
python3 run_eval.py <exercise_id> solution --repo-root /path/to/portable-gpu-programming
python3 run_eval.py <exercise_id> stub --repo-root /path/to/portable-gpu-programming
```
