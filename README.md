# eval-harness

Grading harness for the exercises in CSC's
[Portable GPU Programming](https://github.com/anni-moisala/portable-gpu-programming)
training material — built to grade what a coding agent produces, but works
the same way for a human's solution.

For each registered exercise it builds candidate source against a reference,
runs the resulting binary on Roihu (via Slurm), and checks the output/source
against known-good criteria, split into two tiers:

- **Tier A** — numeric/behavioral correctness (does the program produce the
  expected result).
- **Tier B** — structural correctness (did the candidate actually use the
  construct the exercise is about, e.g. `Kokkos::parallel_for`, `omp target`,
  `MPI_Sendrecv`) — needed because some exercises' unported/unfixed stub
  already produces the same numeric answer, just without the intended
  parallel construct.

## Layout

- `run_eval.py` — CLI entry point (see below).
- `lib.py` — shared build/run helpers (compilers, Slurm submission).
- `exercises/` — one module per registered exercise: what files it needs,
  how to build/run/grade them.
- `repo-no-solutions/` — the 22 registered exercises' unsolved stubs, trimmed
  from the training repo to just what's gradable (see its own README).

## Prerequisites

You do **not** need to clone the
[portable-gpu-programming](https://github.com/anni-moisala/portable-gpu-programming)
training repo just to grade a candidate's attempt (`run_eval.py <repo-root>`
or `run_eval.py <exercise_id> <candidate-dir>`) — `repo-no-solutions/` already
carries everything those need.

You only need a local clone of it, at the path `EVAL_HARNESS_REPO_ROOT` points
to (see below), for:

- `run_eval.py --self-test` — the harness's own acceptance test, and
- `run_eval.py <exercise_id> solution|stub` — grading against the official
  reference solution/stub directly.

Both pull the reference solution and pristine stub content from that repo.

## Usage

```
python3 run_eval.py --self-test                      # harness's own acceptance test
python3 run_eval.py <exercise_id> solution|stub       # grade one exercise's known solution/stub
python3 run_eval.py <exercise_id> <candidate-dir>     # grade one exercise against an arbitrary directory
python3 run_eval.py <repo-root>                       # sweep every exercise against <repo-root>/<exercise-path>
```

To grade an agent's attempt:

1. **Copy `repo-no-solutions/` somewhere else first** — don't hand the agent
   the original. Grading re-copies whatever files it finds into a scratch
   workdir, so nothing here is destroyed by grading itself, but the agent
   will edit these files in place, and you want `repo-no-solutions/` to stay
   a clean starting point for the next attempt/agent:

   ```
   cp -r repo-no-solutions my-agent-attempt
   ```

2. Point the agent at `my-agent-attempt/` and let it work through each
   exercise's own `README.md`.
3. Grade the result: `python3 run_eval.py my-agent-attempt` (sweeps every
   exercise), or one at a time with
   `python3 run_eval.py <exercise_id> my-agent-attempt/<exercise-path>`.

## Configuration

`lib.py`'s cluster/account/path config reads from environment variables,
falling back to this environment's own values if unset — so it runs
unmodified here, and elsewhere by exporting overrides instead of editing
the source:

| Variable | Default | What it is |
|---|---|---|
| `EVAL_HARNESS_REPO_ROOT` | `/scratch/dac/amoisala/portable-gpu-programming` | Clone of the training repo (see Prerequisites) |
| `EVAL_HARNESS_RUNS_ROOT` | `/scratch/dac/amoisala/eval-harness-runs` | Scratch dir grading builds/runs happen in |
| `EVAL_HARNESS_KOKKOS_ROOT_CUDA` | `/scratch/dac/amoisala/kokkos/kokkos-cuda` | Kokkos CUDA-backend install used by `cmake_build_cuda`/`nvcc_wrapper` |
| `EVAL_HARNESS_MPICXX` / `EVAL_HARNESS_MPICC` | this cluster's spack install path | MPI C++/C compiler wrappers |
| `EVAL_HARNESS_SRUN_ACCOUNT` | `dac` | Slurm account jobs are billed to |
| `EVAL_HARNESS_SRUN_GPU_PARTITION` | `gputest` | Slurm partition every build/run job actually lands on |

One partition is deliberately **not** configurable: `SRUN_CPU_PARTITION`
(`"small"`) is Roihu's x86_64 CPU-only partition, incompatible with every
binary this harness builds (all aarch64, matching the login node) — see the
comment on it and on `run_nvhpc_local` in `lib.py` before ever pointing
anything at it again.
