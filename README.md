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

## Usage

```
python3 run_eval.py --self-test                      # harness's own acceptance test
python3 run_eval.py <exercise_id> solution|stub       # grade one exercise's known solution/stub
python3 run_eval.py <exercise_id> <candidate-dir>     # grade one exercise against an arbitrary directory
python3 run_eval.py <repo-root>                       # sweep every exercise against <repo-root>/<exercise-path>
```

To grade an agent's attempt: point it at `repo-no-solutions/`, then run
`python3 run_eval.py repo-no-solutions` (or grade one exercise at a time)
once it's done.

## Before this runs anywhere else

`lib.py` hardcodes this environment's absolute paths and Slurm account —
none of it is a secret, but it won't run elsewhere without editing:

- `REPO_ROOT`, `RUNS_ROOT`, `KOKKOS_ROOT_CUDA` — absolute paths under this
  user's `/scratch`.
- `MPICXX`, `MPICC` — this cluster's specific MPI install path.
- `SRUN_ACCOUNT` — the Slurm account to bill jobs to.
- `SRUN_CPU_PARTITION`/`SRUN_GPU_PARTITION` — Roihu's CPU racks are x86_64
  while the GPU racks (and this harness's own compilers) are aarch64, so
  `run_nvhpc_local` deliberately requests a GPU-partition node even for a
  CPU-only build; see the comment in `lib.py` before changing this.
