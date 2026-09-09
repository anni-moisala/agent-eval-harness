<!--
SPDX-FileCopyrightText: 2026 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# repo-no-solutions

Unsolved-stub copy of the exercises graded by `eval-harness`
(`../run_eval.py`), trimmed to only the 22 exercise directories the harness
actually grades. Sourced from the
[portable-gpu-programming](https://github.com/csc-training/portable-gpu-programming)
training material repository (CSC – IT Center for Science); each exercise's
own `README.md` carries its task instructions. See `COPYING` and `LICENSES/`
for licensing.

Point an agent at this directory to attempt the exercises, then grade the
result with:

    python3 ../run_eval.py <exercise_id> repo-no-solutions/<exercise-path>

or sweep every exercise at once:

    python3 ../run_eval.py repo-no-solutions
