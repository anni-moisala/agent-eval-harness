#!/usr/bin/env python3
"""CLI for the exercise grading harness.

Usage:
  run_eval.py --self-test
      Build+run+grade every registered exercise against both its known
      solution/ (must PASS) and its unsolved stub (must FAIL). This is the
      acceptance test for the grading logic itself.

  run_eval.py <exercise_id> solution|stub
      Grade one exercise against its repo solution/ or stub.

  run_eval.py <exercise_id> <path-to-candidate-dir>
      Grade one exercise against an arbitrary directory (e.g. what an agent
      produced) containing the same files the exercise module declares in
      FILES.

  run_eval.py <repo-root>
      Grade every registered exercise against <repo-root>/<exercise-path>
      (e.g. repo-no-solutions, which mirrors the same track/exercises/name
      layout as lib.REPO_ROOT). Exercises missing declared FILES there are
      skipped rather than failing the whole sweep.
"""

import argparse
import importlib
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

REGISTRY = [
    "cpp_01_templates",
    "cpp_02_lambdas",
    "kokkos_00_hello",
    "kokkos_01_parallel_axpy",
    "kokkos_02_parallel_dot_product",
    "kokkos_04_axpy_view",
    "kokkos_05_subviews",
    "kokkos_06_poisson",
    "kokkos_09_mpi_message_exchange",
    "openmp_02_axpy",
    "openmp_03_axpy_data",
    "openmp_04_heat",
    "openmp_05_reduction_sum",
    "openmp_06_heat_reduction",
    "openmp_07_heat_async",
    "openmp_08_heat_io",
    "openmp_10_axpy_device_functions",
    "openmp_11_interop",
    "openmp_12_axpy_blas",
    "openmp_13_heat_kernels",
    "openmp_15_mpi_send_recv",
    "openmp_16_heat_multi_gpu",
]


def load(name):
    return importlib.import_module(f"exercises.{name}")


def resolve_source(module, exercise_id, variant):
    """variant is 'solution' or 'stub'. Copies the module's declared FILES
    into a fresh workdir. For 'stub', a file can instead come from:
      - git HEAD (module.GIT_HEAD_STUB_FILES), when the working tree copy
        can't be trusted (e.g. we already solved it earlier this session), or
      - a different exercise's stub (module.STUB_FILE_SOURCES: {fname:
        repo-relative path}), for exercises with no stub of their own whose
        README says to start from another exercise's code.
    For 'solution', a file can instead come from a differently-named file
    within solution/ (module.SOLUTION_FILE_RENAMES: {fname: filename-within-
    solution-dir}), for exercises whose solution/ has several variants and
    none is named exactly like the stub file, or from an arbitrary
    repo-relative path (module.SOLUTION_FILE_SOURCES: {fname: repo-relative
    path}), for exercises whose solution/ has no code of its own because the
    stub itself is already the answer (a "no code changes needed" exercise)."""
    workdir = lib.fresh_workdir(exercise_id, variant)
    base_dir = os.path.join(lib.REPO_ROOT, module.EXERCISE_PATH,
                             "solution" if variant == "solution" else "")
    git_head_files = set(getattr(module, "GIT_HEAD_STUB_FILES", []))
    stub_file_sources = getattr(module, "STUB_FILE_SOURCES", {})
    solution_renames = getattr(module, "SOLUTION_FILE_RENAMES", {})
    solution_file_sources = getattr(module, "SOLUTION_FILE_SOURCES", {})
    for fname in module.FILES:
        dst = os.path.join(workdir, fname)
        if os.path.dirname(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        if variant == "stub" and fname in git_head_files:
            relpath = os.path.join(module.EXERCISE_PATH, fname)
            with open(dst, "w") as f:
                f.write(lib.stub_file_from_git(relpath))
        elif variant == "stub" and fname in stub_file_sources:
            shutil.copyfile(os.path.join(lib.REPO_ROOT, stub_file_sources[fname]), dst)
        elif variant == "solution" and fname in solution_renames:
            shutil.copyfile(os.path.join(base_dir, solution_renames[fname]), dst)
        elif variant == "solution" and fname in solution_file_sources:
            shutil.copyfile(os.path.join(lib.REPO_ROOT, solution_file_sources[fname]), dst)
        else:
            shutil.copyfile(os.path.join(base_dir, fname), dst)
    return workdir


def resolve_candidate(module, exercise_id, candidate_dir):
    workdir = lib.fresh_workdir(exercise_id, "candidate")
    for fname in module.FILES:
        dst = os.path.join(workdir, fname)
        if os.path.dirname(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(candidate_dir, fname), dst)
    return workdir


def grade_workdir(module, workdir):
    try:
        built = module.build(workdir)
    except lib.BuildError as e:
        return {"A": (False, f"BUILD FAILED: {e}")}
    results = module.run(built)
    return module.grade(results, workdir)


def fmt_tier(tier_result):
    ok, detail = tier_result
    return f"{'PASS' if ok else 'FAIL'} ({detail})"


def tier_score(grade):
    """(tiers passed, tiers evaluated) for one grade dict."""
    total = len(grade)
    passed = sum(1 for v in grade.values() if v[0])
    return passed, total


def sweep(candidates_root):
    """Grade every registered exercise against <candidates_root>/<module.EXERCISE_PATH>
    (e.g. repo-no-solutions), the same layout module.EXERCISE_PATH already uses under
    lib.REPO_ROOT. Exercises whose candidate dir is missing declared FILES are skipped
    rather than crashing the whole sweep."""
    total_passed = total_total = 0
    skipped = []

    for name in REGISTRY:
        module = load(name)
        candidate_dir = os.path.join(candidates_root, module.EXERCISE_PATH)
        print(f"=== {name} ===")
        missing = [f for f in module.FILES if not os.path.exists(os.path.join(candidate_dir, f))]
        if missing:
            print(f"  SKIPPED: missing {missing} under {candidate_dir}")
            skipped.append(name)
            print()
            continue

        workdir = resolve_candidate(module, name, candidate_dir)
        grade = grade_workdir(module, workdir)
        for tier in ("A", "B"):
            if tier in grade:
                print(f"  {tier}: {fmt_tier(grade[tier])}")
        passed, total = tier_score(grade)
        total_passed += passed
        total_total += total
        print(f"  score: {passed}/{total} tiers passed")
        print(f"  workdir: {workdir}")
        print()

    print(f"overall: {total_passed}/{total_total} tiers passed "
          f"across {len(REGISTRY) - len(skipped)} graded exercises")
    if skipped:
        print(f"skipped ({len(skipped)}): {', '.join(skipped)}")
    return total_total > 0 and total_passed == total_total


def self_test():
    all_ok = True
    solution_passed = solution_total = 0
    stub_passed = stub_total = 0

    for name in REGISTRY:
        module = load(name)
        for variant in ("solution", "stub"):
            print(f"=== {name} [{variant}] ===")
            workdir = resolve_source(module, name, variant)
            grade = grade_workdir(module, workdir)
            for tier in ("A", "B"):
                if tier in grade:
                    print(f"  {tier}: {fmt_tier(grade[tier])}")

            passed, total = tier_score(grade)
            print(f"  score: {passed}/{total} tiers passed")
            if variant == "solution":
                solution_passed += passed
                solution_total += total
            else:
                stub_passed += passed
                stub_total += total

            # NO_UNSOLVED_STUB: exercises where the stub IS the answer (README
            # says "no code changes needed") -- both variants are expected to PASS.
            expect_pass = True if getattr(module, "NO_UNSOLVED_STUB", False) else (variant == "solution")
            actual_pass = all(v[0] for v in grade.values())
            ok = (actual_pass == expect_pass)
            all_ok &= ok
            print(f"  expected {'PASS' if expect_pass else 'FAIL'}, "
                  f"got {'PASS' if actual_pass else 'FAIL'} -> {'OK' if ok else '**UNEXPECTED**'}")
            print(f"  workdir: {workdir}")
        print()

    print(f"solution tier score: {solution_passed}/{solution_total} "
          f"(expect {solution_total}/{solution_total} -- solutions should clear every check)")
    print(f"stub tier score:     {stub_passed}/{stub_total} "
          f"(expect well below {stub_total}/{stub_total} -- stubs are unsolved by design)")
    print("SELF-TEST", "PASSED" if all_ok else "FAILED")
    return all_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--repo-root", default=lib.REPO_ROOT,
                    help="clone of the original training repo (with solutions), "
                         "only needed for --self-test or a solution|stub grade "
                         f"(default: {lib.REPO_ROOT})")
    p.add_argument("exercise", nargs="?",
                    help="exercise id, or a repo root directory to sweep every "
                         "exercise against")
    p.add_argument("variant", nargs="?",
                    help="'solution', 'stub', or a path to a candidate directory")
    args = p.parse_args()
    lib.REPO_ROOT = args.repo_root

    if not lib.SRUN_ACCOUNT:
        p.error("ACCOUNT is not set -- export it to your "
                 "Slurm account before running the harness.")

    if args.self_test:
        sys.exit(0 if self_test() else 1)

    if args.exercise and not args.variant:
        if not os.path.isdir(args.exercise):
            p.error(f"'{args.exercise}' is not a directory -- pass a repo root to "
                     "sweep all exercises, or give a variant: "
                     "run_eval.py <exercise_id> solution|stub|<candidate-dir>")
        sys.exit(0 if sweep(args.exercise) else 1)

    if not args.exercise or not args.variant:
        p.error("usage: run_eval.py <exercise_id> solution|stub|<candidate-dir>, or run_eval.py <repo-root>")

    module = load(args.exercise)
    if args.variant in ("solution", "stub"):
        workdir = resolve_source(module, args.exercise, args.variant)
    else:
        workdir = resolve_candidate(module, args.exercise, args.variant)

    grade = grade_workdir(module, workdir)
    for tier in ("A", "B"):
        if tier in grade:
            print(f"{tier}: {fmt_tier(grade[tier])}")
    passed, total = tier_score(grade)
    print(f"score: {passed}/{total} tiers passed")
    print(f"workdir: {workdir}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
