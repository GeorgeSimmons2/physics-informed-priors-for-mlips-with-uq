"""
Unit tests for mace/lib.py and the functions the mace/ scripts define (stdlib unittest; no extra dependency).

    .venv/bin/python test/python/run.py        (bash run.sh test runs this)

Runs every test_*.py here, then fails if a function in mace/ or mace/pops/ has no test.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness  # noqa: E402

suite = unittest.defaultTestLoader.discover(HERE, pattern="test_*.py", top_level_dir=HERE)
result = unittest.TextTestRunner(verbosity=2).run(suite)
missing = sorted(harness.inventory() - harness.COVERED)
if missing:
    print("untested: " + ", ".join(f"{f}: {n}" for f, n in missing))
print(f"\n{result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors, "
      f"{len(result.expectedFailures)} expected failures; {len(harness.inventory()) - len(missing)} of "
      f"{len(harness.inventory())} functions covered")
sys.exit(0 if result.wasSuccessful() and not missing else 1)
