"""
test/python/harness.py — load the functions a script defines without running the script, and track coverage.

The mace/ scripts are top-level pipelines whose functions read module globals (calc, basis, OUT, ...).
script_namespace(relpath, inject) parses the file, keeps its import statements and top-level def/class nodes, drops
everything else, and executes them in a fresh namespace pre-filled with `inject`.  Scripts are never edited.
"""
import ast
import os
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MACE = os.path.join(REPO, "mace")
# mace/lib.py creates $OUT/mace/... on import: point OUT at a scratch directory before anything imports it
os.environ.setdefault("OUT", tempfile.mkdtemp(prefix="pytest_out_"))
os.environ.setdefault("THREADS", "4")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, MACE)            # `import lib` in mace/*.py and (via a dropped sys.path line) mace/pops/*.py

COVERED = set()


def tested(relpath, *names):
    """Mark relpath's functions (or Class.method) as tested."""
    COVERED.update((relpath, n) for n in names)


def script_namespace(relpath, inject=None, only=None):
    """A namespace holding relpath's imports and top-level defs/classes (all, or those in `only`), `inject` first."""
    path = os.path.join(REPO, relpath)
    tree = ast.parse(open(path).read(), filename=path)
    body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and (only is None or n.name in only))]
    ns = {"__name__": "under_test_" + os.path.basename(relpath).replace(".", "_"), "__file__": path}
    ns.update(inject or {})
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
    return ns


def defined(relpath):
    """Top-level function names, and Class.method for classes."""
    tree = ast.parse(open(os.path.join(REPO, relpath)).read())
    out = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            out.append(n.name)
        elif isinstance(n, ast.ClassDef):
            out += [f"{n.name}.{m.name}" for m in n.body if isinstance(m, ast.FunctionDef)]
    return out


def inventory():
    files = sorted(os.path.relpath(os.path.join(d, f), REPO) for d in (MACE, os.path.join(MACE, "pops"))
                   for f in os.listdir(d) if f.endswith(".py"))
    return {(f, n) for f in files for n in defined(f)}
