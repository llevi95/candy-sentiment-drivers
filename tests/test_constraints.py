"""Guards for constraints in the project brief."""

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "candy"

# Modelling may use only NumPy and SciPy; pandas is allowed as a container.
MODELLING_ALLOWED = {"numpy", "scipy", "pandas"}
# Elsewhere the package may also use plotting.
PACKAGE_ALLOWED = MODELLING_ALLOWED | {"matplotlib"}
MODELLING_MODULES = {"estimators.py"}


def third_party_imports(path: Path) -> set[str]:
    roots = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return {r for r in roots if r not in sys.stdlib_module_names and r != "candy"}


def test_modelling_uses_only_numpy_and_scipy():
    for name in MODELLING_MODULES:
        assert third_party_imports(SRC / name) <= MODELLING_ALLOWED, name


def test_package_imports_only_approved_libraries():
    offenders = {
        path.name: sorted(third_party_imports(path) - PACKAGE_ALLOWED)
        for path in SRC.rglob("*.py")
    }

    assert {k: v for k, v in offenders.items() if v} == {}
