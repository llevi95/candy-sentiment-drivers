"""Guards for constraints in the project brief."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "candy"
FORBIDDEN_MODELLING = {"statsmodels", "sklearn", "linearmodels", "pingouin", "patsy"}


def test_modelling_uses_only_numpy_and_scipy():
    offenders = []
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = [node.module.split(".")[0]]
            else:
                continue
            offenders += [f"{path.name}: {r}" for r in roots if r in FORBIDDEN_MODELLING]

    assert offenders == []
