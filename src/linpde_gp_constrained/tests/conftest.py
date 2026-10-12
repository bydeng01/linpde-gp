"""Pytest discovery setup for the linpde_gp_constrained tests.

Adds ``src/linpde_gp_constrained/`` (i.e. the package's parent directory)
to ``sys.path`` so that ``import linpde_gp_constrained`` resolves whether
or not the package has been installed via ``pip install -e``. This lets
``pytest tests/`` work from any directory without requiring a developer
to set ``PYTHONPATH`` by hand.

Tests that depend on the full linpde-gp + probnum + jax stack continue to
``pytest.importorskip("linpde_gp")`` at the top of the module; the path
manipulation here only resolves our own package.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Add ``src/linpde_gp_constrained/`` (one level up from ``tests/``) to
# sys.path so the inner ``linpde_gp_constrained`` package is importable.
_PKG_PARENT = os.path.abspath(os.path.join(_HERE, ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)
