"""Test that all public functions defined in sampling submodules are exported from __init__.py."""
import importlib
import inspect
import pkgutil
from types import ModuleType

import bnode_core.data_generation.sampling as sampling_pkg


def _public_functions(module: ModuleType) -> set[str]:
    return {
        name
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == module.__name__
    }


def test_all_sampling_functions_exported():
    missing = []
    pkg_path = sampling_pkg.__path__
    pkg_name = sampling_pkg.__name__

    for _, module_name, _ in pkgutil.iter_modules(pkg_path):
        full_name = f"{pkg_name}.{module_name}"
        module = importlib.import_module(full_name)
        for func_name in _public_functions(module):
            if not hasattr(sampling_pkg, func_name):
                missing.append(f"{full_name}.{func_name}")

    assert not missing, (
        "The following functions are defined in sampling submodules but not exported "
        f"from sampling/__init__.py:\n" + "\n".join(f"  - {m}" for m in missing)
    )
