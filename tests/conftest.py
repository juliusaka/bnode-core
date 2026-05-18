import os
import pytest
from pathlib import Path
from hydra.core.global_hydra import GlobalHydra

# Root of the bnode-core repository (one level up from tests/)
BNODE_CORE_ROOT = Path(__file__).resolve().parent.parent

SUBMODULE_PATHS = [
    BNODE_CORE_ROOT / "resources" / "data" / "surrogate-test-data",
    BNODE_CORE_ROOT / "resources" / "models" / "surrogate-test-models",
]


@pytest.fixture(scope="session", autouse=True)
def _chdir_to_bnode_core():
    """Ensure the working directory is bnode-core/ for all tests.

    Many tests use relative paths like 'resources/config' or
    './tests/_results/' that are relative to the bnode-core root.
    """
    original_cwd = os.getcwd()
    os.chdir(BNODE_CORE_ROOT)
    yield
    os.chdir(original_cwd)


@pytest.fixture(scope="session", autouse=True)
def _check_submodules():
    """Check that required git submodules under resources/ are cloned.

    This runs before all other tests. If any submodule directory is missing
    or empty, the entire test session is aborted with a clear error message.
    """
    missing = []
    for path in SUBMODULE_PATHS:
        if not path.exists() or not any(path.iterdir()):
            missing.append(str(path.relative_to(BNODE_CORE_ROOT)))
    if missing:
        pytest.fail(
            f"Required git submodules are not cloned: {', '.join(missing)}.\n"
            f"Run 'git submodule update --init' from {BNODE_CORE_ROOT}",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def _clear_global_hydra_state():
    """Keep Hydra's process-global singleton from leaking across tests."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()
