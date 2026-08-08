import importlib.util
from pathlib import Path

import pytest

_GEN_PATH = Path(__file__).resolve().parent.parent / "scripts" / "generate_fixtures.py"
_spec = importlib.util.spec_from_file_location("generate_fixtures", _GEN_PATH)
_generate_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_generate_fixtures)


@pytest.fixture(scope="session", autouse=True)
def _build_fixtures() -> None:
    """Regenerate fixtures once per test session so they're always in sync with the generator."""
    _generate_fixtures.main()


@pytest.fixture
def fixtures_dir() -> Path:
    return _generate_fixtures.FIXTURES_DIR
