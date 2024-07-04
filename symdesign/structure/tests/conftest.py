from pathlib import Path

import pytest


@pytest.fixture(scope='module')
def resource_path() -> Path:
    return Path(__file__).parent / 'resources'


@pytest.fixture(scope='module')
def c3_directory_path(resource_path: Path) -> Path:
    return resource_path / 'C3'


@pytest.fixture(scope='module')
def d4_directory_path(resource_path: Path) -> Path:
    return resource_path / 'D4'


@pytest.fixture(scope='module')
def d2_directory_path(resource_path: Path) -> Path:
    return resource_path / 'D2'


@pytest.fixture(scope='module')
def t33_directory_path(resource_path: Path) -> Path:
    return resource_path / 'T33'
