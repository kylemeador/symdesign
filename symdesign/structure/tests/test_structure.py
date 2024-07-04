from pathlib import Path

import pytest

from symdesign.structure import Entity, Pose
from symdesign.structure.base import SymmetryBase


@pytest.fixture(scope='module')
def c3_pose(c3_directory_path: Path) -> Pose:
    return Pose.from_file(c3_directory_path / '2gtr.pdb1', symmetry='C3')


@pytest.fixture(scope='module')
def t33_pose(t33_directory_path: Path) -> Pose:
    return Pose.from_file(t33_directory_path / 'T33-ml34.pdb', symmetry='T33')


def test_pose_parent(c3_pose: Pose, t33_pose: Pose):
    assert c3_pose.is_parent()
    assert not c3_pose.is_dependent()
    assert all(e.is_dependent() for e in c3_pose.chains)
    assert all(e.is_dependent() for e in c3_pose.entities)

    assert t33_pose.is_parent()
    assert not t33_pose.is_dependent()
    assert all(e.is_dependent() for e in t33_pose.entities)
    assert all(e is c for e in t33_pose.entities for c in e.chains[:1])
    assert all(c.is_dependent() for e in t33_pose.entities for c in e.chains[:1])
    assert all(c.is_parent() for e in t33_pose.entities for c in e.chains[1:])


def test_symmetry_base():
    sb = SymmetryBase()
    sb.symmetry = "C3"
    assert sb.is_symmetric()
    sb.symmetry = "C1"
    assert sb.symmetry is None
    assert not sb.is_symmetric()
    assert not sb.has_symmetric_dependents()


def test_pose_symmetric_dependents(c3_pose: Pose, t33_pose: Pose):
    assert c3_pose.symmetric_dependents == []
    assert t33_pose.has_symmetric_dependents()
    assert all(isinstance(d, Entity) for d in t33_pose.symmetric_dependents)
    t33_pose.symmetric_dependents = None
    assert not t33_pose.has_symmetric_dependents()


def test_pose_symmetry(c3_pose: Pose, t33_pose: Pose):
    assert c3_pose.symmetry == 'C3'
    assert c3_pose.sym_entry.number == 135
    assert t33_pose.symmetry == 'T33'
    assert t33_pose.sym_entry.number == 54
