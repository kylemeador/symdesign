"""Provide objects for manipulation and modeling of protein structures

This module allows the user to load protein structural files, manipulate their coordinates, and measure aspects of their
 position, relationships, and shape properties. Additionally, identify symmetry in the structures and inherently
 model symmetric relationships

The module contains the following classes:

- `Pose()` - Create a symmetry-aware object to manipulate a collection of Entity instances
- `Model()` - Create an object to manipulate a collection of Chain instances
- `Entity()` - Create a symmetry-aware object to manipulate structurally homologous Chain instances

The module contains the following functions:

"""
from . import utils, coordinates, fragment, sequence
from . import base
from . import model
from .model import Entity, Model, Pose, Structure