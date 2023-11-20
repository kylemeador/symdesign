from __future__ import annotations

import logging
import os
from typing import AnyStr

import numpy as np

from symdesign import utils
putils = utils.path

# Globals
logger = logging.getLogger(__name__)
number_of_nanohedra_components = 2



def retrieve_pose_transformation_from_nanohedra_docking(pose_file: AnyStr) -> list[dict]:
    """Gather pose transformation information for the Pose from Nanohedra output

    Args:
        pose_file: The file containing pose information from Nanohedra output
    Returns:
        The pose transformation arrays as found in the pose_file
    """
    with open(pose_file, 'r') as f:
        pose_transformation = {}
        for line in f.readlines():
            # all parsing lacks PDB number suffix such as PDB1 or PDB2 for hard coding in dict key
            if line[:20] == 'ROT/DEGEN MATRIX PDB':
                # data = eval(line[22:].strip())
                data = [[float(item) for item in group.split(', ')]
                        for group in line[22:].strip().strip('[]').split('], [')]
                pose_transformation[int(line[20:21])] = {'rotation': np.array(data)}
            elif line[:15] == 'INTERNAL Tx PDB':
                try:  # This may have values of None
                    data = np.array([float(item) for item in line[17:].strip().strip('[]').split(', ')])
                except ValueError:  # we received a string which is not a float
                    data = utils.symmetry.origin
                pose_transformation[int(line[15:16])]['translation'] = data
            elif line[:18] == 'SETTING MATRIX PDB':
                # data = eval(line[20:].strip())
                data = [[float(item) for item in group.split(', ')]
                        for group in line[20:].strip().strip('[]').split('], [')]
                pose_transformation[int(line[18:19])]['rotation2'] = np.array(data)
            elif line[:22] == 'REFERENCE FRAME Tx PDB':
                try:  # This may have values of None
                    data = np.array([float(item) for item in line[24:].strip().strip('[]').split(', ')])
                except ValueError:  # we received a string which is not a float
                    data = utils.symmetry.origin
                pose_transformation[int(line[22:23])]['translation2'] = data

    return [pose_transformation[idx] for idx, _ in enumerate(pose_transformation, 1)]


def get_components_from_nanohedra_docking(pose_file: AnyStr) -> list[str]:
    """Gather information on the docking componenet identifiers for the docked Pose from a Nanohedra output

    Args:
        pose_file: The file containing pose information from Nanohedra output
    Returns:
        The names of the models used during Nanohedra
    """
    entity_names = []
    with open(pose_file, 'r') as f:
        for line in f.readlines():
            if line[:15] == 'DOCKED POSE ID:':
                pose_identifier = line[15:].strip().replace('_DEGEN_', '-DEGEN_').replace('_ROT_', '-ROT_').replace('_TX_', '-tx_')
            elif line[:31] == 'Canonical Orientation PDB1 Path':
                canonical_pdb1 = line[31:].strip()
            elif line[:31] == 'Canonical Orientation PDB2 Path':
                canonical_pdb2 = line[31:].strip()

        if pose_identifier:
            entity_names = pose_identifier.split('-DEGEN_')[0].split('-')

        if len(entity_names) != number_of_nanohedra_components:  # probably old format without use of '-'
            entity_names = list(map(os.path.basename, [os.path.splitext(canonical_pdb1)[0],
                                                       os.path.splitext(canonical_pdb2)[0]]))
    return entity_names


def get_sym_entry_from_nanohedra_directory(nanohedra_dir: AnyStr) -> utils.SymEntry.SymEntry:
    """Handles extraction of Symmetry info from Nanohedra outputs.

    Args:
        nanohedra_dir: The path to a Nanohedra master output directory
    Raises:
        FileNotFoundError: If no nanohedra master log file is found
        SymmetryError: If no symmetry is found
    Returns:
        The SymEntry specified by the Nanohedra docking run
    """
    log_path = os.path.join(nanohedra_dir, putils.master_log)
    try:
        with open(log_path, 'r') as f:
            for line in f.readlines():
                if 'Nanohedra Entry Number: ' in line:  # "Symmetry Entry Number: " or
                    return utils.SymEntry.symmetry_factory.get(int(line.split(':')[-1]))  # sym_map inclusion?
    except FileNotFoundError:
        raise FileNotFoundError(
            f'Nanohedra master directory is malformed. Missing required docking file {log_path}')
    raise utils.InputError(
        f"Nanohedra master docking file {log_path} is malformed. Missing required info 'Nanohedra Entry Number:'")
