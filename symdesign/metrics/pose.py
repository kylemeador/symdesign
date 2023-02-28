from collections import defaultdict
from typing import Iterable, Callable

from symdesign.structure.fragment import GhostFragment, Fragment
from symdesign.structure.fragment.db import FragmentDatabase, nanohedra_fragment_match_score
# from symdesign.structure.model import Pose
import symdesign.structure


def get_center_match_scores(fragment_info: list[tuple[GhostFragment, Fragment, float]]):
    match_scores = defaultdict(list)
    for ghost_frag, surf_frag, match_score in fragment_info:
        match_scores[ghost_frag.index].append(match_score)
        match_scores[surf_frag.index].append(match_score)

    return match_scores


def get_total_match_scores(fragment_info: list[tuple[GhostFragment, Fragment, float]], frag_db: FragmentDatabase):
    match_scores = defaultdict(list)
    for ghost_frag, surf_frag, match_score in fragment_info:
        # match_scores[ghost_frag.index].append(match_score)
        # match_scores[surf_frag.index].append(match_score)
        for idx1, idx2 in [(ghost_frag.index + j, surf_frag.index + j) for j in frag_db.fragment_range]:
            match_scores[idx1].append(match_score)
            match_scores[idx2].append(match_score)

    return match_scores


def nanohedra_score(pose: 'structure.model.Pose'):
    total_match_scores = get_total_match_scores(pose.fragment_pairs, pose.fragment_db)
    return nanohedra_fragment_match_score(total_match_scores)


def nanohedra_score_center(pose: 'structure.model.Pose'):
    center_match_scores = get_center_match_scores(pose.fragment_pairs)
    return nanohedra_fragment_match_score(center_match_scores)


def nanohedra_score_normalized(pose: 'structure.model.Pose'):
    total_match_scores = get_total_match_scores(pose.fragment_pairs, pose.fragment_db)
    return nanohedra_fragment_match_score(total_match_scores) / len(total_match_scores)


def nanohedra_score_center_normalized(pose: 'structure.model.Pose'):
    center_match_scores = get_center_match_scores(pose.fragment_pairs)
    return nanohedra_fragment_match_score(center_match_scores) / len(center_match_scores)


pose_metric_map = {
    'nanohedra_score': nanohedra_score,
    'nanohedra_score_center': nanohedra_score_center,
    'nanohedra_score_normalized': nanohedra_score_normalized,
    'nanohedra_score_center_normalized': nanohedra_score_center_normalized,
}
"""This is a mapping of the Nanohedra or Pose based measure to the method to retrieve it from the Pose"""


def format_metric_functions(scores: Iterable[str]) -> dict[str, Callable]:
    score_functions = {}
    for score in scores:
        pose_func: Callable = pose_metric_map.get(score)
        if pose_func:
            score_functions[score] = pose_func

    return score_functions
