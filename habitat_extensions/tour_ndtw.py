from collections import defaultdict
from typing import Dict, List

import dtw  # https://github.com/DynamicTimeWarping/dtw-python
import numpy as np


def compute_episodes_per_tour(tours: Dict[str, List]) -> Dict[str, int]:
    """Determine the number of episodes in each tour."""
    eps_per_tour = defaultdict(int)
    for tour_id, path in tours.items():
        for i in range(1, len(path)):
            if path[i]["episode_id"] != path[i - 1]["episode_id"]:
                eps_per_tour[tour_id] += 1

    return eps_per_tour


def window_align_func(iw, jw, query_size, reference_size, alignments):
    window = np.ones((query_size, reference_size), dtype=np.bool)

    # for each alignment (i, j), make col j False except (i,j)
    for (i, j) in alignments:
        window[:, j] = False
        window[i, j] = True

    return window


def extract_ep_order(path):
    eps = [p["episode_id"] for p in path]
    eps_single = []
    for i in range(1, len(eps)):
        if eps[i - 1] != eps[i]:
            eps_single.append(eps[i - 1])
    eps_single.append(eps[-1])
    return eps_single


def alignments_from_paths(agent_path, gt_path):
    gt_path = [p for p in gt_path if p["phase"] == "agent"]
    agent_path = [p for p in agent_path if p["phase"] == "agent"]

    assert extract_ep_order(gt_path) == extract_ep_order(
        agent_path
    ), "agent and GT episode orders do not match."

    alen = len(agent_path)
    gtlen = len(gt_path)

    agent_alignment_points = []
    for i in range(1, alen):
        if agent_path[i]["episode_id"] != agent_path[i - 1]["episode_id"]:
            agent_alignment_points.append(i - 1)  # stopping point
            agent_alignment_points.append(i)  # starting point

    gt_alignment_points = []
    for i in range(1, gtlen):
        if gt_path[i]["episode_id"] != gt_path[i - 1]["episode_id"]:
            gt_alignment_points.append(i - 1)  # stopping point
            gt_alignment_points.append(i)  # starting point

    assert len(agent_alignment_points) == len(
        gt_alignment_points
    ), "mismatch in number of alignment points."
    return list(zip(agent_alignment_points, gt_alignment_points))


def novel_only(path):
    """Ignore steps where the agent does not change position"""
    if len(path) == 0:
        return path

    new_path = [path[0]]
    if len(path) == 1:
        return path

    for i in range(1, len(path)):
        if path[i - 1] != path[i]:
            new_path.append(path[i])
    return new_path


def aggregate_scores(t_ndtws, episodes_per_tour):
    """aggregate tour scores to a dataset split score weighted by episode count"""
    ndtw_score = 0
    total_eps = sum(episodes_per_tour.values())
    for tour_id, tndtw in t_ndtws.items():
        ndtw_score += tndtw * (episodes_per_tour[tour_id] / total_eps)

    return ndtw_score


def compute_tour_ndtw(
    agent_paths: Dict[str, List],
    gt_paths: Dict[str, List],
    success_distance: float = 3.0,
    verbose: bool = False,
) -> float:
    """Compute an aggregated nDTW score for a dataset split."""
    if not set(gt_paths.keys()) == set(agent_paths.keys()):
        raise ValueError("tours are different")

    if verbose:
        print("t-ndtw   len(tour)")

    # compute the constrained ndtw of each tour
    t_ndtws = {}
    for tour_id, agent_path in agent_paths.items():
        agent_path = novel_only(agent_path)
        gt_path = novel_only(gt_paths[tour_id])
        gt_path = gt_paths[tour_id]

        alignments = alignments_from_paths(agent_path, gt_path)

        ap = [p["position"] for p in agent_path if p["phase"] == "agent"]
        gtp = [p["position"] for p in gt_path if p["phase"] == "agent"]
        dtw_dist = dtw.dtw(
            ap,
            gtp,
            step_pattern="symmetric1",
            window_type=window_align_func,
            window_args={"alignments": alignments},
        ).distance
        t_ndtws[tour_id] = np.exp(-dtw_dist / (len(gtp) * success_distance))
        if verbose:
            print(round(t_ndtws[tour_id], 4), "\t", len(gtp))

    episodes_per_tour = compute_episodes_per_tour(gt_paths)
    return aggregate_scores(t_ndtws, episodes_per_tour)
