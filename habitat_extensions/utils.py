import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import quaternion
from habitat.core.utils import try_cv2_import
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.utils.visualizations import maps as habitat_maps
from habitat.utils.visualizations.utils import draw_collision, images_to_video
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from numpy import ndarray

from habitat_extensions import maps

cv2 = try_cv2_import()


def observations_to_image(
    observation: Dict[str, Any], info: Dict[str, Any]
) -> ndarray:
    """Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    observation_size = -1
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"][:, :, :3]
        egocentric_view.append(rgb)

    # draw depth map if observation has depth info. resize to rgb size.
    if "depth" in observation:
        if observation_size == -1:
            observation_size = observation["depth"].shape[0]
        depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        depth_map = cv2.resize(
            depth_map,
            dsize=(observation_size, observation_size),
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view.append(depth_map)

    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)

    frame = egocentric_view

    map_k = None
    if "top_down_map_vlnce" in info:
        map_k = "top_down_map_vlnce"
    elif "top_down_map" in info:
        map_k = "top_down_map"

    if map_k is not None:
        td_map = info[map_k]["map"]

        td_map = maps.colorize_topdown_map(
            td_map,
            info[map_k]["fog_of_war_mask"],
            fog_of_war_desat_amount=0.75,
        )
        td_map = habitat_maps.draw_agent(
            image=td_map,
            agent_center_coord=info[map_k]["agent_map_coord"],
            agent_rotation=info[map_k]["agent_angle"],
            agent_radius_px=min(td_map.shape[0:2]) // 24,
        )
        if td_map.shape[1] < td_map.shape[0]:
            td_map = np.rot90(td_map, 1)

        if td_map.shape[0] > td_map.shape[1]:
            td_map = np.rot90(td_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = td_map.shape
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        td_map = cv2.resize(
            td_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((egocentric_view, td_map), axis=1)
    return frame


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[ndarray],
    episode_id: Union[str, int],
    checkpoint_idx: int,
    metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    fps: int = 10,
) -> None:
    """Generate video according to specified information. Using a custom
    verion instead of Habitat's that passes FPS to video maker.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v:.2f}")

    video_name = f"episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(
        metric_strs
    )
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name, fps=fps)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", checkpoint_idx, images, fps=fps
        )


def heading_from_quaternion(quat: quaternion.quaternion) -> float:
    # https://github.com/facebookresearch/habitat-lab/blob/v0.1.7/habitat/tasks/nav/nav.py#L356
    heading_vector = quaternion_rotate_vector(
        quat.inverse(), np.array([0, 0, -1])
    )
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return phi % (2 * np.pi)


def dtw(x, y, dist, warp=1, w=np.inf, s=1.0):
    """
    Copied from: https://github.com/pollen-robotics/dtw/tree/b2c7514bff2abccb40c5f6d65c3c6eae18988bbe

    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """

    def _traceback(D):
        i, j = np.array(D.shape) - 2
        p, q = [i], [j]
        while (i > 0) or (j > 0):
            tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
            if tb == 0:
                i -= 1
                j -= 1
            elif tb == 1:
                i -= 1
            else:  # (tb == 2):
                j -= 1
            p.insert(0, i)
            q.insert(0, j)
        return np.array(p), np.array(q)

    assert len(x)
    assert len(y)
    assert math.isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not math.isinf(w):
        D0 = np.full((r + 1, c + 1), np.inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w) : min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = np.zeros((r + 1, c + 1))
        D0[0, 1:] = np.inf
        D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if math.isinf(w) or (max(0, i - w) <= j <= min(c, i + w)):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not math.isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), np.zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path
