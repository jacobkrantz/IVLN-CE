import os
from functools import lru_cache
from typing import Optional

import cv2
import imutils
import numpy as np
import torch
from habitat.utils.visualizations import maps as habitat_maps
from scipy import ndimage

label_colours = [
    (0, 0, 0),
    (106, 137, 204),  # shelving
    (230, 126, 34),  # chest of drawers
    (7, 153, 146),  # bed
    (248, 194, 145),  # cushion
    (76, 209, 55),  # fireplace
    (255, 168, 1),  # sofa
    (184, 233, 148),  # table
    (39, 174, 96),  # chair
    (229, 80, 57),  # cabinet
    (30, 55, 153),  # plant
    (24, 220, 255),  # (56, 173, 169),    # counter
    (234, 32, 39),  # sink
]


def color_label(label):
    is_tensor = False
    if torch.is_tensor(label):
        is_tensor = True
        label = label.clone().cpu().data.numpy()

    colored_label = np.vectorize(lambda x: label_colours[int(x)])

    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    if not is_tensor:
        return colored
    try:
        return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    except ValueError:
        return torch.from_numpy(colored[np.newaxis, ...])


def visualize_semantic_map(semantic_map: torch.Tensor):
    map_semantic = semantic_map.cpu().numpy().astype(np.int32)
    map_semantic = color_label(map_semantic)
    map_semantic = map_semantic.transpose(1, 2, 0)
    map_semantic = map_semantic.astype(np.uint8)
    return map_semantic


def visualize_occupancy_map(occupancy_map: torch.Tensor):
    occupancy_map = occupancy_map * 255
    occupancy_map = 255 - occupancy_map
    occupancy_map = torch.stack([occupancy_map] * 3, 2)
    return occupancy_map.cpu().numpy()


def visualize_maps(
    occupancy_maps,
    semantic_maps,
    folder,
    counter,
    rgbs=None,
):
    for n, (occupancy_map, semantic_map, rgb) in enumerate(
        zip(occupancy_maps, semantic_maps, rgbs)
    ):
        map_image = visualize_map(  # noqa: F821
            occupancy_map, semantic_map, rgb
        )
        map_image = imutils.resize(
            map_image, width=1000, inter=cv2.INTER_NEAREST
        )
        filename = make_filename(folder, n, counter)
        cv2.imwrite(filename, map_image)


def make_filename(folder: str, batch_no: int, counter: int) -> str:
    new_folder = os.path.join(folder, f"{batch_no:03d}")
    if not os.path.isdir(new_folder):
        os.makedirs(new_folder)
    filename = os.path.join(new_folder, f"{counter:05d}.png")
    return filename


def visualize_egocentric_map(maps, map_visualizer_func, min_width=200):
    output_maps = []
    for map_ in maps:
        map_ = map_visualizer_func(map_)
        map_ = imutils.resize(map_, width=min_width, inter=cv2.INTER_NEAREST)
        map_ = visualize_egocentric_agent(map_)
        output_maps.append(map_)
    return np.stack(output_maps)


def visualize_ego_occupancy_map(occupancy_map):
    return visualize_egocentric_map(occupancy_map, visualize_occupancy_map)


def visualize_ego_semantic_map(semantic_map):
    return visualize_egocentric_map(semantic_map, visualize_semantic_map)


def visualize_egocentric_agent(map_semantic_color):
    row, col = map_semantic_color.shape[:2]
    agent_sprite = get_agent_sprite(max(row, col))
    map_semantic_color = place_agent_at_pixel(
        map_semantic_color, agent_sprite, row // 2, col // 2
    )
    return map_semantic_color


def visualize_allocentric_agent(map_semantic_color, discrete_pose):
    agent_sprite = get_agent_sprite(max(list(map_semantic_color.shape[:2])))
    row, col, heading = discrete_pose
    agent_sprite = ndimage.rotate(agent_sprite, np.rad2deg(heading))
    map_semantic_color = place_agent_at_pixel(
        map_semantic_color, agent_sprite, row, col
    )
    return map_semantic_color


def append_image_horizontally(imgs: list, add_borders=True):
    imgs = [imutils.resize(img, width=640, height=320) for img in imgs]
    if add_borders:
        imgs = [add_border(i) for i in imgs]
    return np.concatenate(imgs, 1)


def append_image_vertically(imgs: list, add_borders=True):
    imgs = [cv2.resize(img, (640, 320)) for img in imgs]
    if add_borders:
        imgs = [add_border(i) for i in imgs]
    return np.concatenate(imgs, 0)


def add_border(map_semantic: np.uint8, border: Optional[np.uint8] = None):
    if border is None:
        border = np.uint8([255, 255, 255])

    map_semantic[:, 0] = border
    map_semantic[0, :] = border
    map_semantic[:, -1] = border
    map_semantic[-1, :] = border
    return map_semantic


@lru_cache(maxsize=1)
def get_agent_sprite(length):
    sprite = habitat_maps.AGENT_SPRITE[:, :, :3]
    sprite_length = int(0.05 * length)
    sprite = cv2.resize(sprite, (sprite_length, sprite_length))
    sprite = ndimage.rotate(sprite, 180)
    return sprite


def place_agent_at_pixel(map_, agent, row, col, BACKGROUND=None):
    if BACKGROUND is None:
        BACKGROUND = np.uint8([0, 0, 0])

    r_min, r_max, c_min, c_max = compute_bounding_box(row, col, agent.shape)
    if (r_min >= 0 and r_max < map_.shape[0]) and (
        c_min >= 0 and c_max < map_.shape[1]
    ):
        ignore_black_mask = agent != BACKGROUND
        map_[r_min:r_max, c_min:c_max][ignore_black_mask] = agent[
            ignore_black_mask
        ]
    return map_


def compute_bounding_box(center_row, center_col, agent_shape):
    r_min = int(center_row) - agent_shape[0] // 2
    c_min = int(center_col) - agent_shape[1] // 2
    r_max = r_min + agent_shape[0]
    c_max = c_min + agent_shape[1]
    return r_min, r_max, c_min, c_max
