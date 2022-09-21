"""
The purpose of this script is to give intuitions on
how to build the GT topdown semantic maps by projecting
egocentric GT semantic labels.
Note: In the paper we didn't use this strategy, rather
we projected the semantic mesh directly.
Check : build_semmap_from_obj_pc.py
"""

import math
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache, reduce
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch_scatter import scatter_max

from ivlnce_baselines.common.mapping_module.projector import (
    PointCloud,
    _transform3D,
)
from ivlnce_baselines.common.mapping_module.rednet import RedNet


def linearize(x):
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(-1, x.shape[3])
    return x.squeeze()


def rotate_around_y_matrix(angle: torch.FloatTensor):
    """https://en.wikipedia.org/wiki/Rotation_matrix"""
    rotatation_matrix = torch.zeros(
        (angle.shape[0], 3, 3), device=angle.device
    )
    rotatation_matrix[:, 0, 0] = torch.cos(angle)
    rotatation_matrix[:, 0, 2] = torch.sin(angle)
    rotatation_matrix[:, 1, 1] = 1
    rotatation_matrix[:, 2, 0] = -torch.sin(angle)
    rotatation_matrix[:, 2, 2] = torch.cos(angle)
    return rotatation_matrix


def generate_batch_indices(batched_data: torch.Tensor):
    data_shape = batched_data.shape
    batch_indices = torch.arange(data_shape[0], device=batched_data.device)
    batch_indices = batch_indices.view(-1, 1, 1, 1)
    batch_indices = batch_indices.expand(
        data_shape[0], 1, data_shape[2], data_shape[3]
    )
    return batch_indices


@dataclass
class EpisodesInfo:
    not_done_masks: torch.LongTensor
    env_names: List[str]
    EPISODE_FINISHED: int = field(default=0, init=False)
    EPISODE_UNFINISHED: int = field(default=1, init=False)

    def __post_init__(self):
        self.env_names = np.asarray(self.env_names)
        self.not_done_masks = self.not_done_masks.squeeze(1)
        self.indices = torch.arange(
            self.not_done_masks.shape[0], device=self.not_done_masks.device
        )

    def finished(self) -> torch.BoolTensor:
        return self.not_done_masks == self.EPISODE_FINISHED

    def unfinished(self) -> torch.BoolTensor:
        return self.not_done_masks == self.EPISODE_UNFINISHED

    def finished_indices(self) -> torch.LongTensor:
        return self.indices[self.finished()]

    @property
    def num_envs(self):
        return len(self.not_done_masks)


@dataclass
class MapDimensions:
    height_meters: float
    width_meters: float
    resolution_meters: float
    num_rows: int = field(init=False)
    num_cols: int = field(init=False)

    def __post_init__(self):
        self.num_rows = math.ceil(self.height_meters / self.resolution_meters)
        self.num_cols = math.ceil(self.width_meters / self.resolution_meters)

    def meters_to_cell_index(
        self, meter_values: torch.FloatTensor
    ) -> torch.LongTensor:
        return (meter_values / self.resolution_meters).round().long()

    def valid_map_bounds(self, rows, cols):
        valid_rows = torch.logical_and(rows >= 0, rows < self.num_rows)
        valid_cols = torch.logical_and(cols >= 0, cols < self.num_cols)
        return torch.logical_and(valid_rows, valid_cols)

    def project_to_map_indices(self, rows_meters, cols_meters):
        rows = self.meters_to_cell_index(rows_meters + self.height_meters / 2)
        cols = self.meters_to_cell_index(cols_meters + self.width_meters / 2)
        return rows, cols


@dataclass
class State:
    pose: torch.FloatTensor = None
    elevation: torch.FloatTensor = None
    heading: torch.FloatTensor = None

    def copy(self):
        return deepcopy(self)


class RobotCurrentState(State):
    @property
    def height(self):
        return self.pose[:, 1]

    def get_camera_matrix(self):
        return _transform3D(
            self.pose,
            self.elevation + torch.pi,
            self.heading,
            device=self.pose.device,
        )


@dataclass
class RobotStartState(State):
    def __post_init__(self):
        self.batch_size = None

    def update_data(self, batch_size: int, device: torch.device):
        if self.batch_size != batch_size:
            if self.batch_size is None or batch_size > self.batch_size:
                self.init_data(batch_size, device)
            else:
                self.reduce_batch_size(batch_size)
            self.batch_size = batch_size

    def init_data(self, batch_size: int, device: torch.device):
        self.pose = torch.zeros((batch_size, 3), device=device)
        self.elevation = torch.zeros((batch_size), device=device)
        self.heading = torch.zeros((batch_size), device=device)

    def reduce_batch_size(self, batch_size: int):
        self.pose = self.pose[:batch_size]
        self.elevation = self.elevation[:batch_size]
        self.heading = self.heading[:batch_size]

    def update(
        self,
        episodes_info: EpisodesInfo,
        current_state: RobotCurrentState,
    ):
        self.update_data(
            batch_size=episodes_info.num_envs,
            device=current_state.pose.device,
        )
        if episodes_info.finished().any():
            episodes_finished = episodes_info.finished()
            self.pose[episodes_finished] = current_state.pose[
                episodes_finished
            ]


@dataclass
class LocalizeRobot:
    current_state: RobotCurrentState = RobotCurrentState()
    start_state: RobotStartState = RobotStartState()

    def __call__(
        self,
        episodes_info: EpisodesInfo,
        robot_current_state: RobotCurrentState,
    ):
        self.current_state = robot_current_state
        self.start_state.update(episodes_info, robot_current_state)
        return self


@dataclass
class Observations:
    semantics: torch.LongTensor
    depth_normalized: torch.FloatTensor
    rgb: torch.FloatTensor
    # predicted_semantics: torch.LongTensor = field(init=False, default=None)


@dataclass
class Pointcloud:
    batch_indices: torch.LongTensor = None
    xyz: torch.FloatTensor = None

    def __len__(self):
        return self.xyz.shape[0]

    @property
    def x(self):
        return self.xyz[:, 0]

    @property
    def heights(self):
        return self.xyz[:, 1]

    @property
    def z(self):
        return self.xyz[:, 2]

    def copy(self):
        return deepcopy(self)

    def concatenate(self, pointcloud):
        self.xyz = torch.cat((self.xyz, pointcloud.xyz))
        self.batch_indices = torch.cat(
            (self.batch_indices, pointcloud.batch_indices)
        )

    def index(self, masks):
        self.xyz = self.xyz[masks]
        self.batch_indices = self.batch_indices[masks]

    def remove_invalid_depth_values(
        self, depth_normalized, depth_min, depth_max
    ):
        masks = torch.logical_and(
            depth_normalized > depth_min,
            depth_normalized < depth_max,
        )
        self.index(masks)

    def remove_invalid_height_values(
        self, robot_height, delta_height_min, delta_height_max
    ):
        robot_height = robot_height[self.batch_indices]
        masks = torch.logical_and(
            self.heights > (robot_height - delta_height_min),
            self.heights < (robot_height + delta_height_max),
        )
        self.index(masks)

    def translate(self, new_origin: torch.FloatTensor):
        self.xyz += new_origin[self.batch_indices]

    def rotate_around_y(self, heading: torch.FloatTensor):
        matrix = rotate_around_y_matrix(heading)[self.batch_indices]
        xyz_values = self.xyz.unsqueeze(-1)
        xyz_values = torch.bmm(matrix, xyz_values)
        self.xyz = xyz_values.squeeze(-1)

    def shift_origin(self, origin: State):
        self.translate(-origin.pose)
        self.rotate_around_y(-origin.heading)


@dataclass
class SemanticPointcloud(Pointcloud):
    batch_indices: torch.LongTensor = None
    xyz: torch.FloatTensor = None
    semantics: torch.LongTensor = None

    def index(self, masks):
        super().index(masks)
        self.semantics = self.semantics[masks]

    def concatenate(self, pointcloud: Pointcloud):
        super().concatenate(pointcloud)
        self.semantics = torch.cat((self.semantics, pointcloud.semantics))

    @classmethod
    def from_npz_file(cls, npz_file, device, batch_ndx=0):
        with np.load(npz_file) as f:
            xyz = f["xyz"]
            semantic = f["semantics"]
            batch_indices = np.zeros_like(semantic, dtype=np.int32) + batch_ndx

        return cls(
            batch_indices=torch.LongTensor(batch_indices).to(device),
            xyz=torch.FloatTensor(xyz).to(device),
            semantics=torch.LongTensor(semantic).to(device).to(torch.uint8),
        )


@dataclass
class WorldSemanticPointcloud(SemanticPointcloud):
    def concatenate(self, semantic_pointcloud: SemanticPointcloud):
        if self.xyz is None:
            self.init_from_pointcloud(semantic_pointcloud)
        else:
            super().concatenate(semantic_pointcloud)

    def init_from_pointcloud(self, semantic_pointcloud: SemanticPointcloud):
        self.xyz = semantic_pointcloud.xyz
        self.batch_indices = semantic_pointcloud.batch_indices
        self.semantics = semantic_pointcloud.semantics

    def clear_completed_episode_data(self, episodes_info: EpisodesInfo):
        if self.xyz is not None:
            self.clear_paused_episodes(episodes_info)
            self.clear_finished_episodes(episodes_info)

    def clear_paused_episodes(self, episodes_info: EpisodesInfo):
        finished = self.batch_indices >= episodes_info.num_envs
        if finished.any():
            self.index(torch.logical_not(finished))

    def clear_finished_episodes(self, episodes_info: EpisodesInfo):
        if episodes_info.finished().any():
            masked_unfinished_indices = compute_masked_unfinished(
                self.batch_indices,
                episodes_info.finished_indices(),
            )
            self.index(masked_unfinished_indices)


def compute_masked_unfinished(batch_indices, finished_indices):
    batch_finished_indices = reduce(
        torch.logical_or, (batch_indices == fi for fi in finished_indices)
    )
    return torch.logical_not(batch_finished_indices)


@dataclass
class CameraParameters:
    vertical_fov_radians: float
    features_spatial_dimensions: tuple
    height_clip: float


class NormalizedDepthToPointcloudTransformation(nn.Module):
    def __init__(
        self,
        camera_parameters: CameraParameters,
        device,
    ):
        super(NormalizedDepthToPointcloudTransformation, self).__init__()
        self.point_cloud = None
        self.batch_size = None
        self.camera_parameters = camera_parameters
        self.device = device

    def create_depth_projector(self, batch_size):
        if self.batch_size != batch_size:
            self.batch_size = batch_size
            self.point_cloud = PointCloud(
                self.camera_parameters,
                batch_size=batch_size,
                world_shift_origin=torch.FloatTensor([0, 0, 0]).to(
                    self.device
                ),
                device=self.device,
            )

    def forward(
        self,
        depth_normalized: torch.FloatTensor,
        robot_current_state: RobotCurrentState,
    ) -> Pointcloud:

        self.create_depth_projector(depth_normalized.shape[0])
        pointcloud_xyz = self.point_cloud.egocentric_depth_to_point_cloud(
            to_depth_meters(depth_normalized),
            robot_current_state.get_camera_matrix(),
        )
        return pointcloud_xyz


def to_depth_meters(
    depth_normalized: torch.FloatTensor, NORMALIZATION_FACTOR: int = 10
):
    return depth_normalized * NORMALIZATION_FACTOR


class GenerateSemanticPointCloud(nn.Module):
    def __init__(
        self, camera_parameters: CameraParameters, device: torch.device
    ):
        super(GenerateSemanticPointCloud, self).__init__()
        self.normalized_depth_to_pointcloud = (
            NormalizedDepthToPointcloudTransformation(
                camera_parameters, device
            )
        )

    def forward(
        self,
        observations: Observations,
        robot_current_state: RobotCurrentState,
    ) -> SemanticPointcloud:
        pointcloud_xyz = self.normalized_depth_to_pointcloud(
            observations.depth_normalized, robot_current_state
        )
        batch_indices = generate_batch_indices(pointcloud_xyz)

        batch_indices_flat = linearize(batch_indices)
        pointcloud_xyz_flat = linearize(pointcloud_xyz)
        semantics_flat = linearize(observations.semantics)
        semantic_pointcloud = SemanticPointcloud(
            batch_indices_flat, pointcloud_xyz_flat, semantics_flat
        )

        depth_normalized_flat = linearize(observations.depth_normalized)
        semantic_pointcloud.remove_invalid_depth_values(
            depth_normalized_flat, depth_min=0.01, depth_max=0.99
        )

        semantic_pointcloud.remove_invalid_height_values(
            robot_current_state.height,
            delta_height_min=1.0,
            delta_height_max=0.5,
        )
        return semantic_pointcloud


class KeepHighestSemanticPointcloud(nn.Module):
    def __init__(self, map_resolution_meters: float):
        super(KeepHighestSemanticPointcloud, self).__init__()
        self.map_resolution_meters = map_resolution_meters

    def forward(
        self, semantic_pointcloud: SemanticPointcloud
    ) -> SemanticPointcloud:
        if len(semantic_pointcloud) > 0:

            flattened_indices = self.extract_flattened_indices(
                semantic_pointcloud
            )
            argmax_order = self.find_highest_points_indices(
                semantic_pointcloud.heights, flattened_indices
            )
            semantic_pointcloud.index(argmax_order)

        return semantic_pointcloud

    def extract_flattened_indices(
        self, semantic_pointcloud: SemanticPointcloud
    ):
        bndxs, rows, cols = self.get_discrete_indices(semantic_pointcloud)
        flattened_indices = self.to_flattened_indices(bndxs, rows, cols)
        return flattened_indices

    def get_discrete_indices(self, semantic_pointcloud: SemanticPointcloud):
        bndxs = semantic_pointcloud.batch_indices
        rows = self.discrete_positive_index(semantic_pointcloud.z)
        cols = self.discrete_positive_index(semantic_pointcloud.x)
        return bndxs, rows, cols

    def discrete_positive_index(
        self, values: torch.FloatTensor
    ) -> torch.LongTensor:
        values = (values / (self.map_resolution_meters / 2)).round().long()
        values = values - values.min()
        return values

    def to_flattened_indices(self, bndxs, rows, cols):
        return bndxs * (rows.max() * cols.max()) + rows * cols.max() + cols

    def find_highest_points_indices(self, heights, flattened_indices):
        _, argmax_order = scatter_max(heights, flattened_indices)
        argmax_order = argmax_order[argmax_order != flattened_indices.shape[0]]
        return argmax_order


@dataclass
class SparseMap:
    batch_ndxs: torch.LongTensor
    rows: torch.LongTensor
    cols: torch.LongTensor
    values: torch.LongTensor

    def index(self, masks: torch.BoolTensor):
        self.batch_ndxs = self.batch_ndxs[masks]
        self.rows = self.rows[masks]
        self.cols = self.cols[masks]
        self.values = self.values[masks]

    def exclude_semantic_labels(self, excluded_labels):
        included_labels = self.values != excluded_labels
        self.index(included_labels)


class AbstractMap(ABC):
    @abstractmethod
    def update(
        self,
        episodes_info: EpisodesInfo,
        semantic_point_cloud: SemanticPointcloud,
        robot_current_state: RobotCurrentState,
    ):
        pass


@dataclass
class DenseMap(AbstractMap):
    map_dimensions: MapDimensions
    device: torch.device
    data: torch.ByteTensor = None
    batch_size: int = None

    def discretize_semantic_pointcloud(
        self, semantic_pointcloud: SemanticPointcloud
    ) -> SparseMap:
        rows, cols = self.map_dimensions.project_to_map_indices(
            rows_meters=semantic_pointcloud.z,
            cols_meters=semantic_pointcloud.x,
        )

        sparse_map = SparseMap(
            semantic_pointcloud.batch_indices,
            rows,
            cols,
            semantic_pointcloud.semantics,
        )

        in_bounds = self.map_dimensions.valid_map_bounds(rows, cols)
        sparse_map.index(in_bounds)

        return sparse_map

    def create_data(self, batch_size):
        if self.batch_size != batch_size:
            if self.batch_size is None or batch_size > self.batch_size:
                self.init_data(batch_size)
            else:
                self.reduce_batch_size(batch_size)
            self.batch_size = batch_size

    def init_data(self, batch_size):
        self.data = torch.zeros(
            (
                batch_size,
                self.map_dimensions.num_rows,
                self.map_dimensions.num_cols,
            ),
            device=self.device,
            dtype=torch.uint8,
        )

    def reduce_batch_size(self, batch_size):
        self.data = self.data[:batch_size]

    def update(
        self,
        episodes_info: EpisodesInfo,
        world_semantic_pointcloud: SemanticPointcloud,
        robot_current_state: RobotCurrentState,
    ):
        self.create_data(batch_size=episodes_info.num_envs)
        map_semantic_pointcloud = world_semantic_pointcloud.copy()
        map_semantic_pointcloud.shift_origin(robot_current_state)
        sparse_map = self.discretize_semantic_pointcloud(
            map_semantic_pointcloud
        )
        return sparse_map

    def update_map(self, b, r, c, values):
        self.data.fill_(0)
        self.data[b, r, c] = values


class OccupancyStatus:
    FREE = 0
    OCCUPIED = 1


class OccupancyMapMemory(DenseMap):
    def update(
        self,
        episodes_info: EpisodesInfo,
        world_semantic_pointcloud: SemanticPointcloud,
        robot_state: RobotCurrentState,
    ):
        sparse_map = super().update(
            episodes_info, world_semantic_pointcloud, robot_state
        )
        self.update_map(
            sparse_map.batch_ndxs,
            sparse_map.rows,
            sparse_map.cols,
            OccupancyStatus.OCCUPIED,
        )


class SemanticLabels:
    FLOOR = 0


class SemanticMapMemory(DenseMap):
    def update(
        self,
        episodes_info: EpisodesInfo,
        world_semantic_pointcloud: SemanticPointcloud,
        robot_state: RobotCurrentState,
    ):
        sparse_map = super().update(
            episodes_info, world_semantic_pointcloud, robot_state
        )
        sparse_map.exclude_semantic_labels(SemanticLabels.FLOOR)
        self.update_map(
            sparse_map.batch_ndxs,
            sparse_map.rows,
            sparse_map.cols,
            sparse_map.values,
        )


class OccupancySemanticMapMemory(AbstractMap):
    def __init__(self, map_dimensions: MapDimensions, device: torch.device):
        self.occupancy_map = OccupancyMapMemory(map_dimensions, device)
        self.semantic_map = SemanticMapMemory(map_dimensions, device)

    def update(
        self,
        episodes_info: EpisodesInfo,
        world_semantic_pointcloud: SemanticPointcloud,
        robot_state: RobotCurrentState,
    ):
        self.occupancy_map.update(
            episodes_info, world_semantic_pointcloud, robot_state
        )
        self.semantic_map.update(
            episodes_info, world_semantic_pointcloud, robot_state
        )

    @property
    def occupancy(self):
        return self.occupancy_map.data

    @property
    def semantic(self):
        return self.semantic_map.data

    @property
    def data(self):
        return torch.stack((self.occupancy_data, self.semantic_data), 1)


class ComputeSemantics(nn.Module):
    pass


class GTSemantics(ComputeSemantics):
    def __init__(self):
        super(GTSemantics, self).__init__()

    def forward(self, observations: Observations) -> Observations:
        if observations.semantics is None:
            raise Exception("Semantic Sensor not in use")
        return observations


class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    src: https://github.com/pratogab/batch-transforms/blob/master/batch_transforms.py
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(
        self, mean, std, inplace=False, dtype=torch.float, device="cpu"
    ):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[
            None, :, None, None
        ]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[
            None, :, None, None
        ]
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor


class PredictSemantics(ComputeSemantics):
    """
    Adapted from
    https://github.com/vincentcartillier/Semantic-MapNet/blob/main/demo.py
    """

    def __init__(self):
        super(PredictSemantics, self).__init__()
        self.model = None
        self.rgb_normalization = None
        self.depth_normalization = None

    def setup_normalization(self, observations: Observations):
        if self.rgb_normalization is None:
            depth_shape = observations.depth_normalized.shape
            self.rgb_normalization = transforms.Compose(
                [
                    lambda x: F.interpolate(
                        x,
                        size=(depth_shape[2], depth_shape[3]),
                        mode="bilinear",
                    ),
                    Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        device=observations.rgb.device,
                    ),
                ]
            )
            self.depth_normalization = Normalize(
                mean=[0.213],
                std=[0.285],
                device=observations.depth_normalized.device,
            )

    def setup_finetuned_rednet(self, device: torch.device):
        if self.model is None:
            cfg_rednet = {
                "arch": "rednet",
                "resnet_pretrained": False,
                "finetune": True,
                "SUNRGBD_pretrained_weights": "",
                "n_classes": 13,
                "upsample_prediction": True,
                "load_model": "data/rednet_mp3d_best_model.pkl",
            }
            self.model = RedNet(cfg_rednet).to(device)
            self.load_model(cfg_rednet["load_model"])
            self.model.eval()
            self.freeze_weights()

    def setup(self, observations: Observations):
        self.setup_normalization(observations)
        self.setup_finetuned_rednet(observations.depth_normalized.device)

    def load_model(self, model_path):
        model_state = torch.load(model_path)["model_state"]
        model_state = self.convert_weights_cuda_cpu(model_state, "cpu")
        self.model.load_state_dict(model_state)

    def freeze_weights(self):
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def convert_weights_cuda_cpu(weights, device):
        names = list(weights.keys())
        is_module = names[0].split(".")[0] == "module"
        if device == "cuda" and not is_module:
            new_weights = {"module." + k: v for k, v in weights.items()}
        elif device == "cpu" and is_module:
            new_weights = {
                ".".join(k.split(".")[1:]): v for k, v in weights.items()
            }
        else:
            new_weights = weights
        return new_weights

    @torch.no_grad()
    def forward(self, observations: Observations) -> Observations:
        if observations.rgb is None:
            raise Exception("RGB Sensor not in use")

        self.setup(observations)

        rgb_normalized = self.rgb_normalization(
            observations.rgb.float() / 255.0
        )
        depth_normalized = self.depth_normalization(
            observations.depth_normalized
        )

        predicted_scores = self.model(rgb_normalized, depth_normalized)
        observations.semantics = predicted_scores.argmax(1, keepdims=True).to(
            torch.uint8
        )

        return observations


class AbstractUpdateWorldSemanticPointcloud(nn.Module):
    pass


class UpdateWorldSemanticPointcloud(AbstractUpdateWorldSemanticPointcloud):
    def __init__(
        self,
        camera_parameters: CameraParameters,
        device: torch.device,
        map_dimensions: MapDimensions,
        compute_semantics: ComputeSemantics,
    ):
        super(UpdateWorldSemanticPointcloud, self).__init__()
        self.compute_semantics = compute_semantics
        self.generate_semantic_pointcloud = GenerateSemanticPointCloud(
            camera_parameters, device
        )
        self.world_semantic_pointcloud = WorldSemanticPointcloud()
        self.keep_highest_points = KeepHighestSemanticPointcloud(
            map_dimensions.resolution_meters
        )

    def forward(
        self,
        episodes_info: EpisodesInfo,
        observations: Observations,
        robot_current_state: RobotCurrentState,
    ):
        observations = self.compute_semantics(observations)

        self.world_semantic_pointcloud.clear_completed_episode_data(
            episodes_info
        )

        local_semantic_pointcloud = self.generate_semantic_pointcloud(
            observations, robot_current_state
        )
        local_semantic_pointcloud = self.keep_highest_points(
            local_semantic_pointcloud
        )

        self.world_semantic_pointcloud.concatenate(local_semantic_pointcloud)
        self.world_semantic_pointcloud = self.keep_highest_points(
            self.world_semantic_pointcloud
        )
        return self.world_semantic_pointcloud


class GetGTWorldSemanticPointcloud(AbstractUpdateWorldSemanticPointcloud):
    def __init__(self, device: torch.device, maps_location: str):
        super(GetGTWorldSemanticPointcloud, self).__init__()
        self.world_semantic_pointcloud = WorldSemanticPointcloud()
        self.device = device
        self.maps_location = maps_location

    @lru_cache()
    def get_map_file(self, env_name):
        return os.path.join(self.maps_location, f"{env_name}.npz")

    def forward(
        self,
        episodes_info: EpisodesInfo,
        observations: Observations,
        robot_current_state: RobotCurrentState,
    ):
        self.world_semantic_pointcloud.clear_completed_episode_data(
            episodes_info
        )
        if episodes_info.finished().any():
            finished_indices = episodes_info.finished_indices()
            for batch_ndx in finished_indices:
                env_name = episodes_info.env_names[batch_ndx]
                map_file = self.get_map_file(env_name)
                semantic_pointcloud = SemanticPointcloud.from_npz_file(
                    map_file, self.device, batch_ndx.item()
                )
                self.world_semantic_pointcloud.concatenate(semantic_pointcloud)

        return self.world_semantic_pointcloud


class FilterPointCloudByRobotHeight(nn.Module):
    def __init__(self, delta_height_min=1.25, delta_height_max=0.75):
        super(FilterPointCloudByRobotHeight, self).__init__()
        self.delta_height_min = delta_height_min
        self.delta_height_max = delta_height_max

    def forward(
        self,
        world_semantic_pointcloud: SemanticPointcloud,
        robot_current_state: RobotCurrentState,
    ):
        world_semantic_pointcloud = world_semantic_pointcloud.copy()
        world_semantic_pointcloud.remove_invalid_height_values(
            robot_current_state.height,
            delta_height_min=self.delta_height_min,
            delta_height_max=self.delta_height_max,
        )
        return world_semantic_pointcloud


class MappingModule(nn.Module):
    def __init__(
        self,
        localize_robot: LocalizeRobot,
        update_world_semantic_pointcloud: AbstractUpdateWorldSemanticPointcloud,
        filter_pointcloud_by_robot_height: FilterPointCloudByRobotHeight,
        map_memory: OccupancySemanticMapMemory,
    ):
        super(MappingModule, self).__init__()

        self.localize_robot = localize_robot
        self.update_world_representation = update_world_semantic_pointcloud
        self.filter_pointcloud_by_robot_height = (
            filter_pointcloud_by_robot_height
        )
        self.map_memory = map_memory

    def forward(
        self,
        episodes_info: EpisodesInfo,
        observations: Observations,
        robot_current_state: RobotCurrentState,
    ) -> OccupancySemanticMapMemory:

        robot_state = self.localize_robot(episodes_info, robot_current_state)

        world_semantic_pointcloud = self.update_world_representation(
            episodes_info, observations, robot_state.current_state
        )

        world_semantic_pointcloud = self.filter_pointcloud_by_robot_height(
            world_semantic_pointcloud, robot_state.current_state
        )

        self.map_memory.update(
            episodes_info,
            world_semantic_pointcloud,
            robot_state.current_state,
        )

        return self.map_memory

    def get_world_semantic_pointcloud(self):
        return self.update_world_representation.world_semantic_pointcloud


def create_iterative_mapper(
    device: torch.device,
    camera_parameters: CameraParameters,
    map_dimensions: MapDimensions,
    semantics_module: ComputeSemantics,
) -> MappingModule:
    localize_robot = LocalizeRobot()
    update_world_representation = UpdateWorldSemanticPointcloud(
        camera_parameters, device, map_dimensions, semantics_module
    )
    filter_pointcloud_by_robot_height = FilterPointCloudByRobotHeight()
    map_memory = OccupancySemanticMapMemory(map_dimensions, device)

    return MappingModule(
        localize_robot,
        update_world_representation,
        filter_pointcloud_by_robot_height,
        map_memory,
    )


def create_known_mapper(
    device: torch.device,
    map_dimensions: MapDimensions,
    maps_location: os.path,
) -> MappingModule:
    localize_robot = LocalizeRobot()
    gt_world_representation = GetGTWorldSemanticPointcloud(
        device, maps_location
    )
    filter_pointcloud_by_robot_height = FilterPointCloudByRobotHeight()
    map_memory = OccupancySemanticMapMemory(map_dimensions, device)

    return MappingModule(
        localize_robot,
        gt_world_representation,
        filter_pointcloud_by_robot_height,
        map_memory,
    )


def create_gt_semantics_iterative_mapper(
    device: torch.device,
    camera_parameters: CameraParameters,
    map_dimensions: MapDimensions,
) -> MappingModule:
    return create_iterative_mapper(
        device, camera_parameters, map_dimensions, GTSemantics()
    )


def create_predicted_semantics_iterative_mapper(
    device: torch.device,
    camera_parameters: CameraParameters,
    map_dimensions: MapDimensions,
) -> MappingModule:
    return create_iterative_mapper(
        device, camera_parameters, map_dimensions, PredictSemantics()
    )


def create_gt_semantics_known_mapper(
    device: torch.device,
    map_dimensions: MapDimensions,
) -> MappingModule:
    return create_known_mapper(
        device, map_dimensions, maps_location="data/known_maps/gt_semantics"
    )


def create_predicted_semantics_known_mapper(
    device: torch.device,
    map_dimensions: MapDimensions,
) -> MappingModule:
    return create_known_mapper(
        device,
        map_dimensions,
        maps_location="data/known_maps/predicted_semantics",
    )
