import math
from typing import Dict

import numpy as np
from gym import spaces
from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from torch import Tensor

from ivlnce_baselines.common.mapping_module.mapper import (
    CameraParameters,
    MapDimensions,
    create_gt_semantics_iterative_mapper,
    create_gt_semantics_known_mapper,
    create_predicted_semantics_iterative_mapper,
    create_predicted_semantics_known_mapper,
)
from ivlnce_baselines.common.mapping_module.setup_mapping_module import (
    extract_camera_parameters,
    extract_egocentric_map_parameters,
    setup_inputs_from_obs_dict,
)
from ivlnce_baselines.common.mapping_module.visualize_semantic_map import (
    visualize_ego_occupancy_map,
    visualize_ego_semantic_map,
)


@baseline_registry.register_obs_transformer()
class Mapper(ObservationTransformer):
    def __init__(
        self,
        camera_parameters: CameraParameters,
        map_dimensions: MapDimensions,
        visualize=False,
    ):
        super(Mapper, self).__init__()
        self.camera_parameters = camera_parameters
        self.map_dimensions = map_dimensions
        self.visualize = visualize
        self.mapping_module = None

        # these keys can be deleted after generating semantic maps
        self.keys_to_delete = [
            "world_robot_orientation",
            "world_robot_pose",
            "semantic",
            "semantic12",
            "env_name",
        ]

    def transform_observation_space(
        self,
        observation_space: spaces.Dict,
    ):
        # TODO: if self.visualize, add observation spaces for viz

        r = self.map_dimensions.resolution_meters
        h = self.map_dimensions.height_meters
        w = self.map_dimensions.width_meters
        nrows = math.ceil(h / r)
        ncols = math.ceil(w / r)

        for new_key in ["occupancy_map", "semantic_map"]:
            observation_space.spaces[new_key] = spaces.Box(
                low=np.iinfo(np.uint8).min,
                high=np.iinfo(np.uint8).max,
                shape=(nrows, ncols),
                dtype=np.uint8,
            )

        for key in self.keys_to_delete:
            if key in observation_space.spaces:
                del observation_space.spaces[key]

        return observation_space

    def forward(self, observations: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.setup_mapping_module(observations)
        observations = self.update_maps_from_observations(observations)
        observations = self.visualize_maps(observations)
        observations = self.delete_extra_information(observations)
        return observations

    def setup_mapping_module(self, observations: Dict[str, Tensor]):
        raise NotImplementedError

    def update_maps_from_observations(self, observations):
        (
            episodes_info,
            input_observations,
            robot_current_state,
        ) = setup_inputs_from_obs_dict(observations)

        occupancy_semantic_map = self.mapping_module(
            episodes_info,
            input_observations,
            robot_current_state,
        )
        observations["occupancy_map"] = occupancy_semantic_map.occupancy
        observations["semantic_map"] = occupancy_semantic_map.semantic
        return observations

    def visualize_maps(self, observations):
        if self.visualize:
            observations["occupancy_map_viz"] = visualize_ego_occupancy_map(
                observations["occupancy_map"]
            )
            observations["semantic_map_viz"] = visualize_ego_semantic_map(
                observations["semantic_map"]
            )
        return observations

    def delete_extra_information(self, observations):
        for key in self.keys_to_delete:
            if key in observations:
                del observations[key]
        return observations

    @classmethod
    def from_config(cls, config: Config, visualize=False):
        camera_parameters = extract_camera_parameters(
            depth_sensor_params=config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR,
            map_sensor_params=config.RL.POLICY.OBS_TRANSFORMS.EGOCENTRIC_MAPPER,
        )
        egocentric_map_parameters = extract_egocentric_map_parameters(
            map_sensor_params=config.RL.POLICY.OBS_TRANSFORMS.EGOCENTRIC_MAPPER,
        )
        return cls(
            camera_parameters=camera_parameters,
            map_dimensions=egocentric_map_parameters,
            visualize=(len(config.VIDEO_OPTION) > 0) or visualize,
        )


@baseline_registry.register_obs_transformer()
class GTSemanticsIterativeMapper(Mapper):
    def setup_mapping_module(self, observations: Dict[str, Tensor]):
        if self.mapping_module is None:
            self.mapping_module = create_gt_semantics_iterative_mapper(
                device=observations["depth"].device,
                camera_parameters=self.camera_parameters,
                map_dimensions=self.map_dimensions,
            )


@baseline_registry.register_obs_transformer()
class PredictedSemanticsIterativeMapper(Mapper):
    def setup_mapping_module(self, observations: Dict[str, Tensor]):
        if self.mapping_module is None:
            self.mapping_module = create_predicted_semantics_iterative_mapper(
                device=observations["depth"].device,
                camera_parameters=self.camera_parameters,
                map_dimensions=self.map_dimensions,
            )


@baseline_registry.register_obs_transformer()
class GTSemanticsKnownMapper(Mapper):
    def setup_mapping_module(self, observations: Dict[str, Tensor]):
        if self.mapping_module is None:
            self.mapping_module = create_gt_semantics_known_mapper(
                device=observations["depth"].device,
                map_dimensions=self.map_dimensions,
            )


@baseline_registry.register_obs_transformer()
class PredictedSemanticsKnownMapper(Mapper):
    def setup_mapping_module(self, observations: Dict[str, Tensor]):
        if self.mapping_module is None:
            self.mapping_module = create_predicted_semantics_known_mapper(
                device=observations["depth"].device,
                map_dimensions=self.map_dimensions,
            )
