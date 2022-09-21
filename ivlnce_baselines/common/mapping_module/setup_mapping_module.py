import numpy as np
from habitat import Config

from ivlnce_baselines.common.mapping_module.mapper import (
    CameraParameters,
    EpisodesInfo,
    MapDimensions,
    Observations,
    RobotCurrentState,
)


def calculate_vertical_fov_in_degrees(depth_sensor_params: Config) -> float:
    horizontal_fov_degrees = depth_sensor_params.HFOV
    vertical_fov_degrees = horizontal_fov_degrees * (
        depth_sensor_params.HEIGHT / depth_sensor_params.WIDTH
    )
    return vertical_fov_degrees


def calculate_verticial_fov_in_radians(depth_sensor_params: Config) -> float:
    vertical_fov_degrees = calculate_vertical_fov_in_degrees(
        depth_sensor_params
    )
    vertical_fov_radians = np.deg2rad(vertical_fov_degrees)
    return vertical_fov_radians


def extract_camera_parameters(
    depth_sensor_params, map_sensor_params
) -> CameraParameters:
    vertical_fov_radians = calculate_verticial_fov_in_radians(
        depth_sensor_params
    )
    return CameraParameters(
        vertical_fov_radians=vertical_fov_radians,
        features_spatial_dimensions=(
            depth_sensor_params.HEIGHT,
            depth_sensor_params.WIDTH,
        ),
        height_clip=map_sensor_params.height_clip,
    )


def extract_egocentric_map_parameters(
    map_sensor_params: Config,
) -> MapDimensions:
    egocentric_map_parameters = MapDimensions(
        height_meters=map_sensor_params.height_meters,
        width_meters=map_sensor_params.width_meters,
        resolution_meters=map_sensor_params.resolution_meters,
    )
    return egocentric_map_parameters


def channel_first_representation(x):
    if x is not None:
        x = x.permute(0, 3, 1, 2)
    return x


def setup_observations(observations_dict: dict) -> Observations:
    semantics = channel_first_representation(
        observations_dict.get("semantic12", None)
    )
    depth_normalized = channel_first_representation(
        observations_dict.get("depth", None)
    )
    rgb = channel_first_representation(observations_dict.get("rgb", None))
    return Observations(
        semantics=semantics, depth_normalized=depth_normalized, rgb=rgb
    )


def setup_inputs_from_obs_dict(observations_dict: dict):
    observations = setup_observations(observations_dict)

    episodes_info = EpisodesInfo(
        observations_dict["not_done_masks"],
        observations_dict["env_name"],
    )

    robot_current_state = RobotCurrentState(
        pose=observations_dict["world_robot_pose"],
        elevation=observations_dict["world_robot_orientation"][:, 0],
        heading=observations_dict["world_robot_orientation"][:, 1],
    )

    return episodes_info, observations, robot_current_state
