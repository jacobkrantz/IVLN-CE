import os
from functools import lru_cache
from typing import Any

import numpy as np
from gym import Space, spaces
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, Simulator
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from scipy.spatial.transform import Rotation as R

from habitat_extensions.task import VLNExtendedEpisode


@registry.register_sensor(name="GlobalGPSSensor")
class GlobalGPSSensor(Sensor):
    """Current agent location in global coordinate frame"""

    cls_uuid: str = "globalgps"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._dimensionality = config.DIMENSIONALITY
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float).min,
            high=np.finfo(np.float).max,
            shape=(self._dimensionality,),
            dtype=np.float,
        )

    def get_observation(self, *args: Any, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position
        if self._dimensionality == 2:
            agent_position = np.array([agent_position[0], agent_position[2]])
        return agent_position.astype(np.float32)


@registry.register_sensor
class VLNOracleProgressSensor(Sensor):
    """Relative progress towards goal"""

    cls_uuid: str = "progress"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ) -> None:
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float)

    def get_observation(self, *args: Any, episode, **kwargs: Any) -> float:
        distance_to_target = self._sim.geodesic_distance(
            self._sim.get_agent_state().position.tolist(),
            episode.goals[0].position,
        )

        # just in case the agent ends up somewhere it shouldn't
        if not np.isfinite(distance_to_target):
            return np.array([0.0])

        distance_from_start = episode.info["geodesic_distance"]
        return np.array(
            [(distance_from_start - distance_to_target) / distance_from_start]
        )


@registry.register_sensor
class ShortestPathSensor(Sensor):
    """Provides the next action to follow the shortest path to the goal."""

    cls_uuid: str = "shortest_path_sensor"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        super().__init__(config=config)
        self.follower = ShortestPathFollower(
            sim, config.GOAL_RADIUS, return_one_hot=False
        )

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float)

    def get_observation(self, *args: Any, episode, **kwargs: Any):
        best_action = self.follower.get_next_action(episode.goals[0].position)
        if best_action is None:
            best_action = HabitatSimActions.STOP
        return np.array([best_action])


@registry.register_sensor
class RxRInstructionSensor(Sensor):
    """Loads pre-computed intruction features from disk in the baseline RxR
    BERT file format.
    https://github.com/google-research-datasets/RxR/tree/7a6b87ba07959f5176aa336192a8c5dc85ca1b8e#downloading-bert-text-features
    """

    cls_uuid: str = "rxr_instruction"

    def __init__(self, *args: Any, config: Config, **kwargs: Any):
        self.features_path = config.features_path
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float).min,
            high=np.finfo(np.float).max,
            shape=(512, 768),
            dtype=np.float,
        )

    def get_observation(
        self, *args: Any, episode: VLNExtendedEpisode, **kwargs
    ):
        features = np.load(
            self.features_path.format(
                split=episode.instruction.split,
                id=int(episode.instruction.instruction_id),
                lang=episode.instruction.language.split("-")[0],
            )
        )
        feats = np.zeros((512, 768), dtype=np.float32)
        s = features["features"].shape
        feats[: s[0], : s[1]] = features["features"]
        return feats


@registry.register_sensor(name="WorldRobotPoseSensor")
class WorldRobotPoseSensor(Sensor):
    """The agents current location in the global coordinate frame
    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions
                to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """

    cls_uuid: str = "world_robot_pose"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._dimensionality,),
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, **kwargs: Any):
        return (
            self._sim.get_agent_state()
            .sensor_states["depth"]
            .position.astype(np.float32)
        )
        # return self._sim.get_agent_state().position.astype(np.float32)


@registry.register_sensor(name="WorldRobotOrientationSensor")
class WorldRobotOrienationSensor(Sensor):
    """The agents current location in the global coordinate frame
    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions
                to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """

    cls_uuid: str = "world_robot_orientation"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._dimensionality = getattr(config, "DIMENSIONALITY", 4)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._dimensionality,),
            dtype=np.float32,
        )

    @staticmethod
    def get_euler_angles(ori):
        orientation = np.array([ori.x, ori.y, ori.z, ori.w])
        r = R.from_quat(orientation)
        elevation, heading, _ = r.as_rotvec()
        elevation_heading = np.asarray([elevation, heading])
        return elevation_heading

    def get_observation(self, *args: Any, **kwargs: Any):
        quat = self._sim.get_agent_state().sensor_states["depth"].rotation
        eh = self.get_euler_angles(quat)
        return eh


@registry.register_sensor(name="Semantic12Sensor")
class Semantic12Sensor(Sensor):
    """ """

    cls_uuid: str = "semantic12"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._dimensionality = getattr(config, "DIMENSIONALITY", 4)
        super().__init__(config=config)

        self.use_fine = ["appliances", "furniture"]
        self.object_whitelist = [
            "shelving",
            "chest_of_drawers",
            "bed",
            "cushion",
            "fireplace",
            "sofa",
            "table",
            "chair",
            "cabinet",
            "plant",
            "counter",
            "sink",
        ]

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._dimensionality,),
            dtype=np.float32,
        )

    @lru_cache(maxsize=1)
    def get_objects_in_house(self, semantic_annotations):
        # semantic_annotations = self._sim.semantic_annotations()
        objects = {
            int(o.id.split("_")[-1]): o
            for o in semantic_annotations.objects
            if o is not None
        }
        return objects

    def render_semantic_12cat(self, buf):
        out = np.zeros_like(buf, dtype=np.uint8)  # class 0 -> void
        object_ids = np.unique(buf)
        for oid in object_ids:
            obj = self.all_objects[oid]
            object_name = obj.category.name(mapping="mpcat40")
            if object_name in self.use_fine:
                object_name = obj.category.name(mapping="raw")
            if object_name in self.object_whitelist:
                object_index = self.object_whitelist.index(object_name) + 1
                out[buf == oid] = object_index
        return out

    def get_observation(
        self,
        observations,
        *args: Any,
        **kwargs: Any,
    ):
        self.all_objects = self.get_objects_in_house(
            self._sim.semantic_annotations()
        )
        semantic12 = self.render_semantic_12cat(observations["semantic"])
        return np.expand_dims(semantic12, 2)


@registry.register_sensor(name="EnvNameSensor")
class GTPointcloudSensor(Sensor):
    cls_uuid: str = "env_name"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._dimensionality = getattr(config, "DIMENSIONALITY", 4)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._dimensionality,),
            dtype=np.float32,
        )

    @staticmethod
    @lru_cache()
    def parse_env_name(filename):
        return os.path.basename(filename).split(".")[0]

    def get_observation(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        return self.parse_env_name(self._sim._current_scene)
