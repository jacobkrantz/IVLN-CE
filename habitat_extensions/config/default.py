from typing import List, Optional, Union

from habitat.config.default import Config as CN
from habitat.config.default import get_config

_C = get_config()
_C.defrost()

# ----------------------------------------------------------------------------
# TOUR-BASED EPISODE ITERATOR
# ----------------------------------------------------------------------------
# whether or not to shuffle the tours order
_C.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE_TOURS = True
# whether or not to shuffle the episode order within tours
_C.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE_EPISODES = True
# if True, episodes in a tour are given a fixed order
_C.ENVIRONMENT.ITERATOR_OPTIONS.specify_episode_order = False
# ----------------------------------------------------------------------------
# ITERATIVE ENVIRONMENT
# ----------------------------------------------------------------------------
# performs an iterative evaluation with both episode and scene resets
_C.ENVIRONMENT.ITERATIVE = CN()
_C.ENVIRONMENT.ITERATIVE.ENABLED = False
_C.ENVIRONMENT.ITERATIVE.ENV_NAME = "VLNCEIterativeEnv"
# The oracle navigates the agent to within a FORWARD_STEP_SIZE radius of the
# start location and within +/- TURN_ANGLE/2 degrees of the start rotation. If
# PRECISE_EPISODE_START is True, then the oracle finishes this navigation with
# a teleportation so the agent starts exactly at the start pose.
_C.ENVIRONMENT.ITERATIVE.PRECISE_EPISODE_START = False
# if True, the oracle calls STOP when it cannot navigate to the position. Else
# a shortest path follower error is thrown. If this is True and
# PRECISE_EPISODE_START is True, then the oracle teleports to the start on a
# failed navigation.
_C.ENVIRONMENT.ITERATIVE.ORACLE_STOP_ON_ERROR = False
# number of steps the oracle can take before we assume the oracle is taking
# infinite actions and raise an error. -1 is no limit.
_C.ENVIRONMENT.ITERATIVE.ORACLE_STEP_ERROR_LIMIT = -1
# if True, the agent is navigated to the goal location via oracle actions
# after the episode is done. Otherwise the agent is conveyed to the next start
# location from wherever it stopped.
_C.ENVIRONMENT.ITERATIVE.ORACLE_GOAL_PHASE = True
# If False, the ORACLE_GOAL phase and ORACLE_START phases are both inactive
# and the agent is teleported to the starting pose of the next episode.
_C.ENVIRONMENT.ITERATIVE.ORACLE_PHASES = True
# ----------------------------------------------------------------------------
# ITERATIVE DATASET
# ----------------------------------------------------------------------------
# tours must contain at least this many episodes
_C.DATASET.MIN_TOUR_SIZE = -1
# specifices inter-navigable episode sets for the entire dataset
_C.DATASET.TOURS_FILE = ""
# how many tours to sample (-1 is all)
_C.DATASET.NUM_TOURS_SAMPLE = -1
# maximum number of episodes per tour to sample (-1 is all)
_C.DATASET.EPISODES_PER_TOUR = -1
# ----------------------------------------------------------------------------
# GPS SENSOR
# ----------------------------------------------------------------------------
_C.TASK.GLOBAL_GPS_SENSOR = CN()
_C.TASK.GLOBAL_GPS_SENSOR.TYPE = "GlobalGPSSensor"
_C.TASK.GLOBAL_GPS_SENSOR.DIMENSIONALITY = 3
# ----------------------------------------------------------------------------
# SEMANTIC12 Sensor
# ----------------------------------------------------------------------------
_C.TASK.SEMANTIC12_SENSOR = CN()
_C.TASK.SEMANTIC12_SENSOR.TYPE = "Semantic12Sensor"
_C.TASK.SEMANTIC12_SENSOR.DIMENSIONALITY = 3
# ----------------------------------------------------------------------------
# WorldRobotPose
# ----------------------------------------------------------------------------
_C.TASK.WORLD_ROBOT_POSE_SENSOR = CN()
_C.TASK.WORLD_ROBOT_POSE_SENSOR.TYPE = "WorldRobotPoseSensor"
_C.TASK.WORLD_ROBOT_POSE_SENSOR.DIMENSIONALITY = 3
# ----------------------------------------------------------------------------
# # GT POINTCLOUD SENSOR
# ----------------------------------------------------------------------------
_C.TASK.ENV_NAME_SENSOR = CN()
_C.TASK.ENV_NAME_SENSOR.TYPE = "EnvNameSensor"
_C.TASK.ENV_NAME_SENSOR.DIMENSIONALITY = 3
# ----------------------------------------------------------------------------
# WorldRobotOrientation
# ----------------------------------------------------------------------------
_C.TASK.WORLD_ROBOT_ORIENTATION_SENSOR = CN()
_C.TASK.WORLD_ROBOT_ORIENTATION_SENSOR.TYPE = "WorldRobotOrientationSensor"
_C.TASK.WORLD_ROBOT_ORIENTATION_SENSOR.DIMENSIONALITY = 3
# ----------------------------------------------------------------------------
# RXR INSTRUCTION SENSOR
# ----------------------------------------------------------------------------
_C.TASK.RXR_INSTRUCTION_SENSOR = CN()
_C.TASK.RXR_INSTRUCTION_SENSOR.TYPE = "RxRInstructionSensor"
_C.TASK.RXR_INSTRUCTION_SENSOR.features_path = "data/datasets/RxR_VLNCE_v0/text_features/rxr_{split}/{id:06}_{lang}_text_features.npz"
_C.TASK.INSTRUCTION_SENSOR_UUID = "instruction"
# ----------------------------------------------------------------------------
# SHORTEST PATH SENSOR
# ----------------------------------------------------------------------------
_C.TASK.SHORTEST_PATH_SENSOR = CN()
_C.TASK.SHORTEST_PATH_SENSOR.TYPE = "ShortestPathSensor"
# all goals can be navigated to within 0.5m.
_C.TASK.SHORTEST_PATH_SENSOR.GOAL_RADIUS = 0.5
# ----------------------------------------------------------------------------
# VLN ORACLE PROGRESS SENSOR
# ----------------------------------------------------------------------------
_C.TASK.VLN_ORACLE_PROGRESS_SENSOR = CN()
_C.TASK.VLN_ORACLE_PROGRESS_SENSOR.TYPE = "VLNOracleProgressSensor"
# ----------------------------------------------------------------------------
# NDTW MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.NDTW = CN()
_C.TASK.NDTW.TYPE = "NDTW"
_C.TASK.NDTW.SPLIT = "val_seen"
_C.TASK.NDTW.FDTW = True  # False: DTW
_C.TASK.NDTW.GT_PATH = (
    "data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz"
)
_C.TASK.NDTW.SUCCESS_DISTANCE = 3.0
# ----------------------------------------------------------------------------
# SDTW MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.SDTW = CN()
_C.TASK.SDTW.TYPE = "SDTW"
# ----------------------------------------------------------------------------
# PATH_LENGTH MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.PATH_LENGTH = CN()
_C.TASK.PATH_LENGTH.TYPE = "PathLength"
# ----------------------------------------------------------------------------
# ORACLE_NAVIGATION_ERROR MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.ORACLE_NAVIGATION_ERROR = CN()
_C.TASK.ORACLE_NAVIGATION_ERROR.TYPE = "OracleNavigationError"
# ----------------------------------------------------------------------------
# ORACLE_SUCCESS MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.ORACLE_SUCCESS = CN()
_C.TASK.ORACLE_SUCCESS.TYPE = "OracleSuccess"
_C.TASK.ORACLE_SUCCESS.SUCCESS_DISTANCE = 3.0
# ----------------------------------------------------------------------------
# ORACLE_SPL MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.ORACLE_SPL = CN()
_C.TASK.ORACLE_SPL.TYPE = "OracleSPL"
# ----------------------------------------------------------------------------
# STEPS_TAKEN MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.STEPS_TAKEN = CN()
_C.TASK.STEPS_TAKEN.TYPE = "StepsTaken"
# ----------------------------------------------------------------------------
# TOP_DOWN_MAP_VLNCE MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.TOP_DOWN_MAP_VLNCE = CN()
_C.TASK.TOP_DOWN_MAP_VLNCE.TYPE = "TopDownMapVLNCE"
_C.TASK.TOP_DOWN_MAP_VLNCE.MAX_EPISODE_STEPS = _C.ENVIRONMENT.MAX_EPISODE_STEPS
_C.TASK.TOP_DOWN_MAP_VLNCE.MAP_RESOLUTION = 1024
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_SOURCE_AND_TARGET = True
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_BORDER = True
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_SHORTEST_PATH = True
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_REFERENCE_PATH = True
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_FIXED_WAYPOINTS = True
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_MP3D_AGENT_PATH = True
_C.TASK.TOP_DOWN_MAP_VLNCE.GRAPHS_FILE = "data/connectivity_graphs.pkl"
_C.TASK.TOP_DOWN_MAP_VLNCE.FOG_OF_WAR = CN()
_C.TASK.TOP_DOWN_MAP_VLNCE.FOG_OF_WAR.DRAW = True
_C.TASK.TOP_DOWN_MAP_VLNCE.FOG_OF_WAR.FOV = 90
_C.TASK.TOP_DOWN_MAP_VLNCE.FOG_OF_WAR.VISIBILITY_DIST = 5.0
# ----------------------------------------------------------------------------
# DATASET EXTENSIONS
# ----------------------------------------------------------------------------
_C.DATASET.ROLES = ["guide"]  # options: "*", "guide", "follower"
# language options by region: "*", "te-IN", "hi-IN", "en-US", "en-IN"
_C.DATASET.LANGUAGES = ["*"]
# a list of episode IDs to allow in dataset creation.
_C.DATASET.EPISODES_ALLOWED = ["*"]


def get_extended_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    """Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.
    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()

    # habitat's episode iterator does not support new iterator options keys
    if not config.ENVIRONMENT.ITERATIVE.ENABLED:
        config.ENVIRONMENT.ITERATOR_OPTIONS = CN(
            init_dict={
                k: v
                for k, v in dict(config.ENVIRONMENT.ITERATOR_OPTIONS).items()
                if k not in ["SHUFFLE_EPISODES", "SHUFFLE_TOURS"]
            }
        )

    if config_paths:
        if isinstance(config_paths, str):
            config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    # set split-dependent metrics to the current split.
    config.TASK.NDTW.SPLIT = config.DATASET.SPLIT

    config.freeze()
    return config
