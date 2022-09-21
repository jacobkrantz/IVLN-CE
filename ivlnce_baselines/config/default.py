from typing import List, Optional, Union

import habitat_baselines.config.default
from habitat.config.default import CONFIG_FILE_SEPARATOR
from habitat.config.default import Config as CN

from habitat_extensions.config.default import (
    get_extended_config as get_task_config,
)

# ----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# ----------------------------------------------------------------------------
_C = CN()
_C.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/vlnce_task.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "dagger"
_C.ENV_NAME = "VLNCEDaggerEnv"
_C.SIMULATOR_GPU_IDS = [0]
_C.VIDEO_OPTION = []  # options: "disk", "tensorboard"
_C.VIDEO_DIR = "data/videos/debug"
_C.TENSORBOARD_DIR = "data/tensorboard_dirs/debug"
_C.RESULTS_DIR = "data/checkpoints/pretrained/evals"

# ----------------------------------------------------------------------------
# EVAL CONFIG
# ----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.SPLIT = "val_seen"
_C.EVAL.EPISODE_COUNT = -1
_C.EVAL.LANGUAGES = ["en-US", "en-IN"]
_C.EVAL.SAMPLE = False
_C.EVAL.USE_CKPT_CONFIG = False
_C.EVAL.SAVE_RESULTS = True
_C.EVAL.ITERATIVE_MAP_RESET = "iterative"  # episodic or iterative
_C.EVAL.ITERATIVE_GT_PATHS = "data/gt_ndtw.json"

# ----------------------------------------------------------------------------
# IMITATION LEARNING CONFIG
# ----------------------------------------------------------------------------
_C.IL = CN()
_C.IL.lr = 2.5e-4
_C.IL.batch_size = 5
_C.IL.epochs = 4
# if true, uses class-based inflection weighting
_C.IL.use_iw = True
# inflection coefficient for RxR training set GT trajectories (guide): 1.9
# inflection coefficient for R2R training set GT trajectories: 3.2
_C.IL.inflection_weight_coef = 3.2
# load an already trained model for fine tuning
_C.IL.load_from_ckpt = False
_C.IL.ckpt_to_load = "data/checkpoints/ckpt.0.pth"
# if True, loads the optimizer state, epoch, and step_id from the ckpt dict.
_C.IL.is_requeue = False

# ----------------------------------------------------------------------------
# IL: DAGGER CONFIG
# ----------------------------------------------------------------------------
_C.IL.DAGGER = CN()
# dataset aggregation rounds (1 for teacher forcing)
_C.IL.DAGGER.iterations = 10
# episodes collected per iteration (size of dataset for teacher forcing)
_C.IL.DAGGER.update_size = 5000
# probability of taking the expert action (1.0 for teacher forcing)
_C.IL.DAGGER.p = 0.75
_C.IL.DAGGER.expert_policy_sensor = "SHORTEST_PATH_SENSOR"
_C.IL.DAGGER.expert_policy_sensor_uuid = "shortest_path_sensor"
_C.IL.DAGGER.lmdb_map_size = 1.0e13
# if True, saves data to disk in fp16 and converts back to fp32 when loading.
_C.IL.DAGGER.lmdb_fp16 = False
# How often to commit the writes to the DB, less commits is
# better, but everything must be in memory until a commit happens.
_C.IL.DAGGER.lmdb_commit_frequency = 500
# If True, load precomputed features directly from lmdb_features_dir.
_C.IL.DAGGER.preload_lmdb_features = False
_C.IL.DAGGER.lmdb_features_dir = (
    "data/trajectories_dirs/debug/trajectories.lmdb"
)
_C.IL.DAGGER.drop_existing_lmdb_features = True

# ----------------------------------------------------------------------------
# POLICY CONFIG
# ----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.POLICY = CN()
_C.RL.POLICY.OBS_TRANSFORMS = CN()
_C.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = []
_C.RL.POLICY.OBS_TRANSFORMS.EGOCENTRIC_MAPPER = CN()
_C.RL.POLICY.OBS_TRANSFORMS.EGOCENTRIC_MAPPER.resolution_meters = 0.1
_C.RL.POLICY.OBS_TRANSFORMS.EGOCENTRIC_MAPPER.height_clip = 0.1
_C.RL.POLICY.OBS_TRANSFORMS.EGOCENTRIC_MAPPER.height_meters = 6.4
_C.RL.POLICY.OBS_TRANSFORMS.EGOCENTRIC_MAPPER.width_meters = 6.4

# ----------------------------------------------------------------------------
# MODELING CONFIG
# ----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.policy_name = "CMAPolicy"

_C.MODEL.ablate_depth = False
_C.MODEL.ablate_rgb = False
_C.MODEL.ablate_map = False
_C.MODEL.ablate_instruction = False
_C.MODEL.tour_memory = False
# keeps the existing episodic memory as-is. adds a cross-episode memory to the model.
_C.MODEL.tour_memory_variant = False
# uses tour memory for action distribution. MODEL.tour_memory_variant must be True.
_C.MODEL.memory_at_end = False
# force the model to be trained in an unrolled RNN fashion (about 10x slower).
_C.MODEL.train_unrolled = False
_C.MODEL.disable_tour_memory = False

_C.MODEL.INSTRUCTION_ENCODER = CN()
_C.MODEL.INSTRUCTION_ENCODER.sensor_uuid = "instruction"
_C.MODEL.INSTRUCTION_ENCODER.vocab_size = 2504
_C.MODEL.INSTRUCTION_ENCODER.use_pretrained_embeddings = True
_C.MODEL.INSTRUCTION_ENCODER.embedding_file = (
    "data/datasets/R2R_VLNCE_v1-3_preprocessed/embeddings.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.dataset_vocab = (
    "data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.fine_tune_embeddings = False
_C.MODEL.INSTRUCTION_ENCODER.embedding_size = 50
_C.MODEL.INSTRUCTION_ENCODER.hidden_size = 128
_C.MODEL.INSTRUCTION_ENCODER.rnn_type = "LSTM"
_C.MODEL.INSTRUCTION_ENCODER.final_state_only = True
_C.MODEL.INSTRUCTION_ENCODER.bidirectional = True

_C.MODEL.RGB_ENCODER = CN()
_C.MODEL.RGB_ENCODER.cnn_type = "TorchVisionResNet50"
_C.MODEL.RGB_ENCODER.output_size = 256
_C.MODEL.RGB_ENCODER.trainable = False

_C.MODEL.DEPTH_ENCODER = CN()
_C.MODEL.DEPTH_ENCODER.cnn_type = "VlnResnetDepthEncoder"
_C.MODEL.DEPTH_ENCODER.output_size = 128
_C.MODEL.DEPTH_ENCODER.backbone = "resnet50"
_C.MODEL.DEPTH_ENCODER.ddppo_checkpoint = (
    "data/ddppo-models/gibson-2plus-resnet50.pth"
)
_C.MODEL.DEPTH_ENCODER.trainable = False

_C.MODEL.SEMANTIC_MAP_ENCODER = CN()
_C.MODEL.SEMANTIC_MAP_ENCODER.classname = "SemanticMapEncoder"
_C.MODEL.SEMANTIC_MAP_ENCODER.num_semantic_classes = 13
_C.MODEL.SEMANTIC_MAP_ENCODER.output_size = 256
_C.MODEL.SEMANTIC_MAP_ENCODER.channels = 32
_C.MODEL.SEMANTIC_MAP_ENCODER.last_ch_mult = 4
_C.MODEL.SEMANTIC_MAP_ENCODER.trainable = True
_C.MODEL.SEMANTIC_MAP_ENCODER.from_pretrained = False
_C.MODEL.SEMANTIC_MAP_ENCODER.checkpoint = ""
_C.MODEL.SEMANTIC_MAP_ENCODER.custom_lr = False
_C.MODEL.SEMANTIC_MAP_ENCODER.lr = 2.5e-6  # 100x smaller than standard

_C.MODEL.STATE_ENCODER = CN()
_C.MODEL.STATE_ENCODER.hidden_size = 512
_C.MODEL.STATE_ENCODER.rnn_type = "GRU"

_C.MODEL.PROGRESS_MONITOR = CN()
_C.MODEL.PROGRESS_MONITOR.use = False
_C.MODEL.PROGRESS_MONITOR.alpha = 1.0  # loss multiplier


def purge_keys(config: CN, keys: List[str]) -> None:
    for k in keys:
        del config[k]
        config.register_deprecated_key(k)


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    """Create a unified config with default values. Initialized from the
    habitat_baselines default config. Overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = CN()
    config.merge_from_other_cfg(habitat_baselines.config.default._C)
    purge_keys(config, ["SIMULATOR_GPU_ID", "TEST_EPISODE_COUNT"])
    config.merge_from_other_cfg(_C.clone())

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        prev_task_config = ""
        for config_path in config_paths:
            config.merge_from_file(config_path)
            if config.BASE_TASK_CONFIG_PATH != prev_task_config:
                config.TASK_CONFIG = get_task_config(
                    config.BASE_TASK_CONFIG_PATH
                )
                prev_task_config = config.BASE_TASK_CONFIG_PATH

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.freeze()
    return config
