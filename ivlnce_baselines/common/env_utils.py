import random
import signal
from multiprocessing.connection import Connection
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import gym
import habitat
from habitat import Config, Env, RLEnv, VectorEnv, make_dataset
from habitat.core.dataset import ALL_SCENES_MASK
from habitat.core.logging import logger
from habitat.core.vector_env import (
    CALL_COMMAND,
    CLOSE_COMMAND,
    COUNT_EPISODES_COMMAND,
    RENDER_COMMAND,
    RESET_COMMAND,
    STEP_COMMAND,
)
from habitat.utils import profiling_wrapper
from habitat_baselines.utils.env_utils import make_env_fn


def construct_envs(
    config: Config,
    env_class: Type[Union[Env, RLEnv]],
    workers_ignore_signals: bool = False,
    auto_reset_done: bool = True,
    episodes_allowed: Optional[List[str]] = None,
) -> VectorEnv:
    """Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
    :param auto_reset_done: Whether or not to automatically reset the env on done

    :return: VectorEnv object created according to specification.
    """

    num_envs_per_gpu = config.NUM_ENVIRONMENTS
    if isinstance(config.SIMULATOR_GPU_IDS, list):
        gpus = config.SIMULATOR_GPU_IDS
    else:
        gpus = [config.SIMULATOR_GPU_IDS]
    num_gpus = len(gpus)
    num_envs = num_gpus * num_envs_per_gpu

    if episodes_allowed is not None:
        config.defrost()
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = episodes_allowed
        config.freeze()

    configs = []
    env_classes = [env_class for _ in range(num_envs)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if ALL_SCENES_MASK in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_envs > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multi-process logic relies on being able"
                " to split scenes uniquely between processes"
            )

        if len(scenes) < num_envs and len(scenes) != 1:
            raise RuntimeError(
                "reduce the number of GPUs or envs as there"
                " aren't enough number of scenes"
            )

        random.shuffle(scenes)

    if len(scenes) == 1:
        scene_splits = [[scenes[0]] for _ in range(num_envs)]
    else:
        scene_splits = [[] for _ in range(num_envs)]
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)

        assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_gpus):
        for j in range(num_envs_per_gpu):
            proc_config = config.clone()
            proc_config.defrost()
            proc_id = (i * num_envs_per_gpu) + j

            task_config = proc_config.TASK_CONFIG
            task_config.SEED += proc_id
            if len(scenes) > 0:
                task_config.DATASET.CONTENT_SCENES = scene_splits[proc_id]

            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpus[i]

            proc_config.freeze()
            configs.append(proc_config)

    envs = ExtendedVectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes)),
        auto_reset_done=auto_reset_done,
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs


def construct_envs_auto_reset_false(
    config: Config, env_class: Type[Union[Env, RLEnv]]
) -> VectorEnv:
    return construct_envs(config, env_class, auto_reset_done=False)


class ExtendedVectorEnv(VectorEnv):
    """Changes from VectorEnv:
    if the environment is VLNCEIterativeEnv,
        1. return observations and tour_done from .reset()
        2. return observations, reward, agent_done, sim_done,
            tour_done, produce_action, info from .step()
    """

    @staticmethod
    @profiling_wrapper.RangeContext("_worker_env")
    def _worker_env(
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        env_fn: Callable,
        env_fn_args: Tuple[Any],
        auto_reset_done: bool,
        mask_signals: bool = False,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
    ) -> None:
        r"""process worker for creating and interacting with the environment."""
        if mask_signals:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)

            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
            signal.signal(signal.SIGUSR2, signal.SIG_IGN)

        env = env_fn(*env_fn_args)
        if parent_pipe is not None:
            parent_pipe.close()
        try:
            command, data = connection_read_fn()
            while command != CLOSE_COMMAND:
                if command == STEP_COMMAND:
                    # different step methods for different Envs
                    if getattr(env, "is_iterative", False):
                        (
                            observations,
                            reward,
                            agent_done,
                            sim_done,
                            tour_done,
                            produce_action,
                            info,
                        ) = env.step(**data)
                        if auto_reset_done and sim_done:
                            (
                                observations,
                                tour_done,
                                produce_action,
                            ) = env.reset()
                        with profiling_wrapper.RangeContext(
                            "worker write after step"
                        ):
                            connection_write_fn(
                                (
                                    observations,
                                    reward,
                                    agent_done,
                                    sim_done,
                                    tour_done,
                                    produce_action,
                                    info,
                                )
                            )
                    elif isinstance(env, (habitat.RLEnv, gym.Env)):
                        # habitat.RLEnv
                        observations, reward, done, info = env.step(**data)
                        if auto_reset_done and done:
                            observations = env.reset()
                        with profiling_wrapper.RangeContext(
                            "worker write after step"
                        ):
                            connection_write_fn(
                                (observations, reward, done, info)
                            )
                    elif isinstance(env, habitat.Env):  # type: ignore
                        # habitat.Env
                        observations = env.step(**data)
                        if auto_reset_done and env.episode_over:
                            (
                                observations,
                                tour_done,
                                produce_action,
                            ) = env.reset()
                        connection_write_fn(
                            (observations, tour_done, produce_action)
                        )
                    else:
                        raise NotImplementedError

                elif command == RESET_COMMAND:
                    if getattr(env, "is_iterative", False):
                        (
                            observations,
                            tour_done,
                            produce_action,
                        ) = env.reset()
                        connection_write_fn(
                            (observations, tour_done, produce_action)
                        )
                    else:
                        observations = env.reset()
                        connection_write_fn(observations)

                elif command == RENDER_COMMAND:
                    connection_write_fn(env.render(*data[0], **data[1]))

                elif command == CALL_COMMAND:
                    function_name, function_args = data
                    if function_args is None:
                        function_args = {}

                    result_or_fn = getattr(env, function_name)

                    if len(function_args) > 0 or callable(result_or_fn):
                        result = result_or_fn(**function_args)
                    else:
                        result = result_or_fn

                    connection_write_fn(result)

                elif command == COUNT_EPISODES_COMMAND:
                    connection_write_fn(len(env.episodes))

                else:
                    raise NotImplementedError(f"Unknown command {command}")

                with profiling_wrapper.RangeContext("worker wait for command"):
                    command, data = connection_read_fn()

        except KeyboardInterrupt:
            logger.info("Worker KeyboardInterrupt")
        finally:
            if child_pipe is not None:
                child_pipe.close()
            env.close()


class ThreadedExtendedVectorEnv(ExtendedVectorEnv, habitat.ThreadedVectorEnv):
    pass
