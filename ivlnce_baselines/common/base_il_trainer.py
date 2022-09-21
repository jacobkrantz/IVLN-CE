import contextlib
import json
import numbers
import os
import time
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import tqdm
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job

from habitat_extensions.tour_ndtw import compute_tour_ndtw
from habitat_extensions.utils import generate_video, observations_to_image
from ivlnce_baselines.common.aux_losses import AuxLosses
from ivlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from ivlnce_baselines.common.mapping_module.visualize_semantic_map import (
    append_image_horizontally,
    append_image_vertically,
)
from ivlnce_baselines.common.utils import (
    add_batched_data_to_observations,
    batch_obs,
    extract_instruction_tokens,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401


class BaseVLNCETrainer(BaseILTrainer):
    """A base trainer for VLN-CE imitation learning."""

    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.obs_transforms = []
        self.start_epoch = 0
        self.step_id = 0

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.to(self.device)

        if self.config.MODEL.SEMANTIC_MAP_ENCODER.custom_lr:
            # optional custom learning rate for the semantic map encoder
            sem_params = []
            regular_params = []
            for name, param in self.policy.named_parameters():
                if name.startswith("net.map_encoder"):
                    sem_params.append(param)
                else:
                    regular_params.append(param)
            self.optimizer = torch.optim.Adam(
                [{"params": sem_params}, {"params": regular_params}],
                lr=self.config.IL.lr,
            )
            sem_lr = self.config.MODEL.SEMANTIC_MAP_ENCODER.lr
            self.optimizer.param_groups[0]["lr"] = sem_lr
        else:
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=self.config.IL.lr
            )

        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            self.policy.load_state_dict(ckpt_dict["state_dict"])
            if config.IL.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                self.start_epoch = ckpt_dict["epoch"] + 1
                self.step_id = ckpt_dict["step_id"]
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")

    def _get_spaces(
        self, config: Config, envs: Optional[Any] = None
    ) -> Tuple[Space]:
        """Gets both the observation space and action space.

        Args:
            config (Config): The config specifies the observation transforms.
            envs (Any, optional): An existing Environment. If None, an
                environment is created using the config.

        Returns:
            observation space, action space
        """
        if envs is not None:
            observation_space = envs.observation_spaces[0]
            action_space = envs.action_spaces[0]

        else:
            env = get_env_class(self.config.ENV_NAME)(config=config)
            observation_space = env.observation_space
            action_space = env.action_space

        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        return observation_space, action_space

    def save_checkpoint(
        self,
        file_name: str,
        dagger_it: int = 0,
        epoch: int = 0,
        step_id: int = 0,
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        torch.save(
            {
                "state_dict": self.policy.state_dict(),
                "config": self.config,
                "optim_state": self.optimizer.state_dict(),
                "dagger_it": dagger_it,
                "epoch": epoch,
                "step_id": step_id,
            },
            os.path.join(self.config.CHECKPOINT_FOLDER, file_name),
        )

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    def _update_agent(
        self,
        observations,
        prev_actions,
        not_done_masks,
        corrected_actions,
        weights,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
    ):
        T, N = corrected_actions.size()

        recurrent_hidden_states = torch.zeros(
            N,
            self.policy.net.num_recurrent_layers,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )

        AuxLosses.clear()

        distribution, _ = self.policy.build_distribution(
            observations, recurrent_hidden_states, prev_actions, not_done_masks
        )

        logits = distribution.logits
        logits = logits.view(T, N, -1)

        action_loss = F.cross_entropy(
            logits.permute(0, 2, 1), corrected_actions, reduction="none"
        )
        action_loss = ((weights * action_loss).sum(0) / weights.sum(0)).mean()

        aux_mask = (weights > 0).view(-1)
        aux_loss = AuxLosses.reduce(aux_mask)

        loss = action_loss + aux_loss
        loss = loss / loss_accumulation_scalar
        loss.backward()

        if step_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.item()
        return loss.item(), action_loss.item(), aux_loss

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        recurrent_hidden_states,
        not_done_masks,
        prev_actions,
        batch,
        rgb_frames=None,
    ):
        # pausing envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            recurrent_hidden_states = recurrent_hidden_states[state_index]
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            recurrent_hidden_states,
            not_done_masks,
            prev_actions,
            batch,
            rgb_frames,
        )

    def _pause_iterative_envs(
        self,
        envs_to_pause,
        envs,
        recurrent_hidden_states,
        agent_episode_not_done_masks,
        sim_episode_not_done_masks,
        tour_not_done_masks,
        action_masks,
        prev_actions,
        batch,
        rgb_frames=None,
    ):
        # pausing envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            recurrent_hidden_states = recurrent_hidden_states[state_index]
            agent_episode_not_done_masks = agent_episode_not_done_masks[
                state_index
            ]
            sim_episode_not_done_masks = sim_episode_not_done_masks[
                state_index
            ]
            tour_not_done_masks = tour_not_done_masks[state_index]
            action_masks = action_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

            # delete any batch-idx-specific memory in the policy.
            del_idxs_fn = getattr(self.policy.net, "delete_batch_idx", None)
            if callable(del_idxs_fn):
                del_idxs_fn(envs_to_pause)

        return (
            envs,
            recurrent_hidden_states,
            agent_episode_not_done_masks,
            sim_episode_not_done_masks,
            tour_not_done_masks,
            action_masks,
            prev_actions,
            batch,
            rgb_frames,
        )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
        metrics=None,
    ) -> None:
        """Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint
        Returns:
            None
        """
        if metrics is None:
            metrics = "distance_to_goal success spl ndtw path_length oracle_success steps_taken".split()

        # sometimes the index does not align with the actual ckpt number
        with contextlib.suppress(Exception):
            checkpoint_index = int(checkpoint_path.split(".")[-2])

        start_from = getattr(self.config.EVAL, "START_FROM", 0)
        if checkpoint_index < start_from:
            logger.info(f"skipping ckpt: starting from {start_from}.")
            return

        logger.info(f"checkpoint_path: {checkpoint_path}")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(
                self.load_checkpoint(checkpoint_path, map_location="cpu")[
                    "config"
                ]
            )
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        config.TASK_CONFIG.TASK.NDTW.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE_TOURS = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE_EPISODES = (
            False
        )
        config.IL.ckpt_to_load = checkpoint_path
        config.use_pbar = not is_slurm_batch_job()

        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        config.freeze()

        if config.TASK_CONFIG.ENVIRONMENT.ITERATIVE.ENABLED:
            self._eval_checkpoint_iterative(
                config,
                writer=writer,
                checkpoint_index=checkpoint_index,
            )
            return

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname):
                logger.info("skipping -- evaluation exists.")
                return

        envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )
        observation_space, action_space = self._get_spaces(config, envs=envs)

        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.eval()

        rnn_states = torch.zeros(
            envs.num_envs,
            self.policy.net.num_recurrent_layers,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            envs.num_envs, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        observations = envs.reset()
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        observations = add_batched_data_to_observations(
            observations, not_done_masks, "not_done_masks"
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        stats_episodes = {}

        rgb_frames = [[] for _ in range(envs.num_envs)]
        episodes_to_eval = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            episodes_to_eval = min(config.EVAL.EPISODE_COUNT, episodes_to_eval)

        pbar = tqdm.tqdm(total=episodes_to_eval) if config.use_pbar else None
        log_str = (
            f"[Ckpt: {checkpoint_index}]"
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        while envs.num_envs > 0 and len(stats_episodes) < episodes_to_eval:
            current_episodes = envs.current_episodes()

            with torch.no_grad():
                actions, rnn_states = self.policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=not config.EVAL.SAMPLE,
                )
                prev_actions.copy_(actions)

            outputs = envs.step([a[0].item() for a in actions])
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            not_done_masks = torch.tensor(
                [[0] if done else [1] for done in dones],
                dtype=torch.uint8,
                device=self.device,
            )
            observations = add_batched_data_to_observations(
                observations, not_done_masks, "not_done_masks"
            )

            # reset envs and observations if necessary
            for i in range(envs.num_envs):
                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(
                        frame, current_episodes[i].instruction.instruction_text
                    )

                    if (
                        "occupancy_map_viz" in batch
                        and "semantic_map_viz" in batch
                    ):
                        map_frame = [
                            batch["occupancy_map_viz"][i],
                            batch["semantic_map_viz"][i],
                        ]
                        map_frame = append_image_horizontally(map_frame)
                        frame = append_image_vertically([frame, map_frame])

                    rgb_frames[i].append(frame)

                if not dones[i]:
                    continue

                stats_episodes[current_episodes[i].episode_id] = {
                    k: infos[i][k] for k in metrics
                }
                observations[i] = envs.reset_at(i)[0]
                prev_actions[i] = torch.zeros(1, dtype=torch.long)

                if config.use_pbar:
                    pbar.update()
                else:
                    logger.info(
                        log_str.format(
                            evaluated=len(stats_episodes),
                            total=episodes_to_eval,
                            time=round(time.time() - start_time),
                        )
                    )

                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=current_episodes[i].episode_id,
                        checkpoint_idx=checkpoint_index,
                        metrics={
                            "spl": stats_episodes[
                                current_episodes[i].episode_id
                            ]["spl"]
                        },
                        tb_writer=writer,
                    )
                    rgb_frames[i] = []

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
            observations = add_batched_data_to_observations(
                observations, not_done_masks, "not_done_masks"
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (
                envs,
                rnn_states,
                not_done_masks,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                envs,
                rnn_states,
                not_done_masks,
                prev_actions,
                batch,
                rgb_frames,
            )

        envs.close()
        if config.use_pbar:
            pbar.close()

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        split = config.TASK_CONFIG.DATASET.SPLIT
        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{split}.json",
            )
            with open(fname, "w") as f:
                json.dump(aggregated_stats, f, indent=4)

        logger.info(f"Episodes evaluated: {num_episodes}")
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.6f}")
            writer.add_scalar(f"eval_{split}_{k}", v, checkpoint_index + 1)

    def _eval_checkpoint_iterative(
        self,
        config,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        """Evaluates a single checkpoint iteratively.
        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint
        Returns:
            None
        """
        if "Iterative" not in config.ENV_NAME:
            config.defrost()
            config.ENV_NAME = config.TASK_CONFIG.ENVIRONMENT.ITERATIVE.ENV_NAME
            config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"iterative_stats_ckpt_{checkpoint_index}_{config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname):
                logger.info("skipping -- evaluation exists.")
                return

        assert self.config.EVAL.ITERATIVE_MAP_RESET in [
            "episodic",
            "iterative",
        ], "config.EVAL.ITERATIVE_MAP_RESET not valid"

        envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )

        observation_space, action_space = self._get_spaces(config, envs=envs)

        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.eval()

        rnn_states = torch.zeros(
            envs.num_envs,
            self.policy.net.num_recurrent_layers,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            envs.num_envs, 1, device=self.device, dtype=torch.long
        )
        agent_episode_not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )
        sim_episode_not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )
        tour_not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )
        action_masks = torch.ones(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        observations, _, _ = [list(x) for x in zip(*envs.reset())]
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )

        reset_masks = (
            agent_episode_not_done_masks
            if self.config.EVAL.ITERATIVE_MAP_RESET == "episodic"
            else tour_not_done_masks
        )
        observations = add_batched_data_to_observations(
            observations, reset_masks, "not_done_masks"
        )

        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        stats_tours = defaultdict(dict)  # tourID -> episodeID -> stats
        dtw_data = defaultdict(list)  # collecting dtw positions for each tour.
        episodes_evaluated = 0

        rgb_frames = [[] for _ in range(envs.num_envs)]
        episodes_to_eval = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            episodes_to_eval = min(config.EVAL.EPISODE_COUNT, episodes_to_eval)

        pbar = tqdm.tqdm(total=episodes_to_eval) if config.use_pbar else None
        log_str = (
            f"[Ckpt: {checkpoint_index}]"
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        while envs.num_envs > 0:
            current_episodes = envs.current_episodes()
            with torch.no_grad():
                actions, rnn_states = self.policy.act_iterative(
                    batch,
                    rnn_states,
                    prev_actions,
                    agent_episode_not_done_masks,
                    sim_episode_not_done_masks,
                    tour_not_done_masks,
                    action_masks,
                    deterministic=not config.EVAL.SAMPLE,
                )
                prev_actions.copy_(actions)

            outputs = envs.step([a[0].item() for a in actions])
            (
                observations,
                _,
                agent_episode_dones,
                sim_episode_dones,
                tour_dones,
                produce_actions,
                infos,
            ) = [list(x) for x in zip(*outputs)]

            # if zero, the agent called stop or reached the max step count.
            agent_episode_not_done_masks = torch.tensor(
                [[0] if done else [1] for done in agent_episode_dones],
                dtype=torch.uint8,
                device=self.device,
            )

            # if zero, the episode is completely finished: the agent is done
            # and the optional oracle navigator has finished conveying the
            # agent to the true goal location.
            sim_episode_not_done_masks = torch.tensor(
                [[0] if done else [1] for done in sim_episode_dones],
                dtype=torch.uint8,
                device=self.device,
            )

            # if zero, the tour has finished iterative eval.
            # The agent's map and/or inter-episode memory should be reset.
            tour_not_done_masks = torch.tensor(
                [[0] if done else [1] for done in tour_dones],
                dtype=torch.uint8,
                device=self.device,
            )

            # if zero, the agent's action prediction is ignored -- no episode
            # is active. Just update the map and/or inter-episode memory.
            action_masks = torch.tensor(
                produce_actions,
                dtype=torch.uint8,
                device=self.device,
            )

            reset_masks = (
                agent_episode_not_done_masks
                if self.config.EVAL.ITERATIVE_MAP_RESET == "episodic"
                else tour_not_done_masks
            )
            observations = add_batched_data_to_observations(
                observations, reset_masks, "not_done_masks"
            )

            # reset envs and observations if necessary
            for i in range(envs.num_envs):
                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(
                        frame, current_episodes[i].instruction.instruction_text
                    )
                    if (
                        "occupancy_map_viz" in batch
                        and "semantic_map_viz" in batch
                    ):
                        map_frame = [
                            batch["occupancy_map_viz"][i],
                            batch["semantic_map_viz"][i],
                        ]
                        map_frame = append_image_horizontally(map_frame)
                        frame = append_image_vertically([frame, map_frame])

                    rgb_frames[i].append(frame)

                if not agent_episode_dones[i]:
                    continue

                ep_id = current_episodes[i].episode_id
                tour_id = current_episodes[i].tour_id
                if ep_id not in stats_tours[tour_id] and len(infos[i]) > 1:
                    episodes_evaluated += 1
                    stats_tours[tour_id][ep_id] = {
                        k: v
                        for k, v in infos[i].items()
                        if isinstance(v, numbers.Number)
                    }

                    if config.use_pbar:
                        pbar.update()
                    else:
                        logger.info(
                            log_str.format(
                                evaluated=episodes_evaluated,
                                total=episodes_to_eval,
                                time=round(time.time() - start_time),
                            )
                        )

                if not sim_episode_dones[i]:
                    continue

                if "dtw_data" in infos[i]:
                    dtw_data[tour_id].extend(infos[i]["dtw_data"])

                (observations[i], tour_done, produce_action,) = envs.reset_at(
                    i
                )[0]
                tour_not_done_masks[i] = int(not tour_done)
                action_masks[i] = int(produce_action)
                prev_actions[i] = torch.zeros(1, dtype=torch.long)

                if len(config.VIDEO_OPTION) > 0:
                    tour_id = current_episodes[i].tour_id
                    ep_id = current_episodes[i].episode_id
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=current_episodes[i].episode_id,
                        checkpoint_idx=checkpoint_index,
                        metrics={"spl": stats_tours[tour_id][ep_id]["spl"]},
                        tb_writer=writer,
                    )
                    rgb_frames[i] = []

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )

            reset_masks = (
                agent_episode_not_done_masks
                if self.config.EVAL.ITERATIVE_MAP_RESET == "episodic"
                else tour_not_done_masks
            )
            observations = add_batched_data_to_observations(
                observations, reset_masks, "not_done_masks"
            )

            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if not sim_episode_dones[i]:
                    continue

                tour_id = next_episodes[i].tour_id
                if next_episodes[i].episode_id in stats_tours[tour_id]:
                    envs_to_pause.append(i)

            (
                envs,
                rnn_states,
                agent_episode_not_done_masks,
                sim_episode_not_done_masks,
                tour_not_done_masks,
                action_masks,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_iterative_envs(
                envs_to_pause,
                envs,
                rnn_states,
                agent_episode_not_done_masks,
                sim_episode_not_done_masks,
                tour_not_done_masks,
                action_masks,
                prev_actions,
                batch,
                rgb_frames,
            )

        envs.close()
        if config.use_pbar:
            pbar.close()

        # save DTW evaluation data
        split = config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            config.RESULTS_DIR,
            f"dtw_data_ckpt_{checkpoint_index}_{split}.json",
        )
        with open(fname, "w") as f:
            json.dump(dtw_data, f, indent=2)

        # save all episode stats for further analysis
        fname = os.path.join(
            config.RESULTS_DIR,
            f"iterative_all_stats_ckpt_{checkpoint_index}_{split}.json",
        )
        with open(fname, "w") as f:
            json.dump(stats_tours, f, indent=2)

        aggregated_stats = defaultdict(float)
        for stats_episodes in stats_tours.values():
            for stat_key in next(iter(stats_episodes.values())).keys():
                aggregated_stats[stat_key] += sum(
                    v[stat_key] for v in stats_episodes.values()
                )
        episodes_evaluated = sum(len(v) for v in stats_tours.values())
        for stat_key in aggregated_stats:
            aggregated_stats[stat_key] /= episodes_evaluated

        # compute t-ndtw
        with open(config.EVAL.ITERATIVE_GT_PATHS, "r") as f:
            gt_paths = json.load(f)
        aggregated_stats["tndtw"] = compute_tour_ndtw(
            agent_paths=dtw_data,
            gt_paths=gt_paths[split],
            success_distance=config.TASK_CONFIG.TASK.NDTW.SUCCESS_DISTANCE,
        )

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"iterative_stats_ckpt_{checkpoint_index}_{split}.json",
            )
            with open(fname, "w") as f:
                json.dump(aggregated_stats, f, indent=4)

        logger.info(f"Episodes evaluated: {episodes_evaluated}")
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.6f}")
            writer.add_scalar(f"eval_{split}_{k}", v, checkpoint_index + 1)
