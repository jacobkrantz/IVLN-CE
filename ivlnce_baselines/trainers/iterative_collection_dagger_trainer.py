import json
from collections import defaultdict

import lmdb
import msgpack_numpy
import numpy as np
import torch
import tqdm
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)

from ivlnce_baselines.common.env_utils import construct_envs
from ivlnce_baselines.common.utils import (
    add_batched_data_to_observations,
    batch_obs,
    extract_instruction_tokens,
)
from ivlnce_baselines.trainers.dagger_trainer import DaggerTrainer


@baseline_registry.register_trainer(name="iterative_collection_dagger")
class IterativeCollectionDaggerTrainer(DaggerTrainer):
    """Changes the _update_dataset() function to"""

    def add_map_to_observations(self, observations, batch, num_envs):
        """adds occupancy_map and semantic_map to observations from batch.
        Removes observation keys used to generate maps that are no longer
        necessary.
        """
        map_k_sum = int("occupancy_map" in batch) + int(
            "semantic_map" in batch
        )
        if map_k_sum == 1:
            raise RuntimeError(
                "either both map keys should exist in the batch or neither"
            )
        elif map_k_sum != 2:
            return observations

        for i in range(num_envs):
            for k in ["occupancy_map", "semantic_map"]:
                observations[i][k] = batch[k][i].cpu().numpy()
                observations[i][k] = batch[k][i].cpu().numpy()

            for k in [
                "semantic",
                "semantic12",
                "world_robot_pose",
                "world_robot_orientation",
                "env_name",
            ]:
                if k in observations[i]:
                    del observations[i][k]

        return observations

    def save_episode_to_disk(self, episode, txn, lmdb_idx, expert_uuid):
        traj_obs = batch_obs(
            [step[0] for step in episode],
            device=torch.device("cpu"),
        )
        del traj_obs[expert_uuid]
        for k, v in traj_obs.items():
            traj_obs[k] = v.numpy()
            if self.config.IL.DAGGER.lmdb_fp16:
                traj_obs[k] = traj_obs[k].astype(np.float16)

        transposed_ep = [
            traj_obs,
            np.array([step[1] for step in episode], dtype=np.int64),
            np.array([step[2] for step in episode], dtype=np.int64),
        ]

        txn.put(
            str(lmdb_idx).encode(),
            msgpack_numpy.packb(transposed_ep, use_bin_type=True),
        )

    def masks_to_tensors(
        self,
        agent_episode_dones,
        sim_episode_dones,
        tour_dones,
        produce_actions,
    ):
        agent_episode_not_done_masks = torch.tensor(
            [[0] if done else [1] for done in agent_episode_dones],
            dtype=torch.uint8,
            device=self.device,
        )
        sim_episode_not_done_masks = torch.tensor(
            [[0] if done else [1] for done in sim_episode_dones],
            dtype=torch.uint8,
            device=self.device,
        )
        tour_not_done_masks = torch.tensor(
            [[0] if done else [1] for done in tour_dones],
            dtype=torch.uint8,
            device=self.device,
        )
        action_masks = torch.tensor(
            produce_actions,
            dtype=torch.uint8,
            device=self.device,
        )
        return (
            agent_episode_not_done_masks,
            sim_episode_not_done_masks,
            tour_not_done_masks,
            action_masks,
        )

    def batch_and_transform(self, observations, not_done_masks):
        """not_done_masks is used to reset maps. If tour_not_done_masks is
        used, then maps are reset only upon a new tour.
        """
        observations = extract_instruction_tokens(
            observations,
            self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
        )
        observations = add_batched_data_to_observations(
            observations, not_done_masks, "not_done_masks"
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        return batch, observations

    def _update_dataset(self, data_it: int, save_tour_idx_data: bool = False):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        expert_uuid = self.config.IL.DAGGER.expert_policy_sensor_uuid

        rnn_states = torch.zeros(
            envs.num_envs,
            self.policy.net.num_recurrent_layers,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            envs.num_envs,
            1,
            device=self.device,
            dtype=torch.long,
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

        batch, observations = self.batch_and_transform(
            observations, tour_not_done_masks
        )

        episodes = [[] for _ in range(envs.num_envs)]
        skips = [False for _ in range(envs.num_envs)]
        sim_episode_dones = [False for _ in range(envs.num_envs)]

        # https://arxiv.org/pdf/1011.0686.pdf
        # Theoretically, any beta function is fine so long as it converges to
        # zero as data_it -> inf. The paper suggests starting with beta = 1 and
        # exponential decay.
        p = self.config.IL.DAGGER.p
        # in Python 0.0 ** 0.0 == 1.0, but we want 0.0
        beta = 0.0 if p == 0.0 else p ** data_it

        ensure_unique_episodes = beta == 1.0

        def hook_builder(tgt_tensor):
            def hook(m, i, o):
                tgt_tensor.set_(o.cpu())

            return hook

        depth_features = None
        depth_hook = None
        if (
            not self.config.MODEL.DEPTH_ENCODER.trainable
            and self.config.MODEL.DEPTH_ENCODER.cnn_type
            == "VlnResnetDepthEncoder"
        ):
            depth_features = torch.zeros((1,), device="cpu")
            depth_hook = self.policy.net.depth_encoder.visual_encoder.register_forward_hook(
                hook_builder(depth_features)
            )

        rgb_features = None
        rgb_hook = None
        if not self.config.MODEL.RGB_ENCODER.trainable and hasattr(
            self.policy.net, "rgb_encoder"
        ):
            rgb_features = torch.zeros((1,), device="cpu")
            rgb_hook = self.policy.net.rgb_encoder.cnn.register_forward_hook(
                hook_builder(rgb_features)
            )

        collected_eps = 0
        ep_ids_collected = None
        if ensure_unique_episodes:
            ep_ids_collected = {
                ep.episode_id for ep in envs.current_episodes()
            }

        with tqdm.tqdm(
            total=self.config.IL.DAGGER.update_size, dynamic_ncols=True
        ) as pbar, lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.config.IL.DAGGER.lmdb_map_size),
        ) as lmdb_env, torch.no_grad():
            start_id = lmdb_env.stat()["entries"]
            txn = lmdb_env.begin(write=True)

            tours_to_idxs = defaultdict(list)
            if save_tour_idx_data:
                if start_id:
                    tours_to_idxs = defaultdict(
                        list, json.loads(txn.get(str(0).encode()).decode())
                    )
                else:
                    start_id += 1

            while collected_eps < self.config.IL.DAGGER.update_size:
                current_episodes = None
                envs_to_pause = None
                if ensure_unique_episodes:
                    envs_to_pause = []
                    current_episodes = envs.current_episodes()

                # when a sim episode is done, save it to disk.
                for i in range(envs.num_envs):
                    if not sim_episode_dones[i]:
                        continue

                    if skips[i]:
                        episodes[i] = []
                        continue

                    lmdb_idx = start_id + collected_eps
                    self.save_episode_to_disk(
                        episodes[i], txn, lmdb_idx, expert_uuid
                    )
                    tour_id = str(episodes[i][0][3])
                    tours_to_idxs[tour_id].append(lmdb_idx)
                    collected_eps += 1  # noqa: SIM113
                    pbar.update()

                    txn.commit()
                    txn = lmdb_env.begin(write=True)

                    if ensure_unique_episodes:
                        if current_episodes[i].episode_id in ep_ids_collected:
                            envs_to_pause.append(i)
                        else:
                            ep_ids_collected.add(
                                current_episodes[i].episode_id
                            )

                    episodes[i] = []

                if ensure_unique_episodes:
                    (
                        envs,
                        rnn_states,
                        agent_episode_not_done_masks,
                        sim_episode_not_done_masks,
                        tour_not_done_masks,
                        action_masks,
                        prev_actions,
                        batch,
                        _,
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
                    )
                    if envs.num_envs == 0:
                        break

                actions, rnn_states = self.policy.act_iterative(
                    batch,
                    rnn_states,
                    prev_actions,
                    agent_episode_not_done_masks,
                    sim_episode_not_done_masks,
                    tour_not_done_masks,
                    action_masks,
                    deterministic=False,
                )
                actions = torch.where(
                    torch.rand_like(actions, dtype=torch.float) < beta,
                    batch[expert_uuid].long(),
                    actions,
                )

                observations = self.add_map_to_observations(
                    observations, batch, envs.num_envs
                )
                for i, current_episode in enumerate(envs.current_episodes()):
                    # only add steps to lmdb if the agent is acting: skip oracle phases
                    if not action_masks[i]:
                        continue

                    if depth_features is not None:
                        observations[i]["depth_features"] = depth_features[i]
                        del observations[i]["depth"]

                    if rgb_features is not None:
                        observations[i]["rgb_features"] = rgb_features[i]

                    if "rgb" in observations[i]:
                        del observations[i]["rgb"]

                    episodes[i].append(
                        (
                            observations[i],
                            prev_actions[i].item(),
                            batch[expert_uuid][i].item(),
                            current_episode.tour_id,
                        )
                    )

                skips = batch[expert_uuid].long() == -1
                actions = torch.where(
                    skips, torch.zeros_like(actions), actions
                )
                skips = skips.squeeze(-1).to(device="cpu", non_blocking=True)
                prev_actions.copy_(actions)
                outputs = envs.step([a[0].item() for a in actions])

                (
                    observations,
                    _,
                    agent_episode_dones,
                    sim_episode_dones,
                    tour_dones,
                    produce_actions,
                    _,
                ) = [list(x) for x in zip(*outputs)]

                (
                    agent_episode_not_done_masks,
                    sim_episode_not_done_masks,
                    tour_not_done_masks,
                    action_masks,
                ) = self.masks_to_tensors(
                    agent_episode_dones,
                    sim_episode_dones,
                    tour_dones,
                    produce_actions,
                )

                batch, observations = self.batch_and_transform(
                    observations, tour_not_done_masks
                )

            if save_tour_idx_data:
                txn.put(
                    str(0).encode(),
                    msgpack_numpy.packb(
                        json.dumps(tours_to_idxs).encode(), use_bin_type=True
                    ),
                    overwrite=True,
                )
                txn.commit()

        envs.close()
        envs = None

        if depth_hook is not None:
            depth_hook.remove()
        if rgb_hook is not None:
            rgb_hook.remove()

        if save_tour_idx_data:
            return tours_to_idxs
        return None
