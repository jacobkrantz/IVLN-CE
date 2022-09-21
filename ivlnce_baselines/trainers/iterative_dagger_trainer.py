import gc
import json

import lmdb
import msgpack_numpy
import torch
import torch.nn.functional as F
import tqdm
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

from ivlnce_baselines.common.aux_losses import AuxLosses
from ivlnce_baselines.common.env_utils import construct_envs
from ivlnce_baselines.common.utils import batch_to
from ivlnce_baselines.trainers.iterative_collection_dagger_trainer import (
    IterativeCollectionDaggerTrainer,
)
from ivlnce_baselines.trainers.tour_dataset import (
    TourSampler,
    TourTrajectoryDataset,
    collate_fn,
)


@baseline_registry.register_trainer(name="iterative_dagger")
class IterativeDaggerTrainer(IterativeCollectionDaggerTrainer):
    def _update_agent(
        self,
        observations,
        prev_actions,
        episode_not_done_masks,
        tour_not_done_masks,
        corrected_actions,
        weights,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
        rnn_states=None,
    ):
        T, N = corrected_actions.size()

        reset_memory = not (
            self.config.MODEL.tour_memory
            or self.config.MODEL.tour_memory_variant
        )
        if rnn_states is None or reset_memory:
            rnn_states = torch.zeros(
                N,
                self.policy.net.num_recurrent_layers,
                self.config.MODEL.STATE_ENCODER.hidden_size,
                device=self.device,
            )
        if self.config.MODEL.tour_memory_variant:
            # reset just episodic memory
            rnn_states[:, : self.policy.net.num_recurrent_layers - 1] *= 0

        AuxLosses.clear()

        # NOTE: rnn_states grad removed
        distribution, rnn_states = self.policy.build_distribution(
            observations,
            rnn_states.detach(),
            prev_actions,
            episode_not_done_masks,
            tour_not_done_masks,
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
        return loss.item(), action_loss.item(), aux_loss, rnn_states

    def train(self) -> None:
        r"""Main method for training IterativeDagger.
        Returns:
            None
        """
        if self.config.IL.DAGGER.preload_lmdb_features:
            try:
                lmdb.open(self.lmdb_features_dir, readonly=True)
            except lmdb.Error as err:
                logger.error(
                    "Cannot open database for teacher forcing preload."
                )
                raise err
        else:
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.config.IL.DAGGER.lmdb_map_size),
            ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                txn.drop(lmdb_env.open_db())

        split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = split
        if (
            self.config.IL.DAGGER.expert_policy_sensor
            not in self.config.TASK_CONFIG.TASK.SENSORS
        ):
            self.config.TASK_CONFIG.TASK.SENSORS.append(
                self.config.IL.DAGGER.expert_policy_sensor
            )

        # if doing teacher forcing, don't switch the scene until it is complete
        if self.config.IL.DAGGER.p == 1.0:
            self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
                -1
            )
        self.config.freeze()

        # Extract the observation and action space.
        single_proc_config = self.config.clone()
        single_proc_config.defrost()
        single_proc_config.NUM_ENVIRONMENTS = 1
        single_proc_config.freeze()
        with construct_envs(
            single_proc_config, get_env_class(self.config.ENV_NAME)
        ) as envs:
            observation_space = envs.observation_spaces[0]
            action_space = envs.action_spaces[0]

        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR,
            flush_secs=self.flush_secs,
            purge_step=0,
        ) as writer:
            for dagger_it in range(self.config.IL.DAGGER.iterations):
                step_id = 0
                if self.config.IL.DAGGER.preload_lmdb_features:
                    with lmdb.open(
                        self.lmdb_features_dir,
                        map_size=int(self.config.IL.DAGGER.lmdb_map_size),
                        readonly=True,
                        lock=False,
                    ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                        tours_to_idxs = json.loads(
                            msgpack_numpy.unpackb(
                                txn.get(str(0).encode()),
                                raw=False,
                            ).decode()
                        )
                else:
                    tours_to_idxs = self._update_dataset(
                        dagger_it
                        + (1 if self.config.IL.load_from_ckpt else 0),
                        save_tour_idx_data=True,
                    )

                if torch.cuda.is_available():
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                gc.collect()

                AuxLosses.activate()
                for epoch in tqdm.trange(
                    self.config.IL.epochs, dynamic_ncols=True
                ):
                    dataset = TourTrajectoryDataset(
                        self.lmdb_features_dir,
                        self.config.IL.use_iw,
                        inflection_weight_coef=self.config.IL.inflection_weight_coef,
                        lmdb_map_size=self.config.IL.DAGGER.lmdb_map_size,
                    )
                    batch_sampler = TourSampler(
                        tours_to_idx=tours_to_idxs,
                        batch_size=self.config.IL.batch_size,
                        shuffle=True,
                        drop_last=True,
                        logger=logger,
                    )
                    num_batches = batch_sampler.get_num_batches()
                    tour_done_idxs = batch_sampler.get_tour_done_idxs()
                    dataset.set_tour_done_idxs(tour_done_idxs)
                    diter = torch.utils.data.DataLoader(
                        dataset,
                        batch_sampler=batch_sampler,
                        num_workers=4,
                        collate_fn=collate_fn,
                        pin_memory=False,
                    )

                    rnn_states = torch.zeros(
                        self.config.IL.batch_size,
                        self.policy.net.num_recurrent_layers,
                        self.config.MODEL.STATE_ENCODER.hidden_size,
                        device=self.device,
                    )

                    for batch in tqdm.tqdm(
                        diter,
                        total=num_batches,
                        leave=False,
                        dynamic_ncols=True,
                    ):
                        (
                            observations_batch,
                            prev_actions_batch,
                            episode_not_done_masks,
                            tour_not_done_mask,
                            corrected_actions_batch,
                            weights_batch,
                        ) = batch_to(batch, self.device, non_blocking=True)

                        (
                            loss,
                            action_loss,
                            aux_loss,
                            rnn_states,
                        ) = self._update_agent(
                            observations_batch,
                            prev_actions_batch,
                            episode_not_done_masks,
                            tour_not_done_mask,
                            corrected_actions_batch,
                            weights_batch,
                            rnn_states=rnn_states,
                        )

                        logger.info(f"train_loss: {loss}")
                        logger.info(f"train_action_loss: {action_loss}")
                        logger.info(f"train_aux_loss: {aux_loss}")
                        logger.info(f"Batches processed: {step_id + 1}.")
                        logger.info(
                            f"On DAgger iter {dagger_it}, Epoch {epoch}."
                        )
                        writer.add_scalar(
                            f"train_loss_iter_{dagger_it}", loss, step_id
                        )
                        writer.add_scalar(
                            f"train_action_loss_iter_{dagger_it}",
                            action_loss,
                            step_id,
                        )
                        writer.add_scalar(
                            f"train_aux_loss_iter_{dagger_it}",
                            aux_loss,
                            step_id,
                        )
                        step_id += 1  # noqa: SIM113

                    self.save_checkpoint(
                        f"ckpt.{dagger_it * self.config.IL.epochs + epoch}.pth",
                        dagger_it=dagger_it,
                        epoch=epoch,
                        step_id=step_id,
                    )
                AuxLosses.deactivate()
