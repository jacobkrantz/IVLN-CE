from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net
from torch import Tensor

from ivlnce_baselines.common.aux_losses import AuxLosses
from ivlnce_baselines.common.utils import CustomFixedCategorical
from ivlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from ivlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
)
from ivlnce_baselines.models.policy import ILPolicy


@baseline_registry.register_policy
class LatentCMAPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        self.tour_memory = model_config.tour_memory
        self.tour_memory_variant = model_config.tour_memory_variant
        self.train_unrolled = model_config.train_unrolled
        super().__init__(
            LatentCMANet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    def act_iterative(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        agent_episode_not_done_masks,
        sim_episode_not_done_masks,
        tour_not_done_masks,
        action_masks,
        deterministic=False,
    ):
        """memory options:
        self.tour_memory: reset RNN states for a new tour (NOT episodic).
        self.tour_memory_variant: both episodic and tour-based RNN states.
        """
        if self.tour_memory_variant:
            episode_masks = agent_episode_not_done_masks
            tour_masks = tour_not_done_masks
        else:
            if self.tour_memory:
                episode_masks = tour_not_done_masks
            else:
                episode_masks = None
            tour_masks = None

        features, rnn_hidden_states = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            action_masks=agent_episode_not_done_masks,
            episode_masks=episode_masks,
            tour_masks=tour_masks,
        )
        distribution = self.action_distribution(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        return action, rnn_hidden_states

    def _view_sequential_inputs(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        agent_episode_not_done_masks,
        tour_not_done_masks=None,
    ):
        """reshape all inputs to iterative single forward. necessary for
        training the multi-step tour memory cell.
        """
        batch_size = rnn_hidden_states.size(0)
        seq_len = agent_episode_not_done_masks.size(0) // batch_size

        new_obs = {}
        for k, v in observations.items():
            if len(v.size()) < 2:
                new_obs[k] = v.view(seq_len, batch_size).permute(1, 0)
            else:
                new_obs[k] = v.view(seq_len, batch_size, *(v.size()[1:]))
                dims = list(range(2, len(new_obs[k].size())))
                new_obs[k] = new_obs[k].permute(1, 0, *dims)

        return (
            new_obs,
            prev_actions.view(seq_len, batch_size, 1).permute(1, 0, 2),
            agent_episode_not_done_masks.view(seq_len, batch_size, 1).permute(
                1, 0, 2
            ),
            tour_not_done_masks.view(seq_len, batch_size, 1).permute(1, 0, 2),
            seq_len,
            batch_size,
        )

    def build_distribution(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        agent_episode_not_done_masks,
        tour_not_done_masks=None,
    ) -> Tuple[CustomFixedCategorical, Tensor]:
        if tour_not_done_masks is None:
            tour_not_done_masks = agent_episode_not_done_masks.clone()

        if self.tour_memory_variant or self.train_unrolled:
            (
                new_obs,
                p_acts,
                agent_masks,
                tour_masks,
                seq_len,
                batch_size,
            ) = self._view_sequential_inputs(
                observations,
                rnn_hidden_states,
                prev_actions,
                agent_episode_not_done_masks,
                tour_not_done_masks,
            )
            features = []
            for i in range(seq_len):
                single_features, rnn_hidden_states = self.net(
                    {k: v[:, i] for k, v in new_obs.items()},
                    rnn_hidden_states,
                    p_acts[:, i],
                    action_masks=agent_masks[:, i],
                    episode_masks=agent_masks[:, i],
                    tour_masks=tour_masks[:, i],
                )
                features.append(single_features)

            features = (
                torch.stack(features, dim=1)
                .permute(1, 0, 2)
                .contiguous()
                .view(batch_size * seq_len, -1)
            )
        else:
            features, rnn_hidden_states = self.net(
                observations,
                rnn_hidden_states,
                prev_actions,
                action_masks=agent_episode_not_done_masks,
                episode_masks=tour_not_done_masks
                if self.tour_memory
                else None,
                tour_masks=None,
            )
        return self.action_distribution(features), rnn_hidden_states

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )


class LatentCMANet(Net):
    r"""A cross-modal attention (CMA) network that contains:
    Instruction encoder
    Depth encoder
    RGB encoder
    CMA state encoder
    """

    def __init__(
        self, observation_space: Space, model_config: Config, num_actions
    ):
        super().__init__()
        self.model_config = model_config
        model_config.defrost()
        model_config.INSTRUCTION_ENCODER.final_state_only = False
        model_config.freeze()

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(
            model_config.INSTRUCTION_ENCODER
        )

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=True,
        )

        # Init the RGB encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "TorchVisionResNet50"
        ], "RGB_ENCODER.cnn_type must be TorchVisionResNet50"

        self.rgb_encoder = TorchVisionResNet50(
            output_size=model_config.RGB_ENCODER.output_size,
            spatial_output=True,
        )

        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        hidden_size = model_config.STATE_ENCODER.hidden_size
        self._hidden_size = hidden_size

        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                self.rgb_encoder.output_shape[0],
                model_config.RGB_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.depth_encoder.output_shape),
                model_config.DEPTH_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )

        # Init the RNN state decoder
        rnn_input_size = model_config.DEPTH_ENCODER.output_size
        rnn_input_size += model_config.RGB_ENCODER.output_size
        rnn_input_size += self.prev_action_embedding.embedding_dim
        if self.model_config.tour_memory_variant:
            rnn_input_size += model_config.STATE_ENCODER.hidden_size

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
            num_layers=1,
        )

        self._output_size = (
            model_config.STATE_ENCODER.hidden_size
            + model_config.RGB_ENCODER.output_size
            + model_config.DEPTH_ENCODER.output_size
            + self.instruction_encoder.output_size
        )

        self.rgb_kv = nn.Conv1d(
            self.rgb_encoder.output_shape[0],
            hidden_size // 2 + model_config.RGB_ENCODER.output_size,
            1,
        )

        self.depth_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            hidden_size // 2 + model_config.DEPTH_ENCODER.output_size,
            1,
        )

        self.state_q = nn.Linear(hidden_size, hidden_size // 2)
        self.text_k = nn.Conv1d(
            self.instruction_encoder.output_size, hidden_size // 2, 1
        )
        self.text_q = nn.Linear(
            self.instruction_encoder.output_size, hidden_size // 2
        )

        self.register_buffer(
            "_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5))
        )

        self.second_state_compress = nn.Sequential(
            nn.Linear(
                self._output_size + self.prev_action_embedding.embedding_dim,
                self._hidden_size,
            ),
            nn.ReLU(True),
        )

        self.second_state_encoder = build_rnn_state_encoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
            num_layers=1,
        )

        assert (
            not self.model_config.memory_at_end
        ) or self.model_config.tour_memory_variant, (
            "`memory_at_end` requires `tour_memory_variant`."
        )
        if self.model_config.memory_at_end:
            self.out_layer = nn.Sequential(
                nn.Linear(self._hidden_size * 2, self._hidden_size),
                nn.ReLU(True),
            )

        self._output_size = model_config.STATE_ENCODER.hidden_size

        self.progress_monitor = nn.Linear(self.output_size, 1)

        self._init_layers()

        self.train()

    @property
    def output_size(self):
        return self._output_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return (
            self.state_encoder.num_recurrent_layers
            + self.second_state_encoder.num_recurrent_layers
            + int(self.model_config.tour_memory_variant)
        )

    def _init_layers(self):
        if self.model_config.PROGRESS_MONITOR.use:
            nn.init.kaiming_normal_(
                self.progress_monitor.weight, nonlinearity="tanh"
            )
            nn.init.constant_(self.progress_monitor.bias, 0)

    def _attn(self, q, k, v, mask=None):
        logits = torch.einsum("nc, nci -> ni", q, k)

        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum("ni, nci -> nc", attn, v)

    def forward(
        self,
        observations,
        rnn_states,
        prev_actions,
        action_masks,
        episode_masks=None,
        tour_masks=None,
    ):
        if self.model_config.disable_tour_memory:
            tour_masks = None

        if episode_masks is None:
            episode_masks = action_masks
        if tour_masks is None:
            tour_masks = episode_masks

        s1_layers = self.state_encoder.num_recurrent_layers
        s2_layers = self.second_state_encoder.num_recurrent_layers

        if self.model_config.tour_memory_variant:
            rnn_states[:, s1_layers + s2_layers :] = (
                tour_masks.view(-1, 1, 1)
                * rnn_states[:, s1_layers + s2_layers :]
            )

        instruction_embedding = self.instruction_encoder(observations)
        depth_embedding = self.depth_encoder(observations)
        depth_embedding = torch.flatten(depth_embedding, 2)

        rgb_embedding = self.rgb_encoder(observations)
        rgb_embedding = torch.flatten(rgb_embedding, 2)

        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * action_masks).long().view(-1)
        )

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        rgb_in = self.rgb_linear(rgb_embedding)
        depth_in = self.depth_linear(depth_embedding)

        state_inputs = [rgb_in, depth_in, prev_actions]
        if self.model_config.tour_memory_variant:
            state_inputs.append(rnn_states[:, s1_layers + s2_layers])
        state_in = torch.cat(state_inputs, dim=1)
        rnn_states_out = rnn_states.detach().clone()
        (state, rnn_states_out[:, 0:s1_layers],) = self.state_encoder(
            state_in,
            rnn_states[:, 0:s1_layers],
            episode_masks,
        )

        # update cross-episode memory with the output of the state encoder.
        if self.model_config.tour_memory_variant:
            with torch.no_grad():
                rnn_states_out[:, s1_layers + s2_layers :] = torch.max(
                    rnn_states_out[:, s1_layers + s2_layers :],
                    rnn_states_out[:, 0:s1_layers],
                )

        text_state_q = self.state_q(state)
        text_state_k = self.text_k(instruction_embedding)
        text_mask = (instruction_embedding == 0.0).all(dim=1)
        text_embedding = self._attn(
            text_state_q, text_state_k, instruction_embedding, text_mask
        )

        rgb_k, rgb_v = torch.split(
            self.rgb_kv(rgb_embedding), self._hidden_size // 2, dim=1
        )
        depth_k, depth_v = torch.split(
            self.depth_kv(depth_embedding), self._hidden_size // 2, dim=1
        )

        text_q = self.text_q(text_embedding)
        rgb_embedding = self._attn(text_q, rgb_k, rgb_v)
        depth_embedding = self._attn(text_q, depth_k, depth_v)

        x = torch.cat(
            [
                state,
                text_embedding,
                rgb_embedding,
                depth_embedding,
                prev_actions,
            ],
            dim=1,
        )
        x = self.second_state_compress(x)
        (
            x,
            rnn_states_out[:, s1_layers : s1_layers + s2_layers],
        ) = self.second_state_encoder(
            x,
            rnn_states[:, s1_layers : s1_layers + s2_layers],
            episode_masks,
        )

        if self.model_config.memory_at_end:
            x = self.out_layer(
                torch.cat([x, rnn_states[:, s1_layers + s2_layers]], dim=1)
            )

        if self.model_config.PROGRESS_MONITOR.use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_loss = F.mse_loss(
                progress_hat.squeeze(1),
                observations["progress"],
                reduction="none",
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss,
                self.model_config.PROGRESS_MONITOR.alpha,
            )

        return x, rnn_states_out
