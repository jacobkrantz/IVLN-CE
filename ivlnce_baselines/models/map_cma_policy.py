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
from ivlnce_baselines.models.encoders.map_encoder import SemanticMapEncoder
from ivlnce_baselines.models.encoders.resnet_encoders import (
    VlnResnetDepthEncoder,
)
from ivlnce_baselines.models.policy import ILPolicy


@baseline_registry.register_policy
class MapCMAPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        config: Config,
    ):
        super().__init__(
            MapCMANet(
                observation_space=observation_space,
                config=config,
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
        features, rnn_hidden_states = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            action_masks=agent_episode_not_done_masks,
            episode_masks=None,
            tour_masks=None,
        )
        distribution = self.action_distribution(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        return action, rnn_hidden_states

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

        features, rnn_hidden_states = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            action_masks=agent_episode_not_done_masks,
        )
        return self.action_distribution(features), rnn_hidden_states

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )


class MapCMANet(Net):
    r"""A cross-modal attention (CMA) network that contains:
    Instruction encoder
    Depth encoder
    Semantic Map encoder
    CMA state encoder
    """

    def __init__(self, observation_space: Space, config: Config, num_actions):
        super().__init__()
        model_config = config.MODEL

        self.model_config = model_config
        model_config.defrost()
        model_config.INSTRUCTION_ENCODER.final_state_only = False
        model_config.freeze()

        # Init the map encoder
        assert model_config.SEMANTIC_MAP_ENCODER.classname in [
            "SemanticMapEncoder"
        ], "SEMANTIC_MAP_ENCODER.classname must be SemanticMapEncoder"
        self.map_encoder = SemanticMapEncoder(
            observation_space,
            model_config.SEMANTIC_MAP_ENCODER.num_semantic_classes,
            model_config.SEMANTIC_MAP_ENCODER.channels,
            model_config.SEMANTIC_MAP_ENCODER.last_ch_mult,
            model_config.SEMANTIC_MAP_ENCODER.trainable,
            model_config.SEMANTIC_MAP_ENCODER.from_pretrained,
            model_config.SEMANTIC_MAP_ENCODER.checkpoint,
        )

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

        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        hidden_size = model_config.STATE_ENCODER.hidden_size
        self._hidden_size = hidden_size

        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.depth_encoder.output_shape),
                model_config.DEPTH_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )
        self.map_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.map_encoder.output_shape),
                model_config.SEMANTIC_MAP_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )

        # Init the RNN state decoder
        rnn_input_size = (
            model_config.DEPTH_ENCODER.output_size
            + model_config.SEMANTIC_MAP_ENCODER.output_size
            + self.prev_action_embedding.embedding_dim
        )

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
            num_layers=1,
        )

        self._output_size = (
            model_config.STATE_ENCODER.hidden_size
            + model_config.DEPTH_ENCODER.output_size
            + self.instruction_encoder.output_size
            + model_config.SEMANTIC_MAP_ENCODER.output_size
        )

        self.dep_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            hidden_size // 2 + model_config.DEPTH_ENCODER.output_size,
            1,
        )

        self.map_kv = nn.Conv1d(
            self.map_encoder.output_shape[0],
            hidden_size // 2 + model_config.SEMANTIC_MAP_ENCODER.output_size,
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

        self._output_size = model_config.STATE_ENCODER.hidden_size

        self.progress_monitor = nn.Linear(self.output_size, 1)

        self._init_layers()

        self.train()

        if not model_config.SEMANTIC_MAP_ENCODER.trainable:
            self.map_encoder.eval()  # freeze batchnorm layers

    @property
    def output_size(self):
        return self._output_size

    @property
    def is_blind(self):
        return self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return (
            self.state_encoder.num_recurrent_layers
            + self.second_state_encoder.num_recurrent_layers
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
        if episode_masks is None:
            episode_masks = action_masks
        if tour_masks is None:
            tour_masks = episode_masks

        s1_layers = self.state_encoder.num_recurrent_layers
        s2_layers = self.second_state_encoder.num_recurrent_layers

        txt_embedding = self.instruction_encoder(observations)
        dep_embedding = torch.flatten(self.depth_encoder(observations), 2)
        map_embedding = torch.flatten(self.map_encoder(observations), 2)

        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * action_masks).long().view(-1)
        )

        if self.model_config.ablate_instruction:
            txt_embedding = txt_embedding * 0
        if self.model_config.ablate_depth:
            dep_embedding = dep_embedding * 0
        if self.model_config.ablate_map:
            map_embedding = map_embedding * 0

        dep_in = self.depth_linear(dep_embedding)
        map_in = self.map_linear(map_embedding)

        state_inputs = [dep_in, map_in, prev_actions]
        state_in = torch.cat(state_inputs, dim=1)
        rnn_states_out = rnn_states.detach().clone()
        (state, rnn_states_out[:, 0:s1_layers],) = self.state_encoder(
            state_in,
            rnn_states[:, 0:s1_layers],
            episode_masks,
        )

        text_state_q = self.state_q(state)
        text_state_k = self.text_k(txt_embedding)
        text_mask = (txt_embedding == 0.0).all(dim=1)
        text_embedding = self._attn(
            text_state_q, text_state_k, txt_embedding, text_mask
        )

        half_h = self._hidden_size // 2
        dep_k, dep_v = torch.split(self.dep_kv(dep_embedding), half_h, dim=1)
        map_k, map_v = torch.split(self.map_kv(map_embedding), half_h, dim=1)

        text_q = self.text_q(text_embedding)
        dep_embedding = self._attn(text_q, dep_k, dep_v)
        map_embedding = self._attn(text_q, map_k, map_v)

        x = torch.cat(
            [
                state,
                text_embedding,
                dep_embedding,
                map_embedding,
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
