import abc
from typing import Any

from habitat_baselines.rl.ppo.policy import Policy

from ivlnce_baselines.common.utils import (
    CategoricalNet,
    CustomFixedCategorical,
)


class ILPolicy(Policy, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions):
        """Defines an imitation learning policy as having functions act() and
        build_distribution().
        """
        super(Policy, self).__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_states = self.net(
            observations, rnn_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        return action, rnn_states

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
        """Option to update internal memory between episodes. Predicts actions
        according to an iterative evaluation procedure.
        Default implementation calls .act(), ignoring inter-episode memory.
        """
        return self.act(
            observations,
            rnn_hidden_states,
            prev_actions,
            agent_episode_not_done_masks,
            deterministic=deterministic,
        )

    def get_value(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def evaluate_actions(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def build_distribution(
        self, observations, rnn_states, prev_actions, masks
    ) -> CustomFixedCategorical:
        features, rnn_states = self.net(
            observations, rnn_states, prev_actions, masks
        )
        return self.action_distribution(features)
