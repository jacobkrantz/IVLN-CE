from typing import Any, Dict, List, Optional, Tuple

import habitat
import habitat_sim
import numpy as np
from habitat import Config, Dataset, logger
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.geometry_utils import quaternion_from_coeff
from habitat_baselines.common.baseline_registry import baseline_registry

from habitat_extensions.utils import heading_from_quaternion


@baseline_registry.register_env(name="VLNCEDaggerEnv")
class VLNCEDaggerEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self) -> Tuple[float, float]:
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations: Observations) -> float:
        return 0.0

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over

    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="VLNCEIterativeEnv")
class VLNCEIterativeEnv(habitat.RLEnv):
    """Perform VLN-CE episodes iteratively. Once an episode is over, the agent
    is moved via oracle actions to the start location of the next episode.
    """

    _phase: str  # "agent"  "oracle_goal"  "oracle_start"
    _progress_check_steps: int = 0
    is_iterative: bool = True

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)
        self._phase = ""
        self.shortest_path_follower = ShortestPathFollower(
            self._env.sim,
            config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE,
            return_one_hot=False,
            stop_on_error=config.TASK_CONFIG.ENVIRONMENT.ITERATIVE.ORACLE_STOP_ON_ERROR,
        )
        self.dtw_data = []

    def _next_phase(self) -> None:
        assert self._phase in ["agent", "oracle_goal", "oracle_start"]
        if self._phase == "agent":
            self._phase = "oracle_goal"
        elif self._phase == "oracle_goal":
            self._phase = "oracle_start"
        elif self._phase == "oracle_start":
            self._phase = "agent"
        self._progress_check_steps = 0

    def get_reward_range(self) -> Tuple[float, float]:
        return (0.0, 0.0)

    def get_reward(self, observations: Observations) -> float:
        return 0.0

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over

    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()

    def append_dtw_step(self):
        self.dtw_data.append(
            {
                "position": self._env.sim.get_agent_state().position.tolist(),
                "phase": self._phase,
                "episode_id": self._env.current_episode.episode_id,
            }
        )

    def reset(self) -> Tuple[Observations, bool, bool]:
        """Resets the environment to a new episode.

        Returns:
            Tuple[Observations, bool, bool]:
                observations dictionary
                tour_done: is the current tour finished.
                produce_action: will an agent action be used in the next .step
        """
        self._phase = "agent"
        produce_action = True
        tour_done = True
        self._progress_check_steps = 0
        self.dtw_data = []

        try:
            prev_episode = self._env.current_episode
        except AssertionError:
            return self._env.reset(), tour_done, produce_action

        prev_agent_state = self._env.sim.get_agent_state()
        prev_tour_id = prev_episode.tour_id
        observations = self._env.reset()
        tour_done = prev_tour_id != self._env.current_episode.tour_id

        if tour_done:
            return observations, tour_done, produce_action

        if self._env._config.ENVIRONMENT.ITERATIVE.ORACLE_PHASES:
            # move agent back to the previous episode ending pose
            self._phase = "oracle_start"
            self._env.sim.set_agent_state(
                position=prev_agent_state.position,
                rotation=prev_agent_state.rotation,
                reset_sensors=True,
            )
            action, _ = self._get_next_action_safe(
                position_to=self._env.current_episode.start_position,
                heading_to=self._env.current_episode.start_rotation,
                teleport_on_failure=True,
            )

            if action == HabitatSimActions.STOP:
                self._next_phase()
            else:
                produce_action = False

        return observations, tour_done, produce_action

    def _get_next_action_safe(
        self,
        position_to: List[float],
        heading_to: Optional[List[float]] = None,
        teleport_on_failure: Optional[bool] = False,
    ) -> Tuple[int, bool]:
        """If _get_next_action fails, return the STOP action.

        if teleport_on_failure is True, then teleport to start location.
        TODO: teleportation should never happen: this messes up
            tour-based mappers.

        Alternative ideas:
        - navigate back to where the agent came from, them SPF to
            the desired point.
        - create an internal "recovery" phase that steps randomly
            until the desired point becomes navigable.
        - run a pretrained DDPPO pointgoal navigation agent.
            Issue: trained PointGoal agents are inflexible to sensor
            suite and embodiment choices.
        - reset to a new tour.
        """
        try:
            next_action = self._get_next_action(position_to, heading_to)
            step_limit = (
                self._env._config.ENVIRONMENT.ITERATIVE.ORACLE_STEP_ERROR_LIMIT
            )
            assert (
                self._progress_check_steps < step_limit or step_limit < 0
            ), "Too many oracle steps."
            succeeded = True
        except (habitat_sim.errors.GreedyFollowerError, AssertionError) as e:
            if isinstance(e, AssertionError):
                reason = "too many steps"
            else:
                reason = "GreedyFollowerError"
            logger.warn(
                f"Oracle _get_next_action() failed. Reason: {reason}."
                f" Episode: {self._env.current_episode.episode_id}"
                f" Position: {self._env.sim.get_agent_state().position}"
                f" Attempted Goal: {position_to}"
                f" Attempted Heading: {heading_to}"
                f" Phase: {self._phase}"
            )
            if teleport_on_failure:
                if heading_to is None:
                    heading_to = self._env.sim.get_agent_state().rotation
                self._env.sim.set_agent_state(
                    position=position_to,
                    rotation=heading_to,
                    reset_sensors=True,
                )

            next_action = HabitatSimActions.STOP
            succeeded = False

        return next_action, succeeded

    def _get_next_action(
        self,
        position_to: List[float],
        heading_to: Optional[List[float]] = None,
    ) -> int:
        """Computes the next oracle action using a shortest path follower.

        Args:
            position_to (List[float]): goal location of the shortest path
                follower. Stops within FORWARD_STEP_SIZE of the goal.
            heading_to (List[float], optional): Once within FORWARD_STEP_SIZE
                of the goal, the agent turns to `heading_to` if specified. The
                agent stops once the heading is off by less than TURN_ANGLE / 2.
                Defaults to None.

        Returns:
            int: oracle action from HabitatSimActions.
        """
        action = self.shortest_path_follower.get_next_action(position_to)
        if action == HabitatSimActions.STOP and heading_to is not None:
            start_rot = np.rad2deg(
                heading_from_quaternion(
                    quaternion_from_coeff(np.array(heading_to))
                )
            )
            current_rot = np.rad2deg(
                heading_from_quaternion(
                    self._env.sim.get_agent_state().rotation
                )
            )
            delta = ((((start_rot - current_rot) % 360) + 540) % 360) - 180
            if abs(delta) >= self._env._config.SIMULATOR.TURN_ANGLE / 2:
                if delta < 0:
                    action = HabitatSimActions.TURN_RIGHT
                else:
                    action = HabitatSimActions.TURN_LEFT

        return action

    def _step_oracle(self) -> Observations:
        """Takes an oracle step in the environment based on the current action
        phase.
        `self._phase` == "oracle_goal": actions convey the agent to the
        current episode's goal location.
        `self._phase` == "oracle_start": actions convey the agent to the
        current episode's start location and heading.

        If the oracle calls HabitatSimActions.STOP, `self._phase` is stepped.

        Returns:
            Observations
        """
        assert self._phase in ["oracle_goal", "oracle_start"]

        if self._phase == "oracle_goal":
            position_to = self._env.current_episode.goals[0].position
            heading_to = None
        else:
            position_to = self._env.current_episode.start_position
            heading_to = self._env.current_episode.start_rotation

        action = self._get_next_action(position_to, heading_to)
        observations = self._env.task.step(
            action={"action": action}, episode=self._env.current_episode
        )

        if self._phase == "oracle_goal":
            position_to = self._env.current_episode.goals[0].position
            heading_to = None
        else:
            position_to = self._env.current_episode.start_position
            heading_to = self._env.current_episode.start_rotation

        next_action, _ = self._get_next_action_safe(
            position_to,
            heading_to,
            teleport_on_failure=self._phase == "oracle_start",
        )

        if next_action == HabitatSimActions.STOP:
            if (
                self._phase == "oracle_start"
                and self._env._config.ENVIRONMENT.ITERATIVE.PRECISE_EPISODE_START
            ):
                self._env.sim.set_agent_state(
                    position=position_to,
                    rotation=heading_to,
                    reset_sensors=True,
                )
            self._next_phase()

        self._progress_check_steps += 1
        return observations

    def step(
        self, *args, **kwargs
    ) -> Tuple[Observations, Any, bool, bool, bool, bool, dict]:
        """Perform an action in the environment.

        Returns:
            Tuple[Observations, Any, bool, bool, bool, dict]:
                observations dictionary
                RL reward
                agent_episode_done: agent's episode over, oracle actions follow
                sim_episode_done: oracle actions are also done
                tour_done: is the current tour finished.
                produce_action: will an agent action be used in the next .step
        """
        observations = {}
        reward = 0.0
        agent_episode_done = True
        sim_episode_done = False
        tour_done = False
        produce_action = False
        info = {}

        self.append_dtw_step()

        if self._phase == "agent":
            observations = self._env.step(*args, **kwargs)
            reward = self.get_reward(observations)
            agent_episode_done = self.get_done(observations)
            produce_action = True
            info = self.get_info(observations)

            if agent_episode_done:
                self._next_phase()
                produce_action = False

                if not self._env._config.ENVIRONMENT.ITERATIVE.ORACLE_PHASES:
                    self._phase = "agent"
                    sim_episode_done = True
                else:
                    next_action, _ = self._get_next_action_safe(
                        self._env.current_episode.goals[0].position
                    )
                    if (
                        next_action == HabitatSimActions.STOP
                        or not self._env._config.ENVIRONMENT.ITERATIVE.ORACLE_GOAL_PHASE
                    ):
                        self._next_phase()
                        sim_episode_done = True

        elif self._phase == "oracle_goal":
            observations = self._step_oracle()
            if self._phase == "oracle_start":
                sim_episode_done = True

        elif self._phase == "oracle_start":
            observations = self._step_oracle()
            if self._phase == "agent":
                produce_action = True

        if agent_episode_done or sim_episode_done:
            info["dtw_data"] = self.dtw_data

        return (
            observations,
            reward,
            agent_episode_done,
            sim_episode_done,
            tour_done,
            produce_action,
            info,
        )
