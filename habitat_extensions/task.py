import gzip
import json
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Union

import attr
from habitat.config import Config
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode

from habitat_extensions.episode_iterator import TourBasedEpisodeIterator

DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"
ALL_LANGUAGES_MASK = "*"
ALL_ROLES_MASK = "*"
ALL_EPISODES_MASK = "*"


@attr.s(auto_attribs=True)
class ExtendedInstructionData:
    instruction_text: str = attr.ib(default=None, validator=not_none_validator)
    instruction_id: Optional[str] = attr.ib(default=None)
    language: Optional[str] = attr.ib(default=None)
    annotator_id: Optional[str] = attr.ib(default=None)
    edit_distance: Optional[float] = attr.ib(default=None)
    timed_instruction: Optional[List[Dict[str, Union[float, str]]]] = attr.ib(
        default=None
    )
    instruction_tokens: Optional[List[str]] = attr.ib(default=None)
    split: Optional[str] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class VLNExtendedEpisode(VLNEpisode):
    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    reference_path: Optional[List[List[float]]] = attr.ib(default=None)
    instruction: ExtendedInstructionData = attr.ib(
        default=None, validator=not_none_validator
    )
    trajectory_id: Optional[Union[int, str]] = attr.ib(default=None)
    tour_id: Optional[str] = attr.ib(default=None)


@registry.register_dataset(name="VLN-CE-v1")
class VLNCEDatasetV1(Dataset):
    """Loads the R2R VLN-CE dataset"""

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def _scene_from_episode(episode: VLNExtendedEpisode) -> str:
        """Helper method to get the scene name from an episode. Assumes
        the scene_id is formated /path/to/<scene_name>.<ext>
        """
        return os.path.splitext(os.path.basename(episode.scene_id))[0]

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        """Return a sorted list of scenes"""
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted(
            {cls._scene_from_episode(episode) for episode in dataset.episodes}
        )

    def get_episode_iterator(self, *args: Any, **kwargs: Any) -> Iterator:
        kwargs.pop("specify_episode_order")
        return super().get_episode_iterator(*args, **kwargs)

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        dataset_filename = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        if ALL_SCENES_MASK not in config.CONTENT_SCENES:
            scenes_to_load = set(config.CONTENT_SCENES)
            self.episodes = [
                episode
                for episode in self.episodes
                if self._scene_from_episode(episode) in scenes_to_load
            ]

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:

        deserialized = json.loads(json_str)
        self.instruction_vocab = VocabDict(
            word_list=deserialized["instruction_vocab"]["word_list"]
        )

        for episode in deserialized["episodes"]:
            # cast integer IDs to strings
            episode["episode_id"] = str(episode["episode_id"])
            episode["trajectory_id"] = str(episode["trajectory_id"])

            episode = VLNExtendedEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = InstructionData(**episode.instruction)
            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)


@registry.register_dataset(name="Iterative-VLN-CE")
class IterativeVLNCEDataset(VLNCEDatasetV1):

    # maps scene_id to a list of episode tours
    tours: Dict[str, List[int]]

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)
        if config is not None:
            with open(config.TOURS_FILE, "r") as f:
                self.tours = self._cast_tours_to_str(
                    json.load(f)[config.SPLIT]
                )
            self._init_episodes_by_tour(
                config.MIN_TOUR_SIZE,
                config.NUM_TOURS_SAMPLE,
                config.EPISODES_PER_TOUR,
            )

    @staticmethod
    def _cast_tours_to_str(tours):
        return {
            k: [[str(eid) for eid in tour] for tour in v]
            for k, v in tours.items()
        }

    def _init_episodes_by_tour(
        self,
        min_tour_size: int = -1,
        num_tours_to_sample: int = -1,
        episodes_per_tour: int = -1,
    ) -> None:
        """Initialize self.episodes according to tour configs."""
        tours_flattened = [
            t for scene_tours in self.tours.values() for t in scene_tours
        ]

        eid_to_tid = {}
        for i, tour in enumerate(tours_flattened):
            for episode in tour:
                eid_to_tid[str(episode)] = str(i)

        tours = [[] for _ in range(len(tours_flattened))]
        for ep in self.episodes:
            if ep.episode_id in eid_to_tid:
                ep.tour_id = eid_to_tid[ep.episode_id]
                tours[int(ep.tour_id)].append(ep)

        # purge small tour
        if min_tour_size >= 0:
            tours = [t for t in tours if len(t) >= min_tour_size]

        # sample tours
        if num_tours_to_sample >= 0:
            tours = random.sample(
                tours,
                k=min(num_tours_to_sample, len(tours)),
            )

        # sample episodes per tour
        if episodes_per_tour >= 0:
            tours = [
                random.sample(t, k=min(episodes_per_tour, len(t)))
                for t in tours
            ]

        self.episodes = [ep for t in tours for ep in t]

    def get_episode_iterator(self, *args: Any, **kwargs: Any) -> Iterator:
        return TourBasedEpisodeIterator(
            *args,
            episodes=self.episodes,
            episode_order=self.tours,
            **kwargs,
        )


@registry.register_dataset(name="RxR-VLN-CE-v1")
class RxRVLNCEDatasetV1(Dataset):
    """Loads the RxR VLN-CE Dataset."""

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict
    annotation_roles: List[str] = ["guide", "follower"]
    languages: List[str] = ["en-US", "en-IN", "hi-IN", "te-IN"]

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []
        self.config = config

        if config is None:
            return

        for role in self.extract_roles_from_config(config):
            with gzip.open(
                config.DATA_PATH.format(split=config.SPLIT, role=role), "rt"
            ) as f:
                self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        if ALL_SCENES_MASK not in config.CONTENT_SCENES:
            scenes_to_load = set(config.CONTENT_SCENES)
            self.episodes = [
                e
                for e in self.episodes
                if self.scene_from_scene_path(e.scene_id) in scenes_to_load
            ]

        if ALL_LANGUAGES_MASK not in config.LANGUAGES:
            languages_to_load = set(config.LANGUAGES)
            self.episodes = [
                episode
                for episode in self.episodes
                if self._language_from_episode(episode) in languages_to_load
            ]

        if ALL_EPISODES_MASK not in config.EPISODES_ALLOWED:
            ep_ids_before = {ep.episode_id for ep in self.episodes}
            ep_ids_to_purge = ep_ids_before - set(config.EPISODES_ALLOWED)
            self.episodes = [
                episode
                for episode in self.episodes
                if episode.episode_id not in ep_ids_to_purge
            ]

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:

        deserialized = json.loads(json_str)

        for episode in deserialized["episodes"]:
            episode = VLNExtendedEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = ExtendedInstructionData(
                **episode.instruction
            )
            episode.instruction.split = self.config.SPLIT
            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        """Return a sorted list of scenes"""
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted(
            {cls.scene_from_scene_path(e.scene_id) for e in dataset.episodes}
        )

    @classmethod
    def extract_roles_from_config(cls, config: Config) -> List[str]:
        if ALL_ROLES_MASK in config.ROLES:
            return cls.annotation_roles
        assert set(config.ROLES).issubset(set(cls.annotation_roles))
        return config.ROLES

    @classmethod
    def check_config_paths_exist(cls, config: Config) -> bool:
        return all(
            os.path.exists(
                config.DATA_PATH.format(split=config.SPLIT, role=role)
            )
            for role in cls.extract_roles_from_config(config)
        ) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def _scene_from_episode(episode: VLNEpisode) -> str:
        """Helper method to get the scene name from an episode.  Assumes
        the scene_id is formated /path/to/<scene_name>.<ext>
        """
        return os.path.splitext(os.path.basename(episode.scene_id))[0]

    @staticmethod
    def _language_from_episode(episode: VLNExtendedEpisode) -> str:
        return episode.instruction.language
