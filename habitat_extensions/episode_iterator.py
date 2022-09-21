import os
import random
from collections import defaultdict
from typing import Dict, List, Sequence, TypeVar

from habitat.core.dataset import Episode, EpisodeIterator

T = TypeVar("T", bound=Episode)


class TourBasedEpisodeIterator(EpisodeIterator):
    def __init__(
        self,
        *args,
        episodes: Sequence[T],
        cycle: bool = True,
        seed: int = None,
        shuffle_tours: bool = True,
        shuffle_episodes: bool = True,
        specify_episode_order: bool = False,
        episode_order: Dict[str, List[List[int]]] = None,
        **kwargs,
    ) -> None:
        self._cycle = cycle
        self._step_count = 0
        self._shuffle_tours = shuffle_tours
        self._shuffle_episodes = shuffle_episodes
        self._specify_episode_order = specify_episode_order
        self._episode_order = episode_order

        if seed is not None:
            random.seed(seed)

        self.episodes = self._init_iterator(episodes)
        self._iterator = iter(self.episodes)
        self.tour_id_to_tour_size = self._compute_tour_sizes()

    def __iter__(self) -> "TourBasedEpisodeIterator":
        return self

    def __next__(self) -> T:
        next_episode = next(self._iterator, None)
        if next_episode is None:
            if not self._cycle:
                raise StopIteration

            self.episodes = self._init_iterator(self.episodes)
            self._iterator = iter(self.episodes)
            next_episode = next(self._iterator)

        return next_episode

    def _init_iterator(self, episodes: Sequence[T]) -> List[T]:
        sparse_tours = [
            [] for _ in range(1 + max(int(ep.tour_id) for ep in episodes))
        ]
        for ep in episodes:
            sparse_tours[int(ep.tour_id)].append(ep)
        tours = [t for t in sparse_tours if len(t)]

        # tour-based shuffle
        if self._shuffle_tours:
            random.shuffle(tours)

        # episode-based shuffle
        if self._shuffle_episodes:
            for t in tours:
                random.shuffle(t)

        if self._specify_episode_order:
            tours = [
                self._order_tour_episodes(t, self._episode_order)
                for t in tours
            ]

        return [ep for t in tours for ep in t]

    def _order_tour_episodes(self, tour, episode_order):
        ep_id = tour[0].episode_id
        scene = os.path.splitext(os.path.basename(tour[0].scene_id))[0]
        for ordered_t in episode_order[scene]:
            if ep_id in ordered_t:
                break
        else:
            raise AssertionError(
                f"episode ID {ep_id} not found in provided order."
            )

        # order `tour` based on `ordered_t` in O(nlogn)
        eid_to_idx = {eid: i for i, eid in enumerate(ordered_t)}
        tour = [(eid_to_idx[e.episode_id], e) for e in tour]
        return [e[1] for e in sorted(tour, key=lambda e: e[0])]

    def _compute_tour_sizes(self):
        tour_id_to_tour_size = defaultdict(int)
        for ep in self.episodes:
            tour_id_to_tour_size[ep.tour_id] += 1
        return tour_id_to_tour_size

    def num_episodes_in_tour(self, tour_id: str) -> int:
        return self.tour_id_to_tour_size[tour_id]
