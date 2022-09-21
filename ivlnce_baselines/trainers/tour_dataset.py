from collections import defaultdict
from logging import Logger
from typing import Dict, Iterable, List, Set, Tuple

import binpacking
import lmdb
import msgpack_numpy
import numpy as np
import torch


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def collate_fn(batch):
    """Each sample in batch: (
        obs,
        prev_actions,
        oracle_actions,
        inflec_weight,
    )
    """

    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(
            pad_amount, *t.size()[1:]
        )
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))

    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    tour_not_done_masks = list(transposed[4])

    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            )

    observations_batch = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid], max_traj_len, fill_val=1.0
            )

        prev_actions_batch[bid] = _pad_helper(
            prev_actions_batch[bid], max_traj_len
        )
        corrected_actions_batch[bid] = _pad_helper(
            corrected_actions_batch[bid], max_traj_len
        )
        weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)
        tour_not_done_masks[bid] = _pad_helper(
            tour_not_done_masks[bid], max_traj_len, fill_val=1
        )

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(
            observations_batch[sensor], dim=1
        )
        observations_batch[sensor] = observations_batch[sensor].view(
            -1, *observations_batch[sensor].size()[2:]
        )

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    tour_not_done_masks = torch.stack(tour_not_done_masks, dim=1).to(
        dtype=torch.uint8
    )
    episode_not_done_masks = torch.ones_like(
        corrected_actions_batch, dtype=torch.uint8
    )
    episode_not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch.view(-1, 1),
        episode_not_done_masks.view(-1, 1),
        tour_not_done_masks.view(-1, 1),
        corrected_actions_batch,
        weights_batch,
    )


class TourSampler(torch.utils.data.Sampler):

    batched_idxs: List[List[int]]
    tour_done_idxs: Set[int]
    batched_idxs_idx: int

    def __init__(
        self,
        tours_to_idx: Dict[int, List[int]],
        batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = True,
        logger: Logger = None,
    ) -> None:
        num_tours: int = len(tours_to_idx.keys())
        assert batch_size <= num_tours

        # How to handle tours in the DAgger sense? (TODO much later.)
        self.batched_idxs, self.tour_done_idxs = self._binpack_and_batch(
            tours_to_idx, batch_size, shuffle, drop_last
        )
        self.batched_idxs_idx = 0
        if logger is not None:
            pre_batch_eps = sum(len(v) for v in tours_to_idx.values())
            post_batch_eps = sum(len(b_t) for b_t in self.batched_idxs)
            logger.info("TourSampler:")
            logger.info(f"\tTours: {num_tours}")
            logger.info(f"\tNum pre-batch episodes: {pre_batch_eps}")
            logger.info(f"\tNum post-batch episodes: {post_batch_eps}")
            logger.info(
                f"\tEpisodes dropped: {pre_batch_eps - post_batch_eps}"
            )
            logger.info(f"\tNum batches: {len(self.batched_idxs)}")

    def _binpack_and_batch(
        self,
        tours_to_idx: Dict[int, List[int]],
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ) -> Tuple[List[List[int]], Set[int]]:
        """Creates a sequence of tour-iterative batches to train a model on.
        Problem reduction: "Multiway number partitioning"
            https://en.wikipedia.org/wiki/Multiway_number_partitioning
            solved greedily via https://github.com/benmaier/binpacking
        """
        bins = binpacking.to_constant_bin_number(
            {k: len(v) for k, v in tours_to_idx.items()}, batch_size
        )

        assert len(bins) == batch_size
        batches = [[] for _ in range(batch_size)]
        tour_done_idxs = set()
        for i, packed_bin in enumerate(bins):
            for k in packed_bin.keys():
                tour_ids = tours_to_idx[k]
                if shuffle:
                    np.random.shuffle(tour_ids)
                batches[i].extend(tour_ids)
                tour_done_idxs.add(tour_ids[0])

        transposed_batches = [
            [] for _ in range(max(len(seq) for seq in batches))
        ]
        for batch in batches:
            for i, elem in enumerate(batch):
                transposed_batches[i].append(elem)

        if drop_last:
            last_full_batch = len(transposed_batches) - 1
            for i, batch in enumerate(transposed_batches):
                if len(batch) < batch_size:
                    last_full_batch = i - 1
                    break
            transposed_batches = transposed_batches[:last_full_batch]

        return transposed_batches, tour_done_idxs

    def get_num_batches(self) -> int:
        return len(self.batched_idxs)

    def get_tour_done_idxs(self) -> Set[int]:
        """Return the set of idxs that start a new tour. This should be
        passed to the dataset so the model resets properly.
        """
        return self.tour_done_idxs

    def __len__(self) -> int:
        return len(self.batched_idxs)

    def __iter__(self) -> Iterable:
        return self

    def __next__(self) -> List[int]:
        self.batched_idxs_idx += 1
        if self.batched_idxs_idx > self.get_num_batches():
            raise StopIteration
        else:
            return self.batched_idxs[self.batched_idxs_idx - 1]


class TourTrajectoryDataset(torch.utils.data.Dataset):

    tour_done_idxs: Set[int]

    def __init__(
        self,
        lmdb_features_dir,
        use_iw,
        inflection_weight_coef=1.0,
        lmdb_map_size=1e9,
    ):
        # could still do a preload: pass sampler order to dataset somehow.
        super().__init__()
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.tour_done_idxs = None

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
        ) as lmdb_env:
            self.length = lmdb_env.stat()["entries"]

    def set_tour_done_idxs(self, tour_done_idxs: Set[int]) -> None:
        self.tour_done_idxs = tour_done_idxs

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Yes, creating a new transaction for every element load is
        inefficient. However, 10K transactions can be created in just 8.5
        seconds. Not an area to spend too much time optimizing.
        an optimization would be to create a single txn and close it by
        overriding the delete method.
        """
        assert (
            self.tour_done_idxs is not None
        ), "Call set_tour_done_idxs to set tour_done_idxs first."

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
        ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
            sample = msgpack_numpy.unpackb(
                txn.get(str(idx).encode()),
                raw=False,
            )

        obs, prev_actions, oracle_actions = sample

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )
        tour_done_mask = torch.ones_like(prev_actions)
        tour_done_mask[0] = int(idx not in self.tour_done_idxs)

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
            tour_done_mask,
        )
