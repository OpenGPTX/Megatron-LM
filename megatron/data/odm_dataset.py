# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""
Online mixing dataset.

From https://arxiv.org/abs/2312.02406.
"""

import math

import numpy as np
import torch

from megatron import print_rank_0
from megatron.core import parallel_state


class ODMDataset(torch.utils.data.Dataset):

    def __init__(
            self, datasets, initial_weights, size, alpha,
            warmup_steps, global_batch_size, seed, *, data_cache_path=None,
    ):

        self.datasets = datasets
        self.num_datasets = len(datasets)
        assert self.num_datasets == len(initial_weights)

        self.size = size
        # Paper doesn't state selected value.
        self.alpha = alpha
        self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        data_parallel_rank = parallel_state.get_data_parallel_rank()
        self.seed = seed

        # By default 1 % of total steps (`self.size //
        # global_batch_size`) in paper.
        if 0 < warmup_steps < 1:
            warmup_steps = round(self.size / global_batch_size * warmup_steps)
        assert warmup_steps >= 0
        self.warmup_steps = warmup_steps

        self.sizes = [len(dataset) for dataset in datasets]
        shard_sizes = [
            len(dataset) // self.data_parallel_size
            for dataset in datasets
        ]
        self.index_starts = [
            shard_size * data_parallel_rank
            for shard_size in shard_sizes
        ]
        self.index_ends = [
            index_start + shard_size
            for (index_start, shard_size) in zip(
                    self.index_starts,
                    shard_sizes,
            )
        ]
        self.last_used_index = self.index_starts.copy()

        # Normalize weights.
        initial_weights = np.array(initial_weights, dtype=np.float64)
        sum_weights = np.sum(initial_weights)
        assert sum_weights > 0.0
        initial_weights /= sum_weights

        desc = "ODM dataset\n\n"
        desc += "Datasets:\n"
        for dataset in datasets:
            desc += dataset.desc + "\n\n"
        desc += f"Weights: {initial_weights}\n"
        desc += f"Size: {size}\n"
        desc += f"Alpha: {alpha}\n"
        desc += f"Warmup steps: {warmup_steps}\n"
        desc += f"Global batch size: {global_batch_size}\n"
        desc += f"Data parallel rank: {data_parallel_rank}\n"
        desc += f"Data parallel size: {self.data_parallel_size}\n"
        desc += f"Seed: {seed}\n"
        self.desc = desc

        self._step = 0
        # We initialize the sum correctly, so we don't need to update it.
        self._need_full_denominator_update = False
        self._batch_idx = 0
        self._prev_batch_idx = 0
        self._dataset_idx = 0
        # Mapping from dataset index to its loss sum.
        self._dataset_losses = {}

        self.eps = 1 / self.num_datasets
        self.rewards = np.zeros((self.num_datasets,))
        self.policy = np.empty((self.num_datasets,))
        # This would just be sum_{i = 1}^num_datasets exp(eps * 0)
        self.policy_denominator_sum = np.array(
            self.num_datasets, dtype=self.rewards.dtype)

        # Check size
        _ = self.get_index(self.size - 1)
        try:
            _ = self.get_index(self.size)
            raise RuntimeError('OMDDataset size is improperly bounded')
        except IndexError:
            pass
        self.update_policy(np.empty((0,)), np.empty((0,), dtype=np.int64))
        print_rank_0('> size of OMD dataset: '
                     '{} samples'.format(self.size))

    def __len__(self):
        return self.size

    def get_index(self, idx):
        dataset_idx = 0
        sample_idx = idx
        while (
                sample_idx - self.sizes[dataset_idx] >= 0
                and dataset_idx < self.num_datasets
        ):
            sample_idx = sample_idx - self.sizes[dataset_idx]
            dataset_idx += 1

        if idx < 0 or dataset_idx >= self.num_datasets:
            raise IndexError(f'index {idx} is out of bounds')

        return self.datasets[dataset_idx][sample_idx]

    def increase_batch_idx(self):
        self._batch_idx += 1

    def update_policy(self, old_rewards, dataset_idxs):
        prev_eps = self.eps
        self.eps = min(
            1 / self.num_datasets,
            math.sqrt(
                math.log(self.num_datasets)
                / (self.num_datasets * self._step)
            ),
        )

        # TODO Maybe reduce over data parallel ranks.
        if self._need_full_denominator_update:
            self.policy_denominator_sum = np.sum(
                np.exp(prev_eps * self.rewards))
        elif len(old_rewards) > 0 and len(dataset_idxs) > 0:
            assert len(old_rewards) == len(dataset_idxs)
            self.policy_denominator_sum = (
                self.policy_denominator_sum
                - np.sum(np.exp(prev_eps * old_rewards))
                + np.sum(np.exp(prev_eps * self.rewards[dataset_idxs]))
            )
        self.policy[:] = (
            (1 - self.num_datasets * self.eps)
            * (np.exp(prev_eps * self.rewards) / self.policy_denominator_sum)
            + self.eps
        )
        self._need_full_denominator_update = prev_eps != self.eps

    def store_loss(self, loss):
        # Could also use `setdefault` to allocate a scalar
        # `numpy.ndarray` every single time. Doing the hash look-up on
        # an int twice is probably faster.
        loss_numpy = loss.detach().cpu().numpy()
        if self._dataset_idx in self._dataset_losses:
            self._dataset_losses[self._dataset_idx] += loss_numpy
        else:
            self._dataset_losses[self._dataset_idx] = loss_numpy
        self.increase_batch_idx()

    def update_rewards(self):
        dataset_idxs = np.array(
            sorted(self._dataset_losses.keys()), dtype=np.int64)
        # Make sure `losses` and `dataset_idxs` have the same order. We
        # could assume they already do because of later Python versions
        # and dictonary keeping ordering, but let's just assume ordered
        # incides are nicer.
        losses = np.concat([
            self._dataset_losses[dataset_idx]
            for dataset_idx in dataset_idxs
        ])
        # TODO Can we unconditionally assume we don't need the `copy`?
        old_rewards = self.rewards[dataset_idxs].copy()
        self.rewards[dataset_idxs] = (
            self.alpha * self.rewards[dataset_idxs]
            + (
                (1 - self.alpha)
                * losses / self.policy[dataset_idxs]
            )
        )

        self._step += 1
        # TODO Reduce rewards over data parallel ranks?
        # TODO Maybe make flag for this. Also check other places.
        if self._step > self.warmup_steps:
            self.update_policy(old_rewards, dataset_idxs)

    def __getitem__(self, idx):
        # FIXME `MegatronPretrainingRandomSampler` needs special
        # handling, but unclear how. `batch_idx` can be handled by
        # updating it from the outside, but `last_used_index` is harder.
        # FIXME This doesn't work if random sampling via
        # `MegatronPretrainingRandomSampler`; `idx` is expected to
        # increase strictly monotonically.

        # Only sample new data group if we have a different batch. In
        # other words, re-use the sampled data group if we're still in
        # the same batch.
        if self._batch_idx != self._prev_batch_idx:
            rng = np.random.RandomState(seed=self.seed + idx)
            self._dataset_idx = rng.choices(self.num_datasets, p=self.policy)
            self._prev_batch_idx = self._batch_idx

        sample_idx = self.last_used_index[self._dataset_idx]
        # FIXME This doesn't work if random sampling via
        # `MegatronPretrainingRandomSampler`; `idx` is expected to
        # increase strictly monotonically.
        self.last_used_index[self._dataset_idx] += 1
        # Reset index back to start so we cycle the subdataset.
        if (
                self.last_used_index[self._dataset_idx]
                >= self.index_ends[self._dataset_idx]
        ):
            self.last_used_index[self._dataset_idx] = \
                self.index_starts[self._dataset_idx]

        return {
            "dataset_idx": self._dataset_idx,
            **self.datasets[self._dataset_idx][sample_idx],
        }
