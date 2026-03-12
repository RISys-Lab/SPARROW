import torch
from torch.utils.data import Sampler
from typing import Iterator, List, Sized


class EqualizedDatasetBatchSampler(Sampler):
    """
    Build batches from each sub-dataset with the same number of samples per epoch.

    The sampler assumes a ConcatDataset and uses dataset_sizes to map local
    indices to global indices. Each emitted batch contains samples from only one
    sub-dataset (compatible with source-homogeneous collators).
    """

    data_source: Sized

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        dataset_sizes: List[int],
        samples_per_dataset: int,
        seed: int = 42,
    ) -> None:
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.dataset_sizes = [int(x) for x in dataset_sizes]
        self.samples_per_dataset = int(samples_per_dataset)
        self.seed = int(seed)

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}.")
        if self.samples_per_dataset <= 0:
            raise ValueError(
                f"samples_per_dataset must be > 0, got {self.samples_per_dataset}."
            )

        self.dataset_offsets = []
        offset = 0
        for size in self.dataset_sizes:
            self.dataset_offsets.append(offset)
            offset += size

        self.samples_per_dataset_rounded = self.samples_per_dataset - (
            self.samples_per_dataset % self.batch_size
        )
        if self.samples_per_dataset_rounded <= 0:
            raise ValueError(
                "samples_per_dataset must be at least one full batch. "
                f"Got samples_per_dataset={self.samples_per_dataset}, batch_size={self.batch_size}."
            )

        # Keep per-dataset cursor + permutation so samples do not overlap until
        # the dataset is fully consumed.
        self._perm_rounds = [0 for _ in self.dataset_sizes]
        self._perms = [None for _ in self.dataset_sizes]
        self._positions = [0 for _ in self.dataset_sizes]
        self._epoch = 0

    @property
    def num_samples(self) -> int:
        return len(self.dataset_sizes) * self.samples_per_dataset_rounded

    def _refresh_perm(self, dataset_idx: int):
        size = self.dataset_sizes[dataset_idx]
        if size <= 0:
            raise ValueError(f"dataset_sizes[{dataset_idx}] is 0.")
        round_id = self._perm_rounds[dataset_idx]
        g = torch.Generator()
        # Stable but different permutation each refill.
        g.manual_seed(self.seed + dataset_idx * 1000003 + round_id * 9176)
        self._perms[dataset_idx] = torch.randperm(size, generator=g)
        self._positions[dataset_idx] = 0
        self._perm_rounds[dataset_idx] += 1

    def _take_local_indices(self, dataset_idx: int, count: int) -> torch.Tensor:
        size = self.dataset_sizes[dataset_idx]
        chunks = []
        remaining = int(count)
        while remaining > 0:
            if self._perms[dataset_idx] is None or self._positions[dataset_idx] >= size:
                self._refresh_perm(dataset_idx)

            pos = self._positions[dataset_idx]
            take = min(remaining, size - pos)
            chunks.append(self._perms[dataset_idx][pos: pos + take])
            self._positions[dataset_idx] = pos + take
            remaining -= take

        return torch.cat(chunks, dim=0)

    def __iter__(self) -> Iterator[int]:
        batches = []
        per_dataset_count = self.samples_per_dataset_rounded

        for dataset_idx, size in enumerate(self.dataset_sizes):
            if size <= 0:
                continue
            local_indices = self._take_local_indices(dataset_idx, per_dataset_count)
            global_indices = local_indices + self.dataset_offsets[dataset_idx]
            batches.extend(torch.split(global_indices, self.batch_size))

        if not batches:
            return

        # Shuffle batch order across data sources every epoch.
        order_gen = torch.Generator()
        order_gen.manual_seed(self.seed + self._epoch)
        batch_order = torch.randperm(len(batches), generator=order_gen).tolist()
        self._epoch += 1

        final_indices = torch.cat([batches[i] for i in batch_order], dim=0)
        yield from final_indices.tolist()

    def __len__(self) -> int:
        return self.num_samples
