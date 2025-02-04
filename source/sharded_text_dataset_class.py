from torch.utils.data import Dataset
import os
import torch
from datasets import load_from_disk, Dataset, concatenate_datasets
class ShardedDataset(torch.utils.data.Dataset):
    def __init__(self, shard_dir):
        self.shard_dir = shard_dir
        self.shard_files = sorted(os.listdir(shard_dir))  # Ensure files are sorted
        self.current_shard = None
        self.current_index = -1
        self.cumulative_sizes = []
        self.shard_lengths = []
        self.length = self.__calculate_total_length__()

    def __calculate_total_length__(self):
        dataset_length = 0
        for shard in self.shard_files:
            shard_length = len(load_from_disk(os.path.join(self.shard_dir, shard)))
            self.shard_lengths.append(shard_length)
            dataset_length += shard_length
        return dataset_length


    def __len__(self):
        if self.length is None:
            self.length = self._calculate_length()
        return self.length
    def load_shard_by_index(self, shard_index):
        if shard_index >= len(self.shard_files) or shard_index < 0:
            return False
        self.current_shard = load_from_disk(os.path.join(self.shard_dir, self.shard_files[shard_index]))
        self.current_index = shard_index
        return True

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {self.length}")

            # Find the shard containing the index
        cumulative_length = 0
        for shard_index, shard_length in enumerate(self.shard_lengths):
            if idx < cumulative_length + shard_length:
                # Load the shard if not already loaded
                if self.current_index != shard_index:
                    shard_path = os.path.join(self.shard_dir, self.shard_files[shard_index])
                    self.current_shard = load_from_disk(shard_path)
                    self.current_index = shard_index

                # Calculate local index within the shard
                local_idx = idx - cumulative_length
                return self.current_shard[local_idx]

            cumulative_length += shard_length

        # This point should not be reached if idx is valid
        raise RuntimeError("Failed to retrieve item from dataset.")
