
from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
from pathlib import Path
import h5py
class HDF5DataLoader(Dataset):
    """Represents an abstract HDF5 dataset.

        Input params:
            file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
            recursive: If True, searches for h5 files in subdirectories.
            load_data: If True, loads all the data immediately into RAM. Use this if
                the dataset is fits into memory. Otherwise, leave this at false and
                the data will load lazily.
            data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
            transform: PyTorch transform to apply to every data instance (default=None).
        """

    def __init__(self, file_path, recursive=False, data_cache_size=1, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert (p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        self.totalNumOfCaches = len(files)
        self.totalNumOfFiles = 0
        self.currentCacheIdx = 0
        for idx, h5dataset_fp in enumerate(files):
            self._add_data_infos(str(h5dataset_fp.resolve()))
        self._load_data()

    def __getitem__(self, index):
        # get data
        if torch.is_tensor(index):
            index = index.tolist()

        checkSize = 0
        for i, data in enumerate(self.data_info):
            checkSize += data["size"]
            if index < checkSize:
                if self.currentCacheIdx != i:
                    self.currentCacheIdx = i
                    self._load_data()
            else:
                break
        img = np.array(self.data_cache[self.currentCacheIdx][index], dtype=np.float32)
        return torch.tensor(img)

    def __len__(self):
        return self.totalNumOfFiles

    def _add_data_infos(self, file_path):
        with h5py.File(file_path, "r") as h5_file:
            # Walk through all groups, extracting datasets
            size = h5_file["images"].shape[0]
            self.totalNumOfFiles += size
            self.data_info.append({'file_path': file_path, "size": size})

    def _load_data(self):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        if len(self.data_cache) == self.data_cache_size:
            self.data_cache = {}

        for i in range(self.data_cache_size):
            id = (self.currentCacheIdx //self.data_cache_size)*self.data_cache_size+i
            if id < self.totalNumOfCaches:
                file_path = self.data_info[id]["file_path"]
                # remove an element from data cache if size was exceeded
                with h5py.File(file_path, "r") as h5_file:
                    images = np.array(h5_file["images"], dtype=np.float32)
                    self.data_cache[id] = images
            else:
                break





