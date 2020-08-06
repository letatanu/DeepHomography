
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

    def __init__(self, file_path, recursive=False, data_cache_size=2, transform=None, patchSize = 128, pValue=32):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform
        self.patchSize = patchSize
        self.pValue = pValue

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
        self.size = 0
        for idx, h5dataset_fp in enumerate(files):
            self._add_data_infos(str(h5dataset_fp.resolve()))

    def __getitem__(self, index):
        # get data
        if torch.is_tensor(index):
            index = index.tolist()

        # print(index)
        # print(size, self.currentCacheIdx, index//size)
        img = np.array(self.get_data(index), np.float32)

        height, width = img.shape
        x = np.random.randint(low=self.pValue, high=width - self.pValue - self.patchSize)
        y = np.random.randint(low=self.pValue, high=height - self.pValue - self.patchSize)
        pos = np.array([x, y], dtype=np.int)

        croppedImg, croppedAppliedImage, H, H_AB = self.randomCreatePatch(img, pos)
        H, H_AB = np.array(H).flatten(), np.array(H_AB).flatten()
        if self.patchSize != 128:
            croppedImg = cv2.resize(croppedImg, (128, 128), interpolation=cv2.INTER_AREA)
            croppedAppliedImage = cv2.resize(croppedAppliedImage, (128, 128), interpolation=cv2.INTER_AREA)

        croppedImg = (croppedImg - 127.5) / 127.5
        croppedAppliedImage = (croppedAppliedImage -127.5)/127.5

        input = self.stack2Arrays(croppedImg, croppedAppliedImage)
        input = np.rollaxis(input, 2, 0)
        return input, H/self.pValue, H_AB, pos

    def get_data(self, index):
        patchID = index // self.size
        file_path = self.data_info[patchID]["file_path"]
        if file_path not in self.data_cache:
            self._load_data(patchID)
        return self.data_cache[file_path][(index%self.size)%self.data_info[patchID]["size"]]

    def __len__(self):
        return self.totalNumOfFiles

    def _add_data_infos(self, file_path):
        with h5py.File(file_path, "r") as h5_file:
            # Walk through all groups, extracting datasets
            size = len(h5_file["images"])
            self.size = size if self.size == 0 else self.size
            self.totalNumOfFiles += size
            self.data_info.append({"file_path" : file_path, "size": size})

    def _load_data(self, patchID):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        if len(self.data_cache) == self.data_cache_size:
            self.data_cache = {}
        for id in range(self.data_cache_size):
            if patchID + id < self.totalNumOfCaches:
                file_path = self.data_info[patchID+id]["file_path"]
                with h5py.File(file_path, "r") as h5_file:
                    images = np.array(h5_file["images"])
                    self.data_cache[file_path] = images

        ##------------------- function to create label --------------------

    def stack2Arrays(self, array1, array2):
        array1 = np.expand_dims(array1, axis=2)
        array2 = np.expand_dims(array2, axis=2)
        return np.concatenate((array1, array2), axis=2)

    def randomCreatePatch(self, image, position):
        croppedImage = image[position[1]: position[1] + self.patchSize, position[0]:position[0] + self.patchSize]
        H = np.random.randint(-self.pValue, self.pValue, size=(4, 2))
        src = np.array([
            position
            , [position[0] + self.patchSize, position[1]]
            , [position[0] + self.patchSize, position[1] + self.patchSize]
            , [position[0], position[1] + self.patchSize]
        ], dtype=np.float32)

        dest = np.array(src + H, dtype=np.float32)
        H_AB = cv2.getPerspectiveTransform(src, dest)
        H_BA = np.linalg.inv(H_AB)
        appliedImg = cv2.warpPerspective(image, H_BA, dsize=(image.shape[1], image.shape[0]))
        croppedAppliedImage = appliedImg[position[1]: position[1] + self.patchSize,
                              position[0]:position[0] + self.patchSize]
        return croppedImage, croppedAppliedImage, H, H_AB



