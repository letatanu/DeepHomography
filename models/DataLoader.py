
from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import os

class DataLoader(Dataset):
    def __init__(self, metaData, patchSize = 128, pValue=32):
        super(DataLoader, self).__init__()
        self.patchSize = patchSize
        self.pValue = pValue
        self.metaData = metaData

    def __len__(self):
        return len(self.metaData)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img = np.array(cv2.imread(list(self.metaData)[index], flags=cv2.IMREAD_GRAYSCALE), dtype=np.float32)
        # return torch.from_numpy((img-127.5)/127.5)
        height, width = img.shape
        x = np.random.randint(low=self.pValue, high=width-self.pValue-self.patchSize)
        y = np.random.randint(low=self.pValue, high=height-self.pValue-self.patchSize)
        pos = np.array([x,y])
        croppedImg, croppedAppliedImage, H, H_AB = self.randomCreatePatch(img, pos)
        H, H_AB = np.array(H).flatten(), np.array(H_AB).flatten()

        if height != 128:
            croppedImg = cv2.resize(croppedImg, (128,128), interpolation=cv2.INTER_AREA)
            croppedAppliedImage = cv2.resize(croppedAppliedImage, (128,128), interpolation=cv2.INTER_AREA)

        croppedImg = (croppedImg - 127.5) / 127.5
        croppedAppliedImage = (croppedAppliedImage - 127.5) / 127.5

        left_Img = np.expand_dims(croppedImg, axis=2)
        right_Img = np.expand_dims(croppedAppliedImage, axis=2)
        input = np.concatenate((left_Img, right_Img), axis=2)
        input = np.rollaxis(input, 2, 0)
        return input, H, H_AB, pos

    def randomCreatePatch(self, image, position):
        croppedImage = image[position[1]: position[1] + self.patchSize, position[0]:position[0] + self.patchSize]
        H = np.random.randint(-self.pValue, self.pValue, size=(4, 2))
        src = np.array([
            position
            , [position[0] + self.patchSize, position[1] ]
            , [position[0] + self.patchSize, position[1] + self.patchSize]
            , [position[0] , position[1] + self.patchSize]
        ], dtype=np.float32)

        dest = np.array(src + H, dtype=np.float32)
        H_AB = cv2.getPerspectiveTransform(src, dest)
        H_BA = np.linalg.inv(H_AB)
        appliedImg = cv2.warpPerspective(image, H_BA, dsize=(image.shape[1], image.shape[0]))
        croppedAppliedImage = appliedImg[position[1]: position[1] + self.patchSize, position[0]:position[0] + self.patchSize]
        return croppedImage, croppedAppliedImage, H, H_AB


