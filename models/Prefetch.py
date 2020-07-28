import torch
import numpy as np
import cv2
import pyarrow as pa
class DataPrefetcher():
    def __init__(self, loader, patchSize = 128, pValue=32):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.patchSize = patchSize
        self.pValue = pValue
        self.preload()

    def preload(self):
        try:
            self.next_imgs = next(self.loader)
        except StopIteration:
            self.next_imgs = None
            self.next_input = torch.tensor([])
            self.next_H = torch.tensor([])
            self.next_H_AB = torch.tensor([])
            self.next_pos = torch.tensor([])
            return
        with torch.cuda.stream(self.stream):
            self.next_imgs = self.next_imgs.cuda(non_blocking=True)
            self.next_imgs = self.next_imgs.float()
            self.next_H = []
            self.next_H_AB = []
            self.next_pos = []
            self.next_input = []
            imgs = self.next_imgs.data.cpu().numpy()
            for img in imgs:
                height, width = img.shape
                x = np.random.randint(low=self.pValue, high=width - self.pValue - self.patchSize)
                y = np.random.randint(low=self.pValue, high=height - self.pValue - self.patchSize)
                pos = [x, y]
                self.next_pos.append(pos)
                croppedImg, croppedAppliedImage, H, H_AB = self.randomCreatePatch(img, pos)
                next_H, next_H_AB = np.array(H).flatten(), np.array(H_AB).flatten()
                self.next_H.append(next_H)
                self.next_H_AB.append(next_H_AB)
                if height != 128:
                    croppedImg = cv2.resize(croppedImg, (128, 128), interpolation=cv2.INTER_AREA)
                    croppedAppliedImage = cv2.resize(croppedAppliedImage, (128, 128), interpolation=cv2.INTER_AREA)

                # croppedImg = (croppedImg - 127.5) / 127.5
                # croppedAppliedImage = (croppedAppliedImage - 127.5) / 127.5

                input = self.stack2Arrays(croppedImg, croppedAppliedImage)
                input = np.rollaxis(input, 2, 0)
                self.next_input.append(input)

    def stack2Arrays(self, array1, array2):
        array1 = np.expand_dims(array1, axis=2)
        array2 = np.expand_dims(array2, axis=2)
        return np.concatenate((array1, array2), axis=2)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = torch.tensor(self.next_input)
        H = torch.tensor(self.next_H)
        H_AB = torch.tensor(self.next_H_AB)
        pos = torch.tensor(self.next_pos)
        self.preload()
        return input, H, H_AB, pos, self.next_imgs

    ##------------------- function to create label --------------------
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
