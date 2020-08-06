import os
import pickle
import numpy as np
import cv2
import shutil
import h5py
from pathlib import Path
class DataSet():
    def __init__(self, path, ROOT_DIR = os.path.dirname(os.path.abspath(__file__)), patchSize=128, pValue = 32):
        self.path = path
        self.rootDir = ROOT_DIR
        self.patchSize =  patchSize
        self.pValue = pValue

    # ------------------------ Co Co ------------------------


    def createCoCoHDF5(self, numImages, datasetTyp="Train"):
        "resizedTrain2017"
        hdf5Dir = os.path.join(self.path, "hdf5{}".format(datasetTyp))
        if os.path.isdir(hdf5Dir):
            shutil.rmtree(hdf5Dir)
            os.mkdir(hdf5Dir)
        else:
            os.mkdir(hdf5Dir)


        trainPath = Path(os.path.join(self.path, "resized{}2017".format(datasetTyp)))
        files = sorted(trainPath.glob('*.jpg'))
        l = len(files)
        images = []
        id = 0
        print("Processing file ... {}".format(id))
        for index, file in enumerate(files):
            f = str(file.resolve())
            img = np.array(cv2.imread(f, flags=cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
            # img = (img - 127.5) / 127.5
            images.append(img)
            if (index+1) % (numImages) == 0 or index == l-1:
                self.createHDF5(images, hdf5Dir, id)
                id += 1
                print("Processing file ... {}".format(id))
                images = []

    def createDeepHomo_CoCoHDF5(self, numImages, datasetTyp="Train"):
        hdf5Dir = os.path.join(self.path, "hdf5_DeepHomo{}".format(datasetTyp))
        if os.path.isdir(hdf5Dir):
            shutil.rmtree(hdf5Dir)
            os.mkdir(hdf5Dir)
        else:
            os.mkdir(hdf5Dir)

        trainPath = Path(os.path.join(self.path, "resized{}2017".format(datasetTyp)))
        files = sorted(trainPath.glob('*.jpg'))
        l = len(files)
        images = []
        poses = []
        Hs = []
        id = 0
        print("Processing file ... {}".format(id))
        for index, file in enumerate(files):
            f = str(file.resolve())
            img = np.array(cv2.imread(f, flags=cv2.IMREAD_GRAYSCALE), dtype=np.uint8)

            height, width = img.shape
            m = 5 if self.patchSize == 128 else 1
            for j in range(m):
                x = np.random.randint(low=self.pValue, high=width - self.pValue - self.patchSize)
                y = np.random.randint(low=self.pValue, high=height - self.pValue - self.patchSize)
                pos = np.array([x, y], dtype=np.int)

                croppedImg, croppedAppliedImage, H, H_AB = self.randomCreatePatch(img, pos)
                H, H_AB = np.array(H).flatten(), np.array(H_AB).flatten()
                if self.patchSize != 128:
                    croppedImg = cv2.resize(croppedImg, (128, 128), interpolation=cv2.INTER_AREA)
                    croppedAppliedImage = cv2.resize(croppedAppliedImage, (128, 128), interpolation=cv2.INTER_AREA)

                input = self.stack2Arrays(croppedImg, croppedAppliedImage)
                input = np.rollaxis(input, 2, 0)
                images.append(input)
                poses.append(pos)
                Hs.append(H)
            if (index+1) % (numImages) == 0 or index == l-1:
                self.createHDF5DeepHomo(images,poses, Hs , hdf5Dir, id)
                id += 1
                print("Processing file ... {}".format(id))
                images = []

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

    def createHDF5DeepHomo(self, images, poses, Hs, path, id):
        newPath = os.path.join(path, "{}_many.h5".format(id))

        file = h5py.File(newPath, "w")

        # Create a dataset in the file
        dataset = file.create_dataset(
            "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
        )
        dataset = file.create_dataset(
            "poses", np.shape(poses), h5py.h5t.STD_I8BE, data=poses
        )
        dataset = file.create_dataset(
            "Hs", np.shape(Hs), h5py.h5t.STD_U8BE, data= Hs
        )
        file.close()
        print("Storing file {}".format(id))

    def createHDF5(self, images, path, id):
        newPath = os.path.join(path, "{}_many.h5".format(id))

        file = h5py.File(newPath, "w")

        # Create a dataset in the file
        dataset = file.create_dataset(
            "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
        )
        file.close()
        print("Storing file {}".format(id))

    def readHDF5(self, id):

        file = h5py.File(self.path / f"{id}_many.h5", "r+")
        images = np.array(file["images"], dtype=np.float)
        return images

    # -------------------------------

    def datasetCoCo(self):
        metaDataDir = os.path.join(self.path, "metadata.txt")
        if not os.path.isfile(metaDataDir):
            d = self.createMetaDataCoCo()
        else:
            with open(metaDataDir, "rb") as f:
                d = pickle.load(f)
        return d
    def createMetaDataCoCo(self):
        d = {"train": [],
             "val": [],
             "test": []
             }

        resizedImgPath = os.path.join(self.path, "resizedTrain2017")
        imgsDir = os.listdir(resizedImgPath)
        for imgName in imgsDir:
            imgDir = os.path.join(resizedImgPath, imgName)
            d["train"].append(imgDir)

        resizedImgPath = os.path.join(self.path, "resizedTest2017")
        imgsDir = os.listdir(resizedImgPath)
        for imgName in imgsDir:
            imgDir = os.path.join(resizedImgPath, imgName)
            d["test"].append(imgDir)

        resizedImgPath = os.path.join(self.path, "resizedVal2017")
        imgsDir = os.listdir(resizedImgPath)
        for imgName in imgsDir:
            imgDir = os.path.join(resizedImgPath, imgName)
            d["val"].append(imgDir)

        with open(os.path.join(self.path, "metadata.txt"), "wb") as f:
            pickle.dump(d, f)
        return d

    def resizeTrainCoCo(self):
        trainPath = os.path.join(self.path, "train2017")
        imgsDir = os.listdir(trainPath)

        resizedImgPath = os.path.join(self.path, "resizedTrain2017")
        if not os.path.isdir(resizedImgPath):
            os.mkdir(resizedImgPath)
        else:
            shutil.rmtree(resizedImgPath)
            os.mkdir(resizedImgPath)

        for imgName in imgsDir:
            imgDir = os.path.join(trainPath, imgName)
            img = cv2.imread(imgDir, flags=cv2.IMREAD_GRAYSCALE)
            resizedImg = cv2.resize(img, (320,240))
            cv2.imwrite(os.path.join(resizedImgPath, imgName), resizedImg)

    def resizeTestCoCo(self):
        trainPath = os.path.join(self.path, "test2017")
        imgsDir = os.listdir(trainPath)

        resizedImgPath = os.path.join(self.path, "resizedTest2017")
        if not os.path.isdir(resizedImgPath):
            os.mkdir(resizedImgPath)
        else:
            shutil.rmtree(resizedImgPath)
            os.mkdir(resizedImgPath)

        for imgName in imgsDir:
            imgDir = os.path.join(trainPath, imgName)
            img = cv2.imread(imgDir, flags=cv2.IMREAD_GRAYSCALE)
            resizedImg = cv2.resize(img, (320, 240))
            cv2.imwrite(os.path.join(resizedImgPath, imgName), resizedImg)

    def resizeValCoCo(self):
        trainPath = os.path.join(self.path, "val2017")
        imgsDir = os.listdir(trainPath)

        resizedImgPath = os.path.join(self.path, "resizedVal2017")
        if not os.path.isdir(resizedImgPath):
            os.mkdir(resizedImgPath)
        else:
            shutil.rmtree(resizedImgPath)
            os.mkdir(resizedImgPath)

        for imgName in imgsDir:
            imgDir = os.path.join(trainPath, imgName)
            img = cv2.imread(imgDir, flags=cv2.IMREAD_GRAYSCALE)
            resizedImg = cv2.resize(img, (640, 480))
            cv2.imwrite(os.path.join(resizedImgPath, imgName), resizedImg)


    def createMetaData(self):
        sequences = os.listdir(self.path)
        metaData = []
        for sequence in sequences:
            img_0Dir = os.path.join(self.path, sequence, "image_0")
            img_1Dir = os.path.join(self.path, sequence, "image_1")

            imgs_0 = os.listdir(img_0Dir)
            imgs_0Dirs = [os.path.join(img_0Dir, x) for x in imgs_0]

            imgs_1 = os.listdir(img_1Dir)
            imgs_1Dirs = [os.path.join(img_1Dir, x) for x in imgs_1]

            metaData.extend(imgs_0Dirs)
            metaData.extend(imgs_1Dirs)

        with open(os.path.join(self.path,"metadata.txt"), 'wb') as f:
            pickle.dump(metaData, f)
        return metaData

    def readDS(self):
        dsPath = os.path.join(self.rootDir, self.path)
        with open(os.path.join(dsPath, "metadata.txt"), 'rb') as f:
            ds = pickle.load(f)
        return ds

    def dataSets(self, k_fold, kth):
        ds = self.readDS()
        trainSet = ds[:kth] + ds[kth + k_fold:-1000]
        valSet = ds[kth:kth + k_fold]
        testSet = ds[-1000:]
        return trainSet, valSet, testSet


def main():
    db = DataSet("/media/slark/M2_SSD/Coco")
    db.resizeTrainCoCo()
    db.resizeValCoCo()
    db.resizeTestCoCo()

    db.createCoCoHDF5(5000, "Val")
    db.createCoCoHDF5(10000, "Train")
    db.createCoCoHDF5(10000, "Test")

if __name__ == '__main__':
    main()




