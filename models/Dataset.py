import os
import pickle
import numpy as np
import cv2
import shutil
import h5py
from pathlib import Path
class DataSet():
    def __init__(self, path, ROOT_DIR = os.path.dirname(os.path.abspath(__file__))):
        self.path = path
        self.rootDir = ROOT_DIR

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
        images = np.array(file["/images"], dtype=np.float)
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

    # def createDataset(self):
    #
    #     def randomCreatePatch(image, position, pValue, patchSize=128):
    #         croppedImage = image[position[0]: position[0] + patchSize, position[1]:position[1] + patchSize]
    #         H = np.random.uniform(-pValue, pValue, size=(4, 2))
    #
    #         src = np.array([
    #             position
    #             , [position[0], position[1] + patchSize]
    #             , [position[0] + patchSize, position[1] + patchSize]
    #             , [position[0] + patchSize, position[1]]
    #         ], dtype=np.float32)
    #
    #         dest = np.array(src + H, dtype=np.float32)
    #         H_AB = cv2.getPerspectiveTransform(src, dest)
    #         H_BA = np.linalg.inv(H_AB)
    #         appliedImg = cv2.warpPerspective(image, H_BA, dsize=(image.shape[1], image.shape[0]))
    #         croppedAppliedImage = appliedImg[position[0]: position[0] + patchSize, position[1]:position[1] + patchSize]
    #         return croppedImage, croppedAppliedImage, H, H_AB
    #
    #     def createdPair(imageDir, writingDir, patchSize=128, pValue=10):
    #         data = {}
    #         imagesDir = os.listdir(imageDir)
    #         imagesDir.sort()
    #         for index, img in enumerate(imagesDir):
    #             pairDir = os.path.join(writingDir, "{}".format(index).zfill(8))
    #             if not os.path.isdir(pairDir):
    #                 os.mkdir(pairDir)
    #             imgDir = os.path.join(imageDir, img)
    #             img = (np.array(cv2.imread(imgDir, flags=cv2.IMREAD_GRAYSCALE), dtype=np.float32))
    #             print("Writting at folder ... {}".format(pairDir))
    #             for i, pos in enumerate(positions):
    #                 croppedImg, croppedAppliedImage, H, H_AB = randomCreatePatch(img, pos, pValue=pValue, patchSize=patchSize)
    #                 croppedImgPath = os.path.join(pairDir, "cropped_{}.png".format(i))
    #                 croppedAppliedImagePath = os.path.join(pairDir, "croppedApplied_{}.png".format(i))
    #                 cv2.imwrite(croppedImgPath, croppedImg)
    #                 cv2.imwrite(croppedAppliedImagePath, croppedAppliedImage)
    #                 data[(croppedImgPath, croppedAppliedImagePath)] = [H, H_AB]
    #                 with open(os.path.join(pairDir, "hab_{}".format(i)), "wb") as f:
    #                     pickle.dump(H_AB, f)
    #                 with open(os.path.join(pairDir, "h_{}".format(i)), "wb") as f:
    #                     pickle.dump(H, f)
    #         return data
    #     datasetName = self.dsname
    #     dsPath = os.path.join(self.rootDir, datasetName)
    #     if not os.path.isdir(dsPath):
    #         os.mkdir(dsPath)
    #     else:
    #         shutil.rmtree(dsPath)
    #         os.mkdir(dsPath)
    #
    #     sequences = os.listdir(self.path)
    #     metaData = {}
    #     for sequence in sequences:
    #         img_0Dir = os.path.join(self.path, sequence, "image_0")
    #         img_1Dir = os.path.join(self.path, sequence, "image_1")
    #
    #         sqDsPath = os.path.join(dsPath, sequence)
    #         if not os.path.isdir(sqDsPath):
    #             os.mkdir(sqDsPath)
    #         # --------------------------------------------
    #         img_0 = os.path.join(sqDsPath, "image_0")
    #         if not os.path.isdir(img_0):
    #             os.mkdir(img_0)
    #         d0 = createdPair(img_0Dir, img_0, patchSize=self.patchSize, pValue=self.pValue)
    #         metaData = {**metaData, **d0}
    #
    #
    #         # --------------------------------------------
    #         img_1 = os.path.join(sqDsPath, "image_1")
    #         if not os.path.isdir(img_1):
    #             os.mkdir(img_1)
    #         d1  = createdPair(img_1Dir, img_1, patchSize=self.patchSize, pValue=self.pValue)
    #         metaData = {**metaData, **d1}
    #
    #     with open(os.path.join(dsPath, "metadata.txt"), 'wb') as f:
    #         pickle.dump(metaData, f)

    def readDS(self):
        dsPath = os.path.join(self.rootDir, self.path)
        ds = None
        with open(os.path.join(dsPath, "metadata.txt"), 'rb') as f:
            ds = pickle.load(f)
        return ds

    def dataSets(self, k_fold, kth):
        ds = self.readDS()
        trainSet = ds[:kth] + ds[kth + k_fold:-1000]
        valSet = ds[kth:kth + k_fold]
        testSet = ds[-1000:]
        return trainSet, valSet, testSet








