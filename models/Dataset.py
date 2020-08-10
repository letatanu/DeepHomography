import os
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

    def createHDF5(self, images, path, id):
        newPath = os.path.join(path, "{}_many.h5".format(id))

        file = h5py.File(newPath, "w")

        # Create a dataset in the file
        dataset = file.create_dataset(
            "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
        )
        file.close()
        print("Storing file {}".format(id))

    # -------------------------------


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




