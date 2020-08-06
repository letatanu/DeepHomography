import torch.nn as nn
import torch.functional as F
import torch
import cv2
import os
import shutil
import numpy as np
import pickle


class SIFT_RANSAC():
    def __init__(self, metaData, patchSize=256, pValue=64, ROOT_DIR=os.path.dirname(os.path.abspath(__file__)),
                 log_interval=100, L=1, visualDir="SIFT_TRAIN"):
        self.patchSize = patchSize
        self.pValue = pValue
        self.metaData = metaData
        self.log_interval = log_interval
        self.L = L
        self.ROOT_DIR = ROOT_DIR
        self.visualDir = os.path.join(ROOT_DIR, visualDir)
        if not os.path.isdir(self.visualDir):
            os.mkdir(self.visualDir)
        else:
            shutil.rmtree(self.visualDir)
            os.mkdir(self.visualDir)

    def cornerError(self, src, H_AB, predictedH_AB):
        src = np.float32(src).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(src.copy(), H_AB)

        # predictedDstPts = np.array([np.append(dstPt, 1) for  dstPt in predictedDstPts], dtype=np.float32)
        predictedDstPts = cv2.perspectiveTransform(src.copy(), predictedH_AB)

        # predictedDstPts = np.array(predictedDstPts)[:,:2]
        d = np.linalg.norm(np.array(dst) - np.array(predictedDstPts), axis=2)
        err = np.mean(d)
        return err, predictedDstPts, dst

    def randomCreatePatch(self, image, position):
        croppedImage = np.array(
            image[position[1]: position[1] + self.patchSize, position[0]:position[0] + self.patchSize], dtype=np.uint8)
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
        appliedImg = np.array(cv2.warpPerspective(image, H_BA, dsize=(image.shape[1], image.shape[0])), dtype=np.uint8)
        croppedAppliedImage = np.array(
            appliedImg[position[1]: position[1] + self.patchSize, position[0]:position[0] + self.patchSize],
            dtype=np.uint8)
        return croppedImage, croppedAppliedImage, H, H_AB, appliedImg, src

    def fit(self):
        min_max_count = 7
        threshold = 0.12
        orb = cv2.xfeatures2d.SIFT_create()
        # ---------------- Matching algorithm --------------------
        # FLANN_INDEX_KDTREE = 0
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # search_params = dict(checks=50)

        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        bf = cv2.BFMatcher()

        # --------------------------------------------------------
        ERR = []
        cornerERR = []
        summary = {
            "H": [],
            "pos": [],
            "err": [],
            "img": [],
            "matches": [],
            "cornerErr": []
        }
        matchesNum = 0
        err = np.nan
        cornerErr = np.nan

        for id, imgDir in enumerate(list(self.metaData)):
            img = np.array(cv2.imread(imgDir, flags=cv2.IMREAD_GRAYSCALE), dtype=np.float32)

            height, width = img.shape
            x = np.random.randint(low=self.pValue, high=width - self.pValue - self.patchSize)
            y = np.random.randint(low=self.pValue, high=height - self.pValue - self.patchSize)
            pos = np.array([x, y])

            croppedImg, croppedAppliedImage, H, H_AB, appliedImg, srcPos = self.randomCreatePatch(img, pos)
            summary["H"].append(H)
            summary["pos"].append(srcPos)
            summary["img"].append(imgDir)

            # ----------------------------------------------------------------

            kps_ref, descs_ref = orb.detectAndCompute(croppedImg, None)
            kps_sensed, descs_sensed = orb.detectAndCompute(croppedAppliedImage, None)
            if len(kps_ref) > 1 and len(kps_sensed) > 1:
                # matches = bf.match(descs_ref, descs_sensed)
                # matches = sorted(matches, key=lambda x: x.distance)
                matches = bf.knnMatch(descs_ref, descs_sensed, k=2)
                # matches = flann.knnMatch(descs_ref, descs_sensed, k=2)
                good = []
                for m, n in matches:
                    if m.distance < threshold * n.distance:
                        good.append([m])
                matchesNum = len(good)

                # matchesNum = len(matches)
                if matchesNum >= min_max_count:
                    # good = matches[:min_max_count]
                    dstPts = np.array([kps_ref[m[0].queryIdx].pt for m in good], dtype=np.float32)
                    srcPts = np.array([kps_sensed[m[0].trainIdx].pt for m in good], dtype=np.float32)
                    M, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC)
                    M = np.array(M, dtype=np.float32)

                    if self.L == 1:
                        err = np.average(np.abs(M - H_AB))
                    elif self.L == 2:
                        err = np.linalg.norm(M - H_AB)
                    if not np.isnan(err):
                        cornerErr, predictedDst, src = self.cornerError(srcPos, H_AB, M)
                        ERR.append(err)
                        cornerERR.append(cornerErr)
                        print("Img {} -- L2 Loss: {:0.6f}, Corner Err: {:0.6f}".format(id, err, cornerErr))

                        # ----------- visual testing -----------
                        if id % self.log_interval == 0 or cornerErr > 50:
                            sensedImgTransformed = cv2.polylines(img.copy(), [np.int32(predictedDst)], True, 125, 1,
                                                                 cv2.LINE_AA)
                            sensedImgTransformed = cv2.polylines(sensedImgTransformed, [np.int32(src)], True, 255, 1,
                                                                 cv2.LINE_AA)
                            w, h = 50, 25
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.rectangle(sensedImgTransformed, (7, 10), (w, h), (0, 0, 0), -1)
                            cv2.putText(sensedImgTransformed, '{:.4f}'.format(float(cornerErr)), (10, 20), font, 0.3,
                                        color=(255, 255, 255), lineType=cv2.LINE_AA)

                            cv2.imwrite(os.path.join(self.visualDir, "{}_sensedImgTestGT.png".format(id)),
                                        sensedImgTransformed)

                            # ----------- Visual Testing --------------------------------------
                            sensedImg = cv2.polylines(appliedImg, [np.int32(srcPos)], True, 255, 1, cv2.LINE_AA)
                            refImg = cv2.polylines(img.copy(), [np.int32(srcPos)], True, 255, 1, cv2.LINE_AA)
                            cv2.imwrite(os.path.join(self.visualDir, "{}_sensedImgGT.png".format(id)), sensedImg)
                            cv2.imwrite(os.path.join(self.visualDir, "{}_refImgGT.png".format(id)), refImg)

                            cv2.imwrite(os.path.join(self.visualDir, "{}_croppedImgGT.png".format(id)), croppedImg)
                            cv2.imwrite(os.path.join(self.visualDir, "{}_croppedAppliedImageGT.png".format(id)),
                                        croppedAppliedImage)
                            reappliedImg = cv2.warpPerspective(croppedAppliedImage, np.linalg.inv(M),
                                                               dsize=(croppedImg.shape[1], croppedImg.shape[0]))
                            cv2.imwrite(os.path.join(self.visualDir, "{}_RappliedImg.png".format(id)), reappliedImg)
                    else:
                        print("Error is Nan")
                        print("M ----- {}".format(M))
                        print("H_AB ---- {}".format(H_AB))
                        print("--------------------------")
                # ------------------------

            summary["err"].append(err)
            summary["matches"].append(matchesNum)
            summary["cornerErr"].append(cornerErr)
        avgErr = np.average(ERR)
        with open(os.path.join(self.visualDir, "SIFT_Sum.txt"), 'wb') as f:
            pickle.dump(summary, f)
        print("The average Err: {:0.6f}, corner Err: {:0.6f} of {} images of the total {} images".format(
            np.mean(np.array(cornerERR)), avgErr, len(ERR), len(list(self.metaData))))
        return ERR

class DeepHomography(nn.Module):
    def __init__(self, inputSize, outputSize=8, type="regression"):
        super(DeepHomography, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
            , nn.BatchNorm2d(64)
            , nn.ReLU()

            , nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
            , nn.BatchNorm2d(64)
            , nn.ReLU()

            , nn.MaxPool2d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
            , nn.BatchNorm2d(64)
            , nn.ReLU()

            , nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
            , nn.BatchNorm2d(64)
            , nn.ReLU()

            , nn.MaxPool2d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
            , nn.BatchNorm2d(128)
            , nn.ReLU()

            , nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
            , nn.BatchNorm2d(128)
            , nn.ReLU()

            , nn.MaxPool2d(kernel_size=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
            , nn.BatchNorm2d(128)
            , nn.ReLU()

            , nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
            , nn.Dropout(0.2)
            , nn.BatchNorm2d(128)
            , nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear((inputSize // 8) * (inputSize // 8) * 128, 1024)
            , nn.Dropout(0.2)
            , nn.LeakyReLU(0.2, inplace=True)
            , nn.Linear(1024, outputSize)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
