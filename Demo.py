from models.HDF5Data import  HDF5DataLoader
from models.model import  DeepHomography, SIFT_RANSAC
import argparse
import datetime
import time
import torch.multiprocessing as mp
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import cv2
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import os
import math

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
'''Dataset path'''
DS_PATH = "/media/slark/Data_1T_ssd/CoCo"


def randomCreatePatch(image, position, patchSize, pValue):
    '''
    This function is to create the random patch positions, random perturbed 4-point
    :param image:
    :param position:
    :param patchSize:
    :param pValue:
    :return:
    '''
    patchImage = image[position[1]: position[1] + patchSize, position[0]:position[0] + patchSize]
    H = np.random.randint(-pValue, pValue, size=(4, 2))
    p = np.array([
        position
        , [position[0] + patchSize, position[1]]
        , [position[0] + patchSize, position[1] + patchSize]
        , [position[0], position[1] + patchSize]
    ], dtype=np.float32)

    perturbedP = np.array(p + H, dtype=np.float32)
    H_AB = cv2.getPerspectiveTransform(p, perturbedP)
    H_BA = np.linalg.inv(H_AB)
    warpedImage = cv2.warpPerspective(image, H_BA, dsize=(image.shape[1], image.shape[0]))
    perturbedPatchImage = warpedImage[position[1]: position[1] + patchSize,
                            position[0]:position[0] + patchSize]
    return patchImage, perturbedPatchImage, H, H_AB, warpedImage, p, perturbedP

def stack2Arrays(array1, array2):
    '''
    This function is to stack 2 arrays in the dimension 2
    :param array1:
    :param array2:
    :return: stack 2 arrays in 2-axis
    '''
    array1 = np.expand_dims(array1, axis=2)
    array2 = np.expand_dims(array2, axis=2)
    return np.concatenate((array1, array2), axis=2)

def preprocessing(imgUrl, patchSize=128, pValue=32):
    '''
    This function is to create the random patch positions, random perturbed 4-point
    :param imgUrl: path of the image
    :param patchSize:
    :param pValue:
    :return: stacked image of patch and perturbed patch, normalized H, H_AB, patch image, perturbed patch image, P, perturbed P
    '''
    img = np.array(cv2.imread(imgUrl, flags=cv2.IMREAD_GRAYSCALE), dtype=np.float)
        
    height, width = img.shape
    x = np.random.randint(low=pValue, high=width - pValue - patchSize)
    y = np.random.randint(low=pValue, high=height - pValue - patchSize)
    pos = np.array([x, y], dtype=np.int)
    source, dst, H, H_AB, _, p, perturbedP = randomCreatePatch(img, pos, patchSize=patchSize, pValue=pValue)
    H, H_AB = np.array(H).flatten(), np.array(H_AB).flatten()
    if patchSize != 128:
        source = cv2.resize(source, (128, 128), interpolation=cv2.INTER_AREA)
        dst = cv2.resize(dst, (128, 128), interpolation=cv2.INTER_AREA)

    croppedImg = (source - 127.5) / 127.5
    croppedAppliedImage = (dst -127.5)/127.5

    input = stack2Arrays(croppedImg, croppedAppliedImage)
    input = np.rollaxis(input, 2, 0)
    return input, H/pValue, H_AB, source, dst, p, perturbedP

def demo(model, device, imgUrl, patchSize=128, pValue=32):
    model.eval()
    print(50 * "*")
    print("Running demo .... ")
    print(50 * "*")
    criterion = nn.MSELoss()
    data, target, H_AB, source, dst, p, perturbedP = preprocessing(imgUrl, patchSize=patchSize, pValue=pValue)
    with torch.no_grad():
        data, target = torch.from_numpy(data), torch.from_numpy(target)
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
        data = torch.unsqueeze(data, 0)
        target = torch.unsqueeze(target, 0)
        output = model(data)
        loss = criterion(output, target)
        print('\nValidation set: Average Error: {:.6f}'.format(loss.item()))

        predictedH = np.reshape(output.cpu().numpy(), (4,2)) * pValue
        H = np.reshape(target.cpu().numpy(), (4,2)) *pValue
        predictedPerturbedP = predictedH + p
        visualize(imgUrl, H, predictedH, p, perturbedP, predictedPerturbedP, patchSize)

def visualize(imgUrl, H, predictedH, p, perturbedP, predictedPerturbedP, patchSize):
    '''
    This function used to visualize the result of demo
    :param imgUrl:
    :param p:
    :param perturbedP:
    :param predictedPerturbedP:
    :return:
    '''
    img = cv2.imread(imgUrl, flags=cv2.IMREAD_COLOR)
    img = np.array(img, np.int32)
    p = np.array([p], dtype=np.int32)
    perturbedP = np.array([perturbedP], dtype=np.int32)
    predictedPerturbedP = np.array([predictedPerturbedP], dtype=np.int32)

    img0 = cv2.polylines(img.copy(), p , True, 255, 3, cv2.LINE_AA)
    img1 = cv2.polylines(img.copy(), perturbedP , True, (100, 125, 255), 3, cv2.LINE_AA)
    img1 = cv2.polylines(img1, predictedPerturbedP , True, (0, 125, 0), 3, cv2.LINE_AA)

    # rendering the perturbed patch and predicted perturbed patch
    cv2.rectangle(img1, (5, 5), (250, 75), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img1, "Predicted Perturbed P", (10, 40), font, 0.5, color=(0, 125, 0), lineType=cv2.LINE_AA)
    cv2.putText(img1, "Perturbed P", (10, 20), font, 0.5, color=(100, 125, 255), lineType=cv2.LINE_AA)

    # calulcating the average corner error
    loss = predictedPerturbedP - perturbedP
    a = np.linalg.norm(loss, axis=-1)
    res = np.mean(a, axis=1)
    cv2.putText(img1, "Average Corner Error: {:0.3f}".format(np.mean(res)), (10, 60), font, 0.5, color=(100, 125, 255), lineType=cv2.LINE_AA)

    cv2.imwrite("origin.png", img0)
    cv2.imwrite("result.png", img1)

    ## demo patch
    p = np.array(p, dtype=np.float32).reshape(4,2)
    position = np.array(p[0], dtype=np.int)
    patchImage = img[position[1]: position[1] + patchSize, position[0]:position[0] + patchSize]

    # ------------------------------
    img = np.array(img, dtype=np.float)
    ##### ------------------------------- ###########
    perturbedP = np.array(perturbedP, dtype=np.float32).reshape(4,2)
    H_AB = cv2.getPerspectiveTransform(p, perturbedP)
    H_BA = np.linalg.inv(H_AB)
    warpedImage = cv2.warpPerspective(img, H_BA, dsize=(img.shape[1], img.shape[0]))
    perturbedPatchImage = warpedImage[position[1]: position[1] + patchSize, position[0]:position[0] + patchSize]


    ## Rewarped -----
    newposition = np.array([0, 0], dtype=np.int)
    src = np.array([
        newposition,
        [newposition[0] + patchSize, newposition[1]],
        [newposition[0] + patchSize, newposition[1] + patchSize],
        [newposition[0], newposition[1] + patchSize]
    ], dtype=np.float32)
    dest = np.array(src + H, dtype=np.float32)
    H_ = cv2.getPerspectiveTransform(src, dest)

    reWarpedPatchImage = np.array(cv2.warpPerspective(perturbedPatchImage, H_, dsize=(patchImage.shape[1], patchImage.shape[0])), dtype=np.uint8)

### --------------------------------
    predictedPerturbedP = np.array(predictedPerturbedP, dtype=np.float32).reshape(4,2)
    predictedH_AB = cv2.getPerspectiveTransform(p, predictedPerturbedP)
    predictedH_BA = np.linalg.inv(predictedH_AB)
    predictedWarpedImage = np.array(cv2.warpPerspective(img, predictedH_BA, dsize=(img.shape[1], img.shape[0])), dtype=np.uint8)
    predictedPerturbedPatchImage = predictedWarpedImage[position[1]: position[1] + patchSize,
                          position[0]:position[0] + patchSize]
    ## Rewarped -----
    src = np.array([
        newposition,
        [newposition[0] + patchSize, newposition[1]],
        [newposition[0] + patchSize, newposition[1] + patchSize],
        [newposition[0], newposition[1] + patchSize]
    ], dtype=np.float32)
    dest = np.array(src + predictedH, dtype=np.float32)
    H_ = cv2.getPerspectiveTransform(src, dest)
    predictedRewarpedPatchImage = np.array(cv2.warpPerspective(predictedPerturbedPatchImage, H_, dsize=(patchImage.shape[1], patchImage.shape[0])), dtype=np.uint8)
    ## ----- Visualize -------------------
    cv2.imwrite("perturbedPatch.png", perturbedPatchImage)
    cv2.imwrite("predicted_PerturbedPatch.png", predictedPerturbedPatchImage)

    cv2.imwrite("patch.png", patchImage)
    cv2.imwrite("rewarped.png", reWarpedPatchImage)
    cv2.imwrite("predictedRewarped.png", predictedRewarpedPatchImage)

def main():
    parser = argparse.ArgumentParser(description='DeepHomography')

    parser.add_argument('--image', type=str, default="example.jpg")
    parser.add_argument('--deviceID', type=int, default=0, metavar='N',
                        help='The GPU ID (default: 0)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for set_1 (default: 64)')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device(args.deviceID)
    else:  # attempt upgrade to Metal acceleration
        use_mps = torch.backends.mps.is_built()
        device = torch.device("mps" if use_mps else "cpu")

    resultModelFile = "pretrained_model"


    # -----------------------------------------------------
    model = DeepHomography(inputSize=128, outputSize=8).to(device)

    if torch.cuda.is_available():
        model.cuda()
    if os.path.isfile(resultModelFile):
        try:
            model.load_state_dict(torch.load(resultModelFile, map_location=device))
        except RuntimeError as e:
            raise RuntimeError(f"Cannot load the saved model:\n{e}")

    demo(model, device, args.image)

if __name__ == "__main__":
    main()
