import numpy as np
import pickle
import cv2
from shapely.geometry import Polygon
text = "/media/slark/Data_New/DeepHomography/SIFT_VAL/SIFT_Sum.txt"
patchSize = 127
with open(text, "rb") as f:
    data = pickle.load(f)

Hs = data["H"]
sources = data["pos"]
err = data["err"]
orgImgs = data["img"]
matchesNum = data["matches"]
image = cv2.imread(orgImgs[0], flags=cv2.IMREAD_GRAYSCALE)
dests = sources


source = np.array(sources[0], dtype=np.float32)
pos = np.array(source[0], dtype=np.int)
dst = np.array(sources[0] + Hs[0], dtype=np.float32)

poly1 = Polygon(source)
poly2 = Polygon(dst)

print(poly1.intersection(poly2).area/poly1.area)


H_AB = cv2.getPerspectiveTransform(source, dst)
H_BA = np.linalg.inv(H_AB)
appliedImg = np.array(cv2.warpPerspective(image, H_BA, dsize=(image.shape[1], image.shape[0])), dtype=np.uint8)
croppedImg = np.array(image[pos[1]: pos[1] + patchSize, pos[0]:pos[0] + patchSize], dtype=np.uint8)
croppedAppliedImage = np.array(appliedImg[pos[1]: pos[1] + patchSize, pos[0]:pos[0] + patchSize], dtype=np.uint8)

position = np.array([0,0], dtype=np.int)
src = np.array([
         position
         ,
        [position[0] + patchSize, position[1]]
        ,
        [position[0] + patchSize, position[1] + patchSize]
            ,
    [position[0], position[1] + patchSize]
], dtype=np.float32)

dest = np.array(src + Hs[0], dtype=np.float32)
H_= cv2.getPerspectiveTransform(src, dest)
rAppliedImage = np.array(cv2.warpPerspective(croppedAppliedImage, H_, dsize=(croppedImg.shape[1], croppedImg.shape[0])), dtype=np.uint8)

appliedImg = cv2.polylines(appliedImg.copy(), [np.int32(source)], True, 255, 1, cv2.LINE_AA)
image = cv2.polylines(image.copy(), [np.int32(source)], True, 255, 1, cv2.LINE_AA)
cv2.imwrite("appliedImg.png", appliedImg)
cv2.imwrite("croppedAppliedImage.png", croppedAppliedImage)
cv2.imwrite("croppedImg.png", croppedImg)
cv2.imwrite("imag.png", image)
cv2.imwrite("rAppliedImage.png", rAppliedImage)
cv2.imwrite("result.png", rAppliedImage+croppedImg)


