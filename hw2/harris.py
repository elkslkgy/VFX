import cv2
import sys
import numpy as np
from scipy import signal

directory = sys.argv[1]
filename = directory + "/readImage.txt"
file = open(filename, "r")
data = file.read().split("\n")

for num in range(len(data)):
    window_size = 4
    k = 0.04
    threshold = 10000000
    Rmax = 0

    name = directory + "/" + data[num]
    img = cv2.imread(name)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = np.float32(gray)
    #這是python裡面自帶的找harris corner的函式
    # newImg = cv2.cornerHarris(gray, 2, 3, 0.04)

    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    x = gray.shape[0]
    y = gray.shape[1]

    result = gray.copy()

    offset = int(window_size/2)

    for i in range(offset, x - offset):
        for j in range(offset, y - offset):
            windowIx2 = Ix2[i - offset:i + offset + 1, j - offset:j + offset + 1]
            windowIy2 = Iy2[i - offset:i + offset + 1, j - offset:j + offset + 1]
            windowIxy = Ixy[i - offset:i + offset + 1, j - offset:j + offset + 1]
            Sx2 = windowIx2.sum()
            Sy2 = windowIy2.sum()
            Sxy = windowIxy.sum()

            det = Sx2 * Sy2 - Sxy ** 2
            trace = Sx2 + Sy2
            R = det - k * (trace ** 2)

            result.itemset((i, j), R)

            if R > Rmax:
                Rmax = R

    for i in range(offset, x - offset):
        for j in range(offset, y - offset):
            if result[i][j] > 0.01*Rmax:
                img.itemset((i, j, 0), 0)
                img.itemset((i, j, 1), 0)
                img.itemset((i, j, 2), 255)

    output = directory + "/dst/" + data[num]
    print(output)
    cv2.imwrite(output, img)


