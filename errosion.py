import cv2 as cv
import os
import numpy as np
directory = 'PennFudanPed/PedMasks'
for filename in os.listdir(directory):
    source_path = directory + '/' + filename
    mask = cv.imread(source_path)

    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 5), np.uint8)

    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    # img_erosion = cv.erode(img, kernel, iterations=1)
    img_dilation = cv.erode(mask, kernel, iterations=1)
    destination_path = 'erroded_masks/' + filename [:-4] + '_erroded.png'

    cv.imwrite(destination_path,
               img_dilation)

    # cv.imshow('Input', img)
    # cv.imshow('Erosion', img_erosion)
