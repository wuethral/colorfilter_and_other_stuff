import cv2 as cv
import os
import numpy as np
directory = 'pliers_masks_seperate_source'
for filename in os.listdir(directory):
    source_path = directory + '/' + filename
    mask = cv.imread(source_path,0)

    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 5), np.uint8)

    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    # img_erosion = cv.erode(img, kernel, iterations=1)
    img_dilation = cv.dilate(mask, kernel, iterations=1)

    ''' 
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv.drawContours(drawing, contours, i, (255,255,255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    img_eroded = cv.erode(out, kernel, iterations=1)
    destination_path = 'pliers_masks_seperate_dest/' + filename [:-4] + '_dilated.png'
    '''
    destination_path = 'pliers_masks_seperate_dest/' + filename [:-4] + '_dilated.png'


    cv.imwrite(destination_path,
               img_dilation)


