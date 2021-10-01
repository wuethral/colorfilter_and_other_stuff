import cv2
import os
import numpy as np
from PIL import Image


video=cv2.VideoCapture('D:/masks/spire1(video23)_2/changing_green_screen/lord_ring_battle.mp4')

#video=cv2.VideoCapture('D:/masks/angle(video20)/angle(20).avi')


def create_new_mask(frame, mask):

    #res_in_method = np.zeros(921600).reshape(480, 640, 3)
    #cv2.imshow('resinmethod', res_in_method)
    res_in_method = frame
    cv2.imwrite('z4.png', frame)
    cv2.imwrite('z5.png', mask)
    for i in range(480):
        for j in range(640):
            if mask[i,j][0] == 0 and mask[i,j][1] == 0 and mask[i,j][2] == 0:
                res_in_method[i,j][0] = 0
                res_in_method[i, j][1] = 0
                res_in_method[i, j][2] = 0
    cv2.imwrite('blabla.png', res_in_method)

            #if mask[i,j][0] == 0 and mask[i,j][1] == 0 and mask[i,j][2] == 0:
          #  res_in_method[i,j] = frame[i,j]


    #cv2.imshow('framiframe', frame)

    ''' 
    for i in range(480):
        for j in range(640):
            print(frame[i,j])
            if mask_matrix[i,j].any == 0:
                print('asdf')
            else:
    '''

    return res_in_method


''' 

background = cv2.imread('green_screen_stuff/nazghul.jpg')
background = cv2.resize(background,(640,480))
image = cv2.imread('D:/masks/spire1(video23)_2/original_images_renamed/image_1.png')
image = cv2.resize(image,(640,480))
image_clone = image.copy()
mask = cv2.imread('D:/masks/spire1(video23)_2/hsv_dbscan_morph_holes_filled(no_canny)/mask_hsv_dbscan_morph_holels_filled1.png')
mask = cv2.resize(mask,(640,480))

res = create_new_mask(image, mask)
green_screen=np.where(res==0, background, res)
cv2.imwrite('green.png', green_screen)
'''
img_nr = 1

while True:

    ret, background = video.read()
    background = cv2.resize(background, (640,480))
    image_path = 'D:/masks/spire1(video23)_2/original_images_renamed/image_' + str(img_nr) + '.png'
    image = cv2.imread(image_path)
    image=cv2.resize(image,(640,480))
    hsv=cv2.cvtColor(background, cv2.COLOR_BGR2HSV)

    mask_path = 'D:/masks/spire1(video23)_2/hsv_dbscan_morph_holes_filled(no_canny)/mask_hsv_dbscan_morph_holels_filled' + str(img_nr) + '.png'
    mask = cv2.imread(mask_path)
    mask=cv2.resize(mask,(640,480))
    
    #mask_matrix = np.zeros((480, 640))
    #for i in range(480):
    #    for j in range(640):
    #        if mask[i,j].any == 0:
    #            mask_matrix[i, j] = 0
    #        else:
    #            mask_matrix[i, j] = 255
    
    #for i in range(1000):
    #    for j in range(1000):
    #        print(mask[i][j])
    #print(frame.shape)
    #print(mask.shape)
    res = create_new_mask(image, mask)

    green_screen=np.where(res==0, background, res)

    cv2.imshow('Mask',mask)
    cv2.imshow('Res',res)
    cv2.imshow('Frame', background)
    cv2.imshow('Green Screen', green_screen)
    k=cv2.waitKey(1000)
    if k==ord('q'):
        break
    img_nr += 1
video.release()
cv2.destroyAllWindows()


