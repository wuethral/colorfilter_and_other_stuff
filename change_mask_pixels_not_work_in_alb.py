import cv2 as cv
import os
from PIL import Image, ImageOps

images_list = os.listdir('Testing_Data_With_Augmentation/images')
image_0 = cv.imread('Testing_Data_With_Augmentation/images/zoom_trans_rot_1_zz_test_anglepink1_cleaned.png')
height = image_0[0].shape[0]
width = image_0[0].shape[1]

def check_pixel_pink(pix, y,x):

    if pix[y, x][0] == 250 and pix[y, x][1] == 14 and pix[y, x][2] == 191:
        return True

def check_pixel_black(pix, y,x):

    if pix[y, x][0] == 0 and pix[y, x][1] == 0 and pix[y, x][2] == 0:
        return True

for image_name in images_list:
    image_path = 'Testing_Data_With_Augmentation/images/' + image_name
    image = Image.open(image_path)
    img_pix = image.load()
    mask_path = 'Testing_Data_With_Augmentation/masks/' + image_name[:15] + 'mask_' + image_name[15:]
    mask = Image.open(mask_path)
    #mask = ImageOps.grayscale(mask)
    pix = mask.load()
    print(mask.size)
    black = Image.open('black.png')
    black = ImageOps.grayscale(black)
    pix_black = black.load()

    for x in range(width):
        for y in range(height):
            if check_pixel_pink(img_pix,y,x) and pix[y, x] != 0:
                print('yes')

            #    print('hi')


    save_path =  'Testing_Data_With_Augmentation/new_masks/' + image_name[:-4] + '_mask.png'

    mask.save(save_path)

