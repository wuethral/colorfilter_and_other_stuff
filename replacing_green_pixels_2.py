from PIL import Image
import cv2 as cv
import os

def checking_pixel_value(pix, x, hsv_img, height):

    for y in range(20, height - 20):
        if (pix[x, y][0] == 0 and pix[x, y][1] == 0 and pix[x, y][2] == 0) and (pix[x+1, y+1][0] == 0 and pix[x+1, y+1][1] == 0 and pix[x+1, y+1][2] == 0): #or (pix[x, y][0] == 255 and pix[x, y][1] == 192 and pix[x, y][2] == 203):
            if hsv_img[y+1,x][0] > 40 and hsv_img[y+1,x][0] < 150 and hsv_img[y+1,x][1] > 60 and hsv_img[y+1,x][1] < 255 and hsv_img[y+1,x][2] > 0 and hsv_img[y+1,x][2] < 255:
                pix[x, y+1] =(0,0,0)
                break



#directory = 'masks_no_errosion_abstract/screw_random_dataset_17_2.png'
directory = 'yyyyyyyyyyy.png'
img = Image.open(directory)
image = cv.imread(directory)
hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
pix = img.load()

width, height = img.size

for i in range(1):
    print(i)

    for x in range(20, width-20):

        #for y in range(20, height-20):
        checking_pixel_value(pix, x, hsv_img, height)
                    #print(img)
img.save('yyyyyyyyyyy.png')

#Image._show(img)