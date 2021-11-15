from PIL import Image
import cv2 as cv
import os

def checking_pixel_value(pix, x, y):

    s = 20
    a = 20
    if pix[x,y][0] != 0 or pix[x,y][1] != 0 or pix[x,y][2] != 0:
        if pix[x-s,y][0] == 0 and pix[x-s,y][1] == 0 and pix[x-s,y][2] == 0:
            #pix[x,y] = (0,0,0)
            pix[x,y] = pix[x+a,y]
        if pix[x+s,y][0] == 0 and pix[x+s,y][1] == 0 and pix[x+s,y][2] == 0:
            #pix[x,y] = (0,0,0)
            pix[x,y] = pix[x-a,y]
        if pix[x,y-s][0] == 0 and pix[x,y-s][1] == 0 and pix[x,y-s][2] == 0:
            #pix[x,y] = (0,0,0)
            pix[x,y] = pix[x,y+a]
        if pix[x,y+s][0] == 0 and pix[x,y+s][1] == 0 and pix[x,y+s][2] == 0:
            #pix[x,y] = (0,0,0)
            pix[x,y] = pix[x,y-a]


    #pix[x, y] = (0, 0, 0)





for image_name in os.listdir('masks_no_errosion_abstract'):

    directory = 'masks_no_errosion_abstract/' + image_name
    img = Image.open(directory)
    image = cv.imread(directory)
    hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    pix = img.load()
    width, height = img.size

    for x in range(20, width-20):
        for y in range(20, height-20):
            if hsv_img[y,x][0] > 40 and hsv_img[y,x][0] < 80 and hsv_img[y,x][1] > 120 and hsv_img[y,x][1] < 255 and hsv_img[y,x][2] > 0 and hsv_img[y,x][2] < 255:
                checking_pixel_value(pix, x, y)
                #print(img)
    source_path = 'replacing_green_pix_with_neighb/' + image_name
    img.save(source_path)
    #Image._show(img)