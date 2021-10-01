from PIL import Image

img = Image.open('mask_hsv_canny_dbscan_holes_filled_2'
                 '.png')
pix = img.load()
count = 0

for i in range(img.height):
    for j in range(img.width):
        if (pix[j,i]) == 255:
            print(pix[j,i])
        #if (pix[j,i]) != 0:
        #    print(pix[j,i])

#print(pix[1079, 1919])
#print(pix[0, 0])



