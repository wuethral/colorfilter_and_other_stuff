from PIL import Image, ImageOps

#img = Image.open('D:/trained_models/Testing_Data_No_Augmentation/masks/zz_test_handpink1_cleaned_mask.png')
img = Image.open('Testing_Data_With_Augmentation/images/zoom_trans_rot_1_zz_test_anglepink1_cleaned_mask.png')
gray_image = ImageOps.grayscale(img)

pix = gray_image.load()

count = 0

for i in range(img.height):
    for j in range(img.width):
        #if (pix[j,i]) != 249 and (pix[j,i]) != 0:
        if (pix[j, i]) != 0:
            print(pix[j,i])


#print(pix[1079, 1919])
#print(pix[0, 0])



