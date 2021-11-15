from PIL import Image

image = Image.open('masks_no_errosion_abstract/angle_random_dataset_2_2.png')
pix = image.load()
width = image.size[0]
height = image.size[1]

def check_black(pix, x, y):
    if pix[x, y][0] == 0 and pix[x, y][1] == 0 and pix[x,y][2] == 0:
        return True

def turn_pink(pix, x, y):
    pix[x, y] = (250, 14, 191)

def turn_black(pix, x, y):
    pix[x, y] = (0, 0, 0)

for x in range(30):
    for y in range(30):
        turn_black(pix, x, y)

for x in range(width):
    for y in range(height):
        if check_black(pix, x, y):
            turn_pink(pix, x, y)

image.save('zzzz.RAW')