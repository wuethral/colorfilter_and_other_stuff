import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread('final_mask_with_morph_61.png')

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
coordinates_of_white_pixels = []
rows,cols = img.shape[:2]
''' 
for i in range(rows):
    for j in range(cols):
        if img[i,j] == 255:
            coordinates_of_white_pixels.append([i,j])
#print(range(rows), range(cols))

#print(coordinates_of_white_pixels)

#cv.imshow('Original', img)
#cv.waitKey(0)
'''
#num_labels, labels = cv.connectedComponents(img)
output = cv.connectedComponentsWithStats(img)

num_labels = output[0]
# The second cell is the label matrix
labels = output[1]
# The third cell is the stat matrix
stats = output[2]
# The fourth cell is the centroid matrix
centroids = output[3]
print('num_labels:',num_labels)

print(stats[0][4])
#print(len(labels))
''' 
for pos in labels[2]:
    if pos == 255:
        count += 1
print(count)
'''
'''  
plt.style.use('seaborn-whitegrid')
fig = plt.figure()
plt.xlabel('x_coordinate')
plt.ylabel('y_coordinate')
plt.title('Position of White Pixels')
#ax = fig.gca()
plt.xlim(-10,1080)
plt.ylim(-10,1920)

for i in range(len(coordinates_of_white_pixels)):
    x_cord = coordinates_of_white_pixels[i][0]
    y_coord = coordinates_of_white_pixels[i][1]
    plt.plot(x_cord,y_coord,marker='.', markersize=1,color='red')

plt.savefig('clusters.png')

plt.grid()
plt.show()
'''