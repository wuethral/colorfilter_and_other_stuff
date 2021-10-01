from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
import cv2 as cv
import numpy as np
# define dataset
#X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

img = cv.imread('final_mask_with_morph_117.png')

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
coordinates_of_white_pixels = []
rows,cols = img.shape[:2]

for i in range(rows):
	for j in range(cols):
		if img[i,j] == 255:
			coordinates_of_white_pixels.append([i,j])
X = np.asarray(coordinates_of_white_pixels)

#print(coordinates_of_white_pixels)
# define the model
#print(X)
model = DBSCAN(eps=2, min_samples=9)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)

# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()

size_of_biggest_cluster = 0
index_of_biggest_cluster = 0
for cluster in clusters:
	row_ix = where(yhat == cluster)
	if row_ix[0].size > size_of_biggest_cluster:
		if max(X[row_ix, 0][0]) == 1079 or max(X[row_ix, 1][0]) == 1919 or min(X[row_ix, 0][0]) == 0 or min(X[row_ix, 1][0]) == 0:
			continue
		else:
			size_of_biggest_cluster = row_ix[0].size
			index_of_biggest_cluster = cluster
for cluster in clusters:
	if cluster == index_of_biggest_cluster:
		continue

	else:
		row_ix = where(yhat == cluster)
		x_coord_to_delete_mask = X[row_ix, 0]
		y_coord_to_delete_mask = X[row_ix, 1]
		for i in range(len(x_coord_to_delete_mask[0])):
			img[x_coord_to_delete_mask[0][i], y_coord_to_delete_mask[0][i]] = 0
cv.imwrite('new_image.png', img)


