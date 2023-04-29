import cv2
import numpy as np
import os
from scipy.optimize import least_squares
import copy
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt

downscale = 5


def common_points(pts1, pts2, pts3):
    '''Here pts1 represent the points image 2 find during 1-2 matching
    and pts2 is the points in image 2 find during matching of 2-3 '''
    indx1 = []
    indx2 = []
    for i in range(pts1.shape[0]):
        a = np.where(pts2 == pts1[i, :])
        if a[0].size == 0:
            pass
        else:
            indx1.append(i)
            indx2.append(a[0][0])

    '''temp_array1 and temp_array2 will which are not common '''
    temp_array1 = np.ma.array(pts2, mask=False)
    temp_array1.mask[indx2] = True
    temp_array1 = temp_array1.compressed()
    temp_array1 = temp_array1.reshape(int(temp_array1.shape[0] / 2), 2)

    temp_array2 = np.ma.array(pts3, mask=False)
    temp_array2.mask[indx2] = True
    temp_array2 = temp_array2.compressed()
    temp_array2 = temp_array2.reshape(int(temp_array2.shape[0] / 2), 2)
    print("Shape New Array", temp_array1.shape, temp_array2.shape)
    return np.array(indx1), np.array(indx2), temp_array1, temp_array2

# Feature detection for two images, followed by feature matching
def find_features(img0, img1):
    img0gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(img0gray, None)
    
    #lk_params = dict(winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    
    kp1, des1 = sift.detectAndCompute(img1gray, None)
    #pts0 = np.float32([m.pt for m in kp0])
    #pts1, st, err = cv2.calcOpticalFlowPyrLK(img0gray, img1gray, pts0, None, **lk_params)
    #pts0 = pts0[st.ravel() == 1]
    #pts1 = pts1[st.ravel() == 1]
    #print(pts0.shape, pts1.shape)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0, des1, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good.append(m)

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])

    return pts0, pts1

# A function, to downscale the image in case SfM pipeline takes time to execute.
def img_downscale(img, downscale):
	downscale = int(downscale/2)
	i = 1
	while(i <= downscale):
		img = cv2.pyrDown(img)
		i = i + 1
	return img


#lire les images 
img_dir ="IMGS/"
img_list = sorted(os.listdir(img_dir))
images = []
for img in img_list:
    if '.jpg' in img.lower() or '.png' in img.lower()  :
        images = images + [img]
i = 0
common_sift_points = []


# Setting the Reference two frames

img0 = img_downscale(cv2.imread(img_dir + '/' + images[i]), downscale)
img1 = img_downscale(cv2.imread(img_dir + '/' + images[i + 1]), downscale)

pts0, pts1 = find_features(img0, img1)

# Here, the total images to be take into consideration can be varied. Ideally, the whole set can be used, or a part of it. For whole lot: use tot_imgs = len(images) - 2
tot_imgs = len(images) - 2 

sift_points = []
for i in tqdm(range(tot_imgs)):
    # Acquire new image to be added to the pipeline and acquire matches with image pair
    img2 = img_downscale(cv2.imread(img_dir + '/' + images[i + 2]), downscale)

    # pts0, pts1 = find_features(img1,img2)

    pts_, pts2 = find_features(img1, img2)
 
    # There will be some common points in pts1 and pts_
    # Find the indices of pts1 that match with pts_
    indx1, indx2, temp1, temp2 = common_points(pts1, pts_, pts2)
    com_pts2 = pts2[indx2]
    com_pts_ = pts_[indx2]
 
    # Store the common SIFT points
    common_sift_points.append(com_pts_)

    # Update variables for the next iteration
    img1 = np.copy(img2)
    pts1 = np.copy(pts2)

    cv2.imshow('image', img2)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()

print(common_sift_points)

centroids = []

for sift_points in common_sift_points:
    x_coords = [pt[0] for pt in sift_points]
    y_coords = [pt[1] for pt in sift_points]
    centroid = (int(np.mean(x_coords)), int(np.mean(y_coords)))
    centroids.append(centroid)

print(centroids)

import numpy as np

# Convert the centroid coordinates to a numpy array
centroids_arr = np.array(centroids)

# Calculate the differences between keypoints and centroids
diffs = []
for i in range(len(common_sift_points)):
    sift_points = common_sift_points[i]
    diff = sift_points - centroids_arr[i]
    diffs.append(diff)

# Reshape the differences into a 2D array
W = np.concatenate(diffs, axis=0)
# Afficher la matrice W
print("Matrice W :")
print(W)

# Calculer la décomposition SVD de la matrice W
U, S, V = np.linalg.svd(W)
# Afficher les résultats
print("Matrice U :")
print(U)
print("Valeurs singulières S :")
print(S)
print("Matrice V :")
print(V)