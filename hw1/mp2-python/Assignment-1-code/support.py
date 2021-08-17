# %%
from tqdm import tqdm
import os, sys, numpy as np, cv2
from scipy import signal
from skimage.util import img_as_float
from skimage.io import imread
import matplotlib
from scipy.ndimage import filters
import matplotlib.pyplot as plt

### erosion function
def erosion(img, conv_filter):
    f_siz_1, f_size_2 = conv_filter.shape
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(1, 1))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = np.logical_and(curr_region , conv_filter)
            conv_sum = np.logical_and.reduce(curr_result,axis=(0,1))  # logical and reduction
            conv_sum = 1 if conv_sum==True else 0
            result[r, c] = conv_sum  # Saving reduction

    return result

a = np.array([[1,1,1,0,1,1,1,1,1,0],
              [1,1,1,0,1,1,1,1,1,0],
              (np.ones(10)),
              (np.ones(10)),
              (np.ones(10)) ])
b = np.ones((3,3))
aeb = erosion(a,b)
##### compairing with inbuilt erode function
aeb == cv2.erode(a,b)
###############################
## set difference:
diff = a-aeb


#### function for non maximal seperation:
def nms(img, angle):
    Z = np.zeros(img.shape, dtype=np.uint8)
    for i in range(1,img.shape[0]-1):
        for j in range(img.shape[1]-1):
            q = 0
            r = 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = img[i, j+1]
                r = img[i, j-1]
            elif (22.5 <= angle[i,j] < 67.5):
                q = img[i+1, j-1]
                r = img[i-1, j+1]
            elif (67.5 <= angle[i,j] < 112.5):
                q = img[i+1, j]
                r = img[i-1, j]
            elif (112.5 <= angle[i,j] < 157.5):
                q = img[i-1, j-1]
                r = img[i+1, j+1]
            if (img[i,j] >= q) and (img[i,j] >= r):
                Z[i,j] = img[i,j]
            else:
                Z[i,j] = 0

## function from contour_demo.py that uses non-maximal suppression
def compute_edges_dxdy(I):
    """Returns the norm of dx and dy as the edge response function."""
    I = I.astype(np.float32) / 255.
    I = cv2.GaussianBlur(I,(5,5),0)
    dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same',boundary='symm')
    dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same',boundary='symm')
    mag = np.sqrt(dx ** 2 + dy ** 2)
    mag = mag / np.max(mag)
    mag = mag * 255.
    mag = np.clip(mag, 0, 255)
    mag = mag.astype(np.uint8)
    angle = np.rad2deg(np.arctan2(dy,dx))
    angle[angle<0]+=180
    new_mag = nms(mag,angle) # performing non maximal suppression
    return new_mag

