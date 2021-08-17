# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage import io,color
import cv2

# %%
def pixel_to_vec(im):
#     m = plt.imread('mandm.png')
    im = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
    im = np.array(im)
    vec = []
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            l = float(im[x,y,0])
            a = float(im[x,y,1])
            b = float(im[x,y,2])
            feature = np.array([l,a,b,x,y])
            vec.append(feature)
    return np.asarray(vec)

# function for calculating distance between two vectors
def distance(a,b):
    s = a-b
    s = np.power(s,2)
    s = np.sum(s)
    return np.sqrt(s)

# function for updating seeds/centroid for k-means
def update_seeds(df,target,n):
    n_seeds = np.zeros(shape=(n,df.shape[1]))
    for i in range(n):
        ix,=np.where(target==i)
        s = np.zeros(df.shape[1])
        for j in ix:
            s = s+df[j]
        n_seeds[i] = s/len(ix)
    return n_seeds

def seeds(df,n):
    t = df.shape[0]
    n_points = np.zeros(shape=(n,df.shape[1]))
    dist = np.zeros(shape=(t,t))
    for i in range(df.shape[0]):
        for j in range(df.shape[0]):
            dist[i][j] = distance(df[i],df[j])
    
        


# Kmeans function
def my_kmeans(df,n,itr):
    target = np.zeros(df.shape[0])
    n_points = np.zeros(shape=(n,df.shape[1]))
    # initial seeds
    for i in range(n):
        n_points[i] = df[np.random.randint(low=0,high=df.shape[0])]
    for epoch in range(itr):
        for i in range(df.shape[0]):
            dist = np.zeros(n)
            for j in range(n):
                dist[j] = distance(df[i],n_points[j])
            c_ix,=np.where(dist==np.min(dist))
            c_ix = c_ix[0]
            target[i] = c_ix
         # updating seeds
        n_points = update_seeds(df,target,n)
    return target



mm = plt.imread('mandm.png')
vectors = pixel_to_vec(mm)

#........................... with x,y cordinates (uncomment this block and comment next block)
# n = 3
# target = my_kmeans(df=vectors,n=n,itr=10)
# m = np.array(mm)

# for i in range(n):
#     ix, = np.where(target==i)
#     for j in ix:
#         row = int(vectors[j][3])
#         col = int(vectors[j][4])
# #         m[row,col,:] = 0
#         m[row,col,(2-i)] = 255
# plt.imshow(m)
#.................................................



#........................... without x,y cordinates
n = 3
target = my_kmeans(df=vectors[:,:3],n=n,itr=10)
m = np.array(mm)
test = np.zeros_like(m)
for i in range(n):
    ix, = np.where(target==i)
    for j in ix:
        row = int(vectors[j][3])
        col = int(vectors[j][4])
        test[row,col,:] = 0
        test[row,col,(2-i)] = 255
plt.imshow(test)
#......................................................
