# %%
import glob
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import cv2

### 3.2.1 Reading images
trains = [plt.imread(file) for file in glob.glob("Yale-FaceA/trainingset/*.png")]
tests = [plt.imread(file) for file in glob.glob("Yale-FaceA/testset/*.png")]
####


# creating a training images tensor
train = []
for i in range(len(trains)):
    f = trains[i].reshape(-1,1)
    train.append(f)
train = np.asarray(train)
train = train.reshape(135,45045) # list of 135 images
train = train.T       # 135 columns -> images

# creating a test images tensor
test = []
for i in range(len(tests)):
    f = tests[i].reshape(-1,1)
    test.append(f)
test = np.asarray(test)
test = test.reshape(10,45045)
test = test.T

# mean face of train images
mean = (np.sum(train,axis=1) / train.shape[1])


# normalizing train_images
n_train = np.zeros_like(train)
for i in range(train.shape[1]):
    n_train[:,i] = train[:,i] - mean 

# normalizing test images
n_test = np.zeros_like(test)
for i in range(10):
    n_test[:,i] = test[:,i] - mean


##### 3.2.2 performing PCA First using standard and then using SVD method

# ................... uncomment block 1 for standard pca method
# Standard method for calculating eigen faces
# ..............(block 1) .............................................. 
# cov_matrix = np.dot(n_train.T , n_train)
# eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
# eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]
# eig_pairs.sort(reverse=True)
# eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenvalues))]
# eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]
# reduced_data = np.array(eigvectors_sort[:10]).transpose()
# eigen_faces = np.dot(train,reduced_data)
#...................................... block 1 ends ..................



# calculating eigen faces using SVD
# ....................(block 2)...................................
U, Sigma, VT = np.linalg.svd(n_train, full_matrices=False)
eigen_faces = U[:,:10]
# .....................................block 2 ends..............


# ...........for standard method uncomment block 1 and comment block 2
# ...........for svd uncomment block 2 and comment block 1



# projection of 135 training-images on eigen faces
train_weights = np.zeros((eigen_faces.shape[1],135))
temp = np.zeros(eigen_faces.shape[1]) # storing weight array of size 10(eigen) of each image
for i in range(135):
    for j in range(eigen_faces.shape[1]):
        temp[j] = np.dot(eigen_faces[:,j].reshape(1,-1) , n_train[:,i].reshape(-1,1))
    train_weights[:,i] = temp

# projection of 10 test_images on eigen faces
test_weights = np.zeros((eigen_faces.shape[1],10))
temp = np.zeros(10)
for i in range(10):
    for j in range(10):
        temp[j] = np.dot(eigen_faces[:,j].reshape(1,-1) , n_test[:,i].reshape(-1,1))
    test_weights[:,i] = temp
    

# a short function for calculatin distance b/w vectors
# .........................
def distance(a,b):
    s = a-b
    s = np.power(s,2)
#     print(s.shape)
    s = np.sum(s)
    return np.sqrt(s)
#.........................

# ............ calculating top 3 closest images to tests , from training...................
# traget is a list, containing 10 array of shape (3,), denoting top 3 indixes of images from train_set--
#                                                                 -- that are closer to test images
target = []
for i in range(test_weights.shape[1]):
    dist = np.zeros(train_weights.shape[1])
    for j in range(train_weights.shape[1]):
        dist[j] = distance(train_weights[:,j],test_weights[:,i])
    c_ix = dist.argsort()[:3] # indices for top 3 matches
    # targets for 10 test images
    target.append(c_ix)
#..........................................................................................

############# task 3.3.6 ###############################################################
############## leave rest behind #################################
############# only 'trains' variable is neede from previous task ##################

my_trains = [plt.imread(file) for file in glob.glob("my_data/train/*.png")]
my_trains = trains + my_trains

# new training set

train = []
for i in range(len(my_trains)):
    f = my_trains[i].reshape(-1,1)
    train.append(f)
train = np.asarray(train)

train = train.reshape(144,45045) 
train = train.T

# my_test test image
test = plt.imread("my_data/test/0.png")
test = np.array(test).reshape(-1,1)


# mean face of train images
mean = (np.sum(train,axis=1) / train.shape[1])

# normalizing train_images
n_train = np.zeros_like(train)
for i in range(train.shape[1]):
    n_train[:,i] = train[:,i] - mean 

# normalizing test image
n_test = test - mean.reshape(-1,1)

# calculating eigen_faces
cov_matrix = np.dot(n_train.T , n_train)
eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]
eig_pairs.sort(reverse=True)
eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenvalues))]
eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]
reduced_data = np.array(eigvectors_sort[:10]).transpose()
eigen_faces = np.dot(train,reduced_data)

# projection of 144 training-images on eigen faces
train_weights = np.zeros((eigen_faces.shape[1],144))
temp = np.zeros(eigen_faces.shape[1]) # storing weight array of size 10(eigen) of each image
for i in range(144):
    for j in range(eigen_faces.shape[1]):
        temp[j] = np.dot(eigen_faces[:,j] , n_train[:,i])
    train_weights[:,i] = temp


# projection of test image on eigenfaces
test_weights = np.zeros((eigen_faces.shape[1]))
for i in range(eigen_faces.shape[1]):
    test_weights[i] = np.dot(eigen_faces[:,i],n_test)



#.................. c_ix stores index of top 3 closest images from new_train_set (144) to my test image
dist = np.zeros(train_weights.shape[1])
for i in range(train_weights.shape[1]):
    dist[i] = distance(train_weights[:,i],test_weights)
c_ix = dist.argsort()[:3]