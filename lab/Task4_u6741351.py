import numpy as np
import cv2 
import matplotlib.pyplot as plt
import math


%matplotlib inline
f1 = cv2.imread('face1_u6741351.jpg')
f2 = cv2.imread('face2_u6741351.jpg')
f3 = cv2.imread('face2_u6741351.jpg')
###################### task 4 #################################
#4.1

face_f1 = f1[50:525 , 115:665] ; # cropping central facial region
face_f1 = cv2.resize(face_f1 , ( 256 ,256)) ; # resizing image

gray_face_f1 = cv2.cvtColor(face_f1 , cv2.COLOR_RGB2GRAY) ; # converting from RGB to grayscale

plt.imshow(gray_face_f1 , cmap = 'gray') ; plt.title('4.1_cropped_face')
plt.show()

#4.2
noise =  15 * np.random.randn(256,256) + 0   # gaussian noise of sigma = 15 and mean = 0
Gnoise = np.add(gray_face_f1 , noise);       # addin noise to image 
Gnoise = Gnoise.astype(np.uint8) ; plt.imshow( Gnoise , cmap = 'gray') ; plt.title('4.2_cropped_face + noise')
plt.show()

#4.3

# calculating histograms
hstG = cv2.calcHist(gray_face_f1 , [0] , None , [256] , [0,256])
hstnoise =  cv2.calcHist(Gnoise , [0] , None , [256] , [0,256])

plt.subplot(1,2,1) ; plt.plot(hstG) ; plt.title('4.3_histogram_before_noise');
plt.subplot(1,2,2) ; plt.plot(hstnoise) ; plt.title('4.4_histogram_after_noise');
plt.show()


#4.4
def convul(img , kernel ):
    
    img_rows , img_cols = img.shape
    k_rows , k_cols = kernel.shape
    
    
    # initializing an empty image for output
    I = np.empty(shape = img.shape , dtype = np.uint8 )
    
    
    for i in range(img_rows - k_rows):
        for j in range(img_cols -k_cols):
            slc = img[i:i+k_rows , j:j+k_cols] # mapping kernel over image
             # performing convolution of mapped image area with kernel
            tmp = np.multiply(slc , kernel)   
            s = np.sum(tmp)
            I[i+int((k_rows)/2) , j+int((k_cols)/2)] = s
    
    
    return I

# generating 5*5 guassian kernel
ker = np.empty((5,5))
row , col = ker.shape
sig = 1.
for i in range(row):
    for j in range(col):
        y = i - 2
        x = j - 2
        ker[i , j] = (1 / (2*math.pi*math.pow(sig , 2))) * math.exp(-(x*x + y*y)/2*sig*sig)

# applying our kernel over the noised image
I = convul(Gnoise , ker)           

plt.imshow(I , cmap = 'gray') ; plt.title ('4.5_convolution_by_custom_function') ; plt.show()

# comparing with inbuilt python function
I2 = cv2.GaussianBlur(Gnoise , (5,5) , 1)
plt.imshow(I2 , cmap = 'gray') ; plt.title('4.6_convolution_by_inbuilt_function') ; plt.show()
