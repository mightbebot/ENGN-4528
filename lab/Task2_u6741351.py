import numpy as np
import cv2 
import matplotlib.pyplot as plt
import math


%matplotlib inline
###################### task 2 #################################
#task 2.1

lenna = cv2.imread('Lenna.png' , 0)
negative = 255 - lenna ;  
plt.subplot(1,2,1) , plt.imshow(lenna , cmap = 'gray') , plt.title('2.1.1_Original_lenna')
plt.subplot(1,2,2) , plt.imshow(negative , cmap = 'gray') , plt.title('2.1.2_negative_lenna')
plt.show()

# 2.2
flipped = lenna[::-1] 
plt.subplot(1,2,1) , plt.imshow(lenna , cmap = 'gray') , plt.title('2.2.1_Original_lenna')
plt.subplot(1,2,2) , plt.imshow(flipped , cmap = 'gray') , plt.title('2.2.2_flipped_lenna')
plt.show()

# 2.3
# when equating images they were being equated with refrence
lenna_RGB = cv2.imread('Lenna.png')
swap = np.zeros( shape = lenna_RGB.shape , dtype = np.uint8)


swap[: , : , 0] = lenna_RGB[: , : , 2]
swap[: , : , 1] = lenna_RGB[: , : , 1]
swap[: , : , 2] = lenna_RGB[: , :, 0]

cv2.imshow('task2.3.1_original_lenna' , lenna_RGB)
cv2.imshow('task2.3.2_swapped_lenna' , swap)
cv2.waitKey(0)

# 2.4
average = np.add(lenna , flipped) / 2 ; 

plt.imshow( np.uint8(average) , cmap = 'gray') , plt.title('2.4_ averaged')
plt.show()

# 2.5
noise = np.random.randint( low = 0 , high = 255 , size = (512,512) , dtype = np.uint8)
lenna_noised = np.uint8(np.add(lenna , noise)) ;  
plt.imshow(lenna_noised , cmap = 'gray') , plt.title("2.5_ lenna_noised")
plt.show()
