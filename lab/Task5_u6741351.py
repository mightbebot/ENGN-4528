import numpy as np
import cv2 
import matplotlib.pyplot as plt
import math


%matplotlib inline
###################### task 5 #################################
f1 = cv2.imread('face1_u6741351.jpg')
f2 = cv2.imread('face2_u6741351.jpg')
f3 = cv2.imread('face2_u6741351.jpg')

def sobel(img , kernel ):
    
    img_rows , img_cols = img.shape
    
    k_rows , k_cols = kernel.shape
   
    
    I = np.empty(shape = img.shape , dtype = np.float32 )
    
    
    for i in range(img_rows - k_rows):
        for j in range(img_cols -k_cols):
            slc = img[i:i+k_rows , j:j+k_cols] # mapping kernel over image
            # performing convolution of mapped image area with kernel
            tmp = np.multiply(slc , kernel)    
            s = np.sum(tmp)
            I[i+int((k_rows)/2) , j+int((k_cols)/2)] = s
    
    
    return I

# defining x and y gradient filters
sobelx = np.array([[-1 , 0 , 1] , [-2 , 0 , 2] , [-1,0,1]])
sobely = np.array([[1 , 2 , 1] , [0 , 0 , 0] , [-1,-2,-1]])

Input_image = cv2.cvtColor(f1 , cv2.COLOR_RGB2GRAY) ;

Gx = sobel(Input_image , sobelx)
Gy = sobel(Input_image , sobely)

# approximating x and y gradients
G_approx = np.sqrt(np.add(np.multiply(Gx , Gx) , np.multiply(Gy , Gy)))

plt.imshow(Input_image , cmap = 'gray') ; plt.title('5.1_input_image')

plt.imshow(np.uint8(G_approx) , cmap = 'gray') ; plt.title('5.2_custom_sobel_on_input_image')
plt.show()

# Implementation 2 
# before passing image to sobel we will apply gaussian blur to input_image
Input_image_with_gauss_blr = cv2.GaussianBlur(Input_image , (5,5) , 1.5)

gx = sobel(Input_image_with_gauss_blr , sobelx)
gy = sobel(Input_image_with_gauss_blr , sobely)
g = np.sqrt(np.add(np.multiply(gx , gx) , np.multiply(gy , gy)))
plt.imshow(np.uint8(g) , cmap = 'gray') ; plt.title('5.3_custom_sobel_on_Input_image with gaussian blr')
plt.show()

# using inbuilt function to check our performance

gradx = cv2.Sobel(Input_image , cv2.CV_32F , 1 , 0 , ksize = 3)
grady = cv2.Sobel(Input_image , cv2.CV_32F , 0 , 1 , ksize = 3)

g_approx = np.sqrt(np.add(np.multiply(gradx , gradx) , np.multiply(grady , grady)))


plt.imshow(np.uint8(g_approx) , cmap = 'gray') ; plt.title('5.4_inbuilt_sobel_on_input_image') ; plt.show()
plt.imshow(np.uint8(G_approx) , cmap = 'gray') ; plt.title('5.2_custom_sobel_on_input_image') ; plt.show()
