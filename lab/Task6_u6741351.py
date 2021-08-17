import numpy as np
import cv2 
import matplotlib.pyplot as plt
import math


%matplotlib inline
###################### task 6 #################################
f1 = cv2.imread('face1_u6741351.jpg')
f2 = cv2.imread('face2_u6741351.jpg')
f3 = cv2.imread('face2_u6741351.jpg')

# implementing backward mapping
def backward_mapping ( I , degree):
    image = cv2.resize(I , ( 512 ,512))
    image = cv2.cvtColor(image , cv2.COLOR_RGB2GRAY)

     
    radians = math.radians(degree)

    # defining rotational matrix 
    r00 = math.cos(radians)
    r01 = -math.sin(radians)
    r10 = math.sin(radians)
    r11 = math.cos(radians)

    output = np.ones(shape = image.shape , dtype = np.uint8 )

    row , col = image.shape
    x = row/2
    y = col/2

    for i in range(row):
        for j in range(col):
            xo = int(r00*i + r01*j + x - r00*x - r01*y)
            yo = int(r10*i + r11*j + y - r10*x - r11*y)
            if (xo > 0 and xo < row) and (yo > 0  and yo < col):
                output[i , j] = image[xo , yo]
    
    return output
f1 = cv2.imread('face1_u6741351.jpg')
f2 = cv2.imread('face2_u6741351.jpg')
f3 = cv2.imread('face2_u6741351.jpg')

result1 = backward_mapping(f2 , -90)
result2 = backward_mapping(f2 , -45)
result3 = backward_mapping(f2 , -15)
result4 = backward_mapping(f2 , 45)
result5 = backward_mapping(f2 , 90)


plt.imshow(result1 , cmap = 'gray') ; plt.title('rotated_at_ -90') ; plt.show()
plt.imshow(result2 , cmap = 'gray') ; plt.title('rotated_at_ -45') ; plt.show()
plt.imshow(result3 , cmap = 'gray') ; plt.title('rotated_at_ -15') ; plt.show()
plt.imshow(result4 , cmap = 'gray') ; plt.title('rotated_at_ 45') ; plt.show()
plt.imshow(result5 , cmap = 'gray') ; plt.title('rotated_at_ 90') ; plt.show()

# implementing forward mapping
def forward_mapping ( I , degree):
    image = cv2.resize(I , ( 512 ,512))
    image = cv2.cvtColor(image , cv2.COLOR_RGB2GRAY)

     
    radians = math.radians(-degree)

    # defining rotational matrix 
    r00 = math.cos(radians)
    r01 = -math.sin(radians)
    r10 = math.sin(radians)
    r11 = math.cos(radians)

    output = np.ones(shape = image.shape , dtype = np.uint8 )

    row , col = image.shape
    x = row/2
    y = col/2

    for i in range(row):
        for j in range(col):
            xo = int(r00*i + r01*j + x - r00*x - r01*y)
            yo = int(r10*i + r11*j + y - r10*x - r11*y)
            if (xo > 0 and xo < row) and (yo > 0  and yo < col):
                output[xo , yo] = image[i , j]
    return output



# compairing forward and backward mapping outputs
resultf = forward_mapping(f2 , -15)

resultb = backward_mapping(f2 , -15)

plt.imshow(resultf , cmap = 'gray') ; plt.title('forward_mapped_at -15') ; plt.show()
plt.imshow(resultb , cmap = 'gray') ; plt.title('backward_mapped_at -90') ; plt.show()
