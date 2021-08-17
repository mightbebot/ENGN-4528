import numpy as np
import cv2 
import matplotlib.pyplot as plt
import math


%matplotlib inline
###################### task 3 #################################
# 3.1 , 3.2
f1 = cv2.imread('face1_u6741351.jpg')
f2 = cv2.imread('face2_u6741351.jpg')
f3 = cv2.imread('face2_u6741351.jpg')

# 3.3
def operation (I):
    i = cv2.resize(I , ( 768 ,512))
    
    ir = i[: , : , 0]    # extracting red channel
    ig = i[: , : , 1]    # extracting green channel
    ib = i[: , : , 2]   # extracting blue channel
   
    # calculating histogram
    hr = cv2.calcHist(ir , [0] , None , [256] , [0,256]) 
    hg = cv2.calcHist(ig , [0] , None , [256] , [0,256])
    hb = cv2.calcHist(ib , [0] , None , [256] , [0,256])
    
    plt.plot(hr) , plt.title('3.1_red-channel_histgrm') , plt.show()
    plt.plot(hg) , plt.title('3.2_green-channel_histgrm') , plt.show()
    plt.plot(hb) , plt.title('3.3_blue-channel_histgrm') , plt.show()
    
    # applying histogram equalization on individual channels
    rhe = cv2.equalizeHist(ir)
    ghe = cv2.equalizeHist(ig)
    bhe = cv2.equalizeHist(ib)
   
    # merging histogram equalization of individual channels into a RGB image
    ihe = np.empty(shape = i.shape , dtype = np.uint8 )
    ihe[: , : , 2] = rhe
    ihe[: , : , 1] = ghe
    ihe[: , : , 0] = bhe
    
    
    plt.imshow( rhe , cmap = 'gray') ; plt.title('3.4_hist_eq on red channel') ; plt.show()
    plt.imshow( ghe, cmap = 'gray') ;  plt.title('3.5_hist_eq on green channel');plt.show()
    plt.imshow( bhe , cmap = 'gray') ;  plt.title('3.6_hist_eq on blue channel'); plt.show()
    plt.imshow(ihe )  ;  plt.title('3.7_hist_eq on RGB image') ; plt.show()
   # cv2.imshow('img',ihe) ; cv2.waitKey(0)
    
    
operation(f2)
