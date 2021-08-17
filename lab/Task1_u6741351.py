import numpy as np
import cv2 
import matplotlib.pyplot as plt
import math


%matplotlib inline
###################### task 1 #################################
a = np.array([[2,4,5],[5,2,200]])
b = a[0 , :]
f = np.random.randn(500 , 1)
g =f[f<0]
x = np.zeros(100) + 0.35
y = 0.6 * np.ones((1,len(x)))
z = x-y
A = np.linspace(1,200)
B = A[::-1]   # from reverse in a step of 1
B[B<=50] = 0 
