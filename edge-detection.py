# -*- coding: utf-8 -*-
"""
LAB5: Edge detection and real life enhancement problem
CSCI 39534
@author: Ramisha Chowdhury
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np

"""
Problem 1: Edge Detection
"""

dog = cv2.imread('C:/Users/rdire/Documents/ImageLAB5/dog.png')
dog = cv2.cvtColor(dog, cv2.COLOR_BGR2GRAY) #Convert to grayscale


#Apply the Roberts operator

#Declare Roberts kernel
Rx = np.array([[1,0],
               [0,-1]])

Ry = np.array([[0,1],
               [-1,0]])

#Initialize with zeros with type double
Roberts_edge = np.zeros_like(dog, dtype=np.float32)

rows, cols = dog.shape
for i in range(rows - 1):
    for j in range(cols -1):
        #Apply Roberts kernel with respect with x
        Gx = (dog[i,j]*Rx[0,0] + dog[i,j+1]*Rx[0,1] + dog[i+1,j]*Rx[1,0] + dog[i+1,j+1]*Rx[1,1])
        #Apply Roberts kernel with respect of y
        Gy = (dog[i,j]*Ry[0,0] + dog[i,j+1]*Ry[0,1] + dog[i+1,j]*Ry[1,0] + dog[i+1,j+1]*Ry[1,1])
        
        #Calculate gradient magnitude
        Roberts_edge[i,j] = np.sqrt(Gx**2 + Gy**2)
        
#Clip to remain [0,255]
Roberts_edge = np.clip(Roberts_edge, 0, 255).astype(np.uint8)

#Apply Perwitt operator

#Declare Perwitt kernel
Px = np.array([[-1,0,1],
               [-1,0,1],
               [-1,0,1]])

Py = np.array([[1,1,1],
               [0,0,0],
               [-1,-1,-1]])

#Initialize with zeros with type double
Perwitt_edge = np.zeros_like(dog, dtype=np.float32)

for i in range(rows - 1):
    for j in range(cols - 1):
        #Apply Perwitt kernel with respect with x
        Gx = (dog[i-1,j-1]*Px[0,0] + dog[i-1,j]*Px[0,1] + dog[i-1,j+1]*Px[0,2]+
              dog[i,j-1]*Px[1,0] + dog[i,j]*Px[1,1] + dog[i,j+1]*Px[1,2] +
              dog[i+1,j-1]*Px[2,0] + dog[i+1,j]*Px[2,1] + dog[i+1,j+1]*Px[2,2])
        #Respect to y
        Gy = (dog[i-1,j-1]*Py[0,0] + dog[i-1,j]*Py[0,1] + dog[i-1,j+1]*Py[0,2]+
              dog[i,j-1]*Py[1,0] + dog[i,j]*Py[1,1] + dog[i,j+1]*Py[1,2] +
              dog[i+1,j-1]*Py[2,0] + dog[i+1,j]*Py[2,1] + dog[i+1,j+1]*Py[2,2])
        
        #Calculate gradient magnitude
        Perwitt_edge[i,j] = np.sqrt(Gx**2 + Gy**2)
        
#Clip to remain [0,255]
Perwitt_edge = np.clip(Perwitt_edge,0,255).astype(np.uint8)


#Apply Sobel kernel

#Declare Sobel kernel
Sx = np.array([[-1,0,1],
               [-2,0,2],
               [-1,0,1]])

Sy = np.array([[1,2,1],
              [0,0,0],
              [-1,-2,-1]])

#Initialize with zeros with type double
Sobel_edge = np.zeros_like(dog, dtype=np.float32)

for i in range(rows - 1):
    for j in range(cols - 1):
        #Apply Sobel kernel with respect to x
        Gx = (dog[i-1,j-1]*Sx[0,0] + dog[i-1,j]*Sx[0,1] + dog[i-1,j+1]*Sx[0,2]+
              dog[i,j-1]*Sx[1,0] + dog[i,j]*Sx[1,1] + dog[i,j+1]*Sx[1,2] +
              dog[i+1,j-1]*Sx[2,0] + dog[i+1,j]*Sx[2,1] + dog[i+1,j+1]*Sx[2,2])
        #Respect to y
        Gy = (dog[i-1,j-1]*Sy[0,0] + dog[i-1,j]*Sy[0,1] + dog[i-1,j+1]*Sy[0,2]+
              dog[i,j-1]*Sy[1,0] + dog[i,j]*Sy[1,1] + dog[i,j+1]*Sy[1,2] +
              dog[i+1,j-1]*Sy[2,0] + dog[i+1,j]*Sy[2,1] + dog[i+1,j+1]*Sy[2,2])
        
        #Calculate gradient magnitude
        Sobel_edge[i,j] = np.sqrt(Gx**2 + Gy**2)
        
#Clip to remain [0,255]
Sobel_edge = np.clip(Sobel_edge,0,255).astype(np.uint8)

#Display 
plt.figure(figsize=(15, 5))

#Original Image
plt.subplot(1, 4, 1)
plt.imshow(dog, cmap='gray')
plt.title('Original Image')
plt.axis('off')

#Roberts Edge Detection
plt.subplot(1, 4, 2)
plt.imshow(Roberts_edge, cmap='gray')
plt.title('Roberts Edge Detection')
plt.axis('off')

#Prewitt Edge Detection
plt.subplot(1, 4, 3)
plt.imshow(Perwitt_edge, cmap='gray')
plt.title('Prewitt Edge Detection')
plt.axis('off')

#Sobel Edge Detection
plt.subplot(1, 4, 4)
plt.imshow(Sobel_edge, cmap='gray')
plt.title('Sobel Edge Detection')
plt.axis('off')

# Show the plot
plt.show()


"""
Problem 2: Removing Snow from Image
"""

snow = cv2.imread('C:/Users/rdire/Documents/ImageLAB5/snow.png')
snow = cv2.cvtColor(snow, cv2.COLOR_BGR2GRAY)

#Apply Sobel
#Function for Sobel Edge Detection
def SobelDetection(image, Sx, Sy):
    rows, cols = image.shape #Get the image dimention
    #Initialize with zeros with type double
    Sobel_edge = np.zeros_like(image, dtype=np.float32)
    
    for i in range(rows - 1):
        for j in range(cols - 1):
            #Apply Sobel kernel with respect to x
            sumX = (image[i-1,j-1]*Sx[0,0] + image[i-1,j]*Sx[0,1] + image[i-1,j+1]*Sx[0,2]+
                  image[i,j-1]*Sx[1,0] + image[i,j]*Sx[1,1] + image[i,j+1]*Sx[1,2] +
                  image[i+1,j-1]*Sx[2,0] + image[i+1,j]*Sx[2,1] + image[i+1,j+1]*Sx[2,2])
            #Respect to y
            sumY = (image[i-1,j-1]*Sy[0,0] + image[i-1,j]*Sy[0,1] + image[i-1,j+1]*Sy[0,2]+
                  image[i,j-1]*Sy[1,0] + image[i,j]*Sy[1,1] + image[i,j+1]*Sy[1,2] +
                  image[i+1,j-1]*Sy[2,0] + image[i+1,j]*Sy[2,1] + image[i+1,j+1]*Sy[2,2])
            
            #Calculate gradient magnitude
            Sobel_edge[i,j] = np.sqrt(sumX**2 + sumY**2)
            
    #Clip to remain [0,255]
    Sobel_edge = np.clip(Sobel_edge,0,255).astype(np.uint8)
    return Sobel_edge

sobel_snow = SobelDetection(snow, Sx, Sy)

#Display
plt.figure()
plt.imshow(sobel_snow, cmap='gray')
plt.title('Sobel Edge Detection on Snow')
plt.axis('off')
plt.show()

#Apply a 5x5 and 7x7 median filter to the grayscale image to reduce snow-related noise
#Function for median filter
def medianFilter(image, kernel_size):
    #Get dimention of image
    rows,cols = image.shape
    #Calculate padding size
    padding_size = kernel_size // 2
    #Create a padded image to handle the borders when applying the median filter
    padded_img = np.pad(image, ((padding_size, padding_size), (padding_size, padding_size)), mode='edge')
    #Initialize the output array with zeros
    m_output = np.zeros((rows, cols))
    #Apply Median filter using nested loops
    for i in range(padding_size, padding_size + rows):
        for j in range(padding_size, padding_size + cols):
            #Extract neighborgood based on padding size
            neighborgood = padded_img[i - padding_size:i+padding_size + 1, 
                                      j - padding_size:j + padding_size + 1]
            #Sort the neighborhood and take the median
            sorted_m_neighborhood = np.sort(neighborgood.flatten())
            median_index = (kernel_size * kernel_size) // 2
            m_output[i-padding_size, j-padding_size] = sorted_m_neighborhood[median_index]
        
    return m_output
    
        
#Apply 5X5 median filter to snow
median_filtered5 = medianFilter(snow, 5)  

#Apply 7X7 median filter to snow
median_filtered7 = medianFilter(snow, 7)

#Display
plt.figure(figsize=(15, 5))

#5X5
plt.subplot(1, 2, 1)
plt.imshow(median_filtered5, cmap='gray')
plt.title('5X5 Median Filter')
plt.axis('off')

#7X7
plt.subplot(1, 2, 2)
plt.imshow(median_filtered7, cmap='gray')
plt.title('7X7 Median Filter')
plt.axis('off')

plt.show()

#Apply Sobel on the median filtered snow
median5_sobel = SobelDetection(median_filtered5, Sx, Sy)
median7_sobel = SobelDetection(median_filtered7, Sx, Sy)  

#Display
plt.figure(figsize=(15, 5))

#5X5
plt.subplot(1, 2, 1)
plt.imshow(median5_sobel, cmap='gray')
plt.title('Sobel on 5X5 Median Filter')
plt.axis('off')

#7X7
plt.subplot(1, 2, 2)
plt.imshow(median7_sobel, cmap='gray')
plt.title('Sobel on 7X7 Median Filter')
plt.axis('off') 

plt.show()   

"""
Problem 3: Removing water droplet
"""
rain = cv2.imread('C:/Users/rdire/Documents/ImageLAB5/rain.png') 
rain = cv2.cvtColor(rain, cv2.COLOR_BGR2GRAY)

#Apply sobel on rain
sobel_rain = SobelDetection(rain, Sx, Sy)

#Display
plt.figure()
plt.imshow(sobel_rain, cmap='gray')
plt.title('Sobel on Rain')
plt.axis('off')
plt.show()

#Apply 5X5 median filter on rain
median_filtered5_rain = medianFilter(rain,5)

#Display
plt.figure()
plt.imshow(median_filtered5_rain, cmap='gray')
plt.title('5X5 Median filter on Rain')
plt.axis('off')
plt.show()

#Apply Gaussian smoothing filter to 5X5 median filtered rain
gaussian_blur5_rain = cv2.GaussianBlur(median_filtered5_rain, (5,5), 1) 

#Display
plt.figure()
plt.imshow(gaussian_blur5_rain, cmap='gray')
plt.title('5X5 Gaussian filter on Rain')
plt.axis('off')
plt.show()

#Apply 7X7 median filter on rain
median_filtered7_rain = medianFilter(rain,7)

#Display
plt.figure()
plt.imshow(median_filtered7_rain, cmap='gray')
plt.title('7X7 Median filter on Rain')
plt.axis('off')
plt.show()

#Apply 7x7 Gaussian smoothing filter to 7X7 median filtered rain
gaussian_blur7_rain = cv2.GaussianBlur(median_filtered7_rain, (7,7),1)

#Display
plt.figure()
plt.imshow(gaussian_blur7_rain, cmap='gray')
plt.title('7X7 Gaussian filter on Rain')
plt.axis('off')
plt.show()

#Apply sobel detection on 5X5, 7X7 Gaussian&Median filtered rain
sobel_rain5 = SobelDetection(gaussian_blur5_rain, Sx, Sy)
sobel_rain7 = SobelDetection(gaussian_blur7_rain, Sx, Sy)

#Display
plt.figure(figsize=(15, 5))

#5X5
plt.subplot(1, 2, 1)
plt.imshow(sobel_rain5, cmap='gray')
plt.title('Sobel on 5X5 rain')
plt.axis('off')

#7X7
plt.subplot(1, 2, 2)
plt.imshow(sobel_rain7, cmap='gray')
plt.title('Sobel on 7X7 rain')
plt.axis('off') 

plt.show() 

#Experiment with unsharp masking to enhance edges
#5X5
mask5 = cv2.subtract(median_filtered5_rain,gaussian_blur5_rain)
k = 1.3
unsharp_rain5 = cv2.add(median_filtered5_rain,(mask5*k))
unsharp_rain5 = unsharp_rain5.astype(np.uint8)

#7x7
mask7 = cv2.subtract(median_filtered7_rain,gaussian_blur7_rain)
unsharp_rain7 = cv2.add(median_filtered7_rain,(mask7*k))
unsharp_rain7 = unsharp_rain7.astype(np.uint8)

#Display
plt.figure(figsize=(15, 5))

#5X5
plt.subplot(1, 2, 1)
plt.imshow(unsharp_rain5, cmap='gray')
plt.title('Unsharp masking 5X5 rain')
plt.axis('off')

#7X7
plt.subplot(1, 2, 2)
plt.imshow(unsharp_rain7, cmap='gray')
plt.title('Unsharp masking 7X7 rain')
plt.axis('off') 

plt.show() 

#Apply sobel on unsharp masking
unsharp_sobel5 = SobelDetection(unsharp_rain5, Sx, Sy)
unsharp_sobel7 = SobelDetection(unsharp_rain7, Sx, Sy)

#Display
plt.figure(figsize=(15, 5))

#5X5
plt.subplot(1, 2, 1)
plt.imshow(unsharp_sobel5, cmap='gray')
plt.title('Unsharp sobel 5X5 rain')
plt.axis('off')

#7X7
plt.subplot(1, 2, 2)
plt.imshow(unsharp_sobel7, cmap='gray')
plt.title('Unsharp sobel 7X7 rain')
plt.axis('off') 

plt.show() 







        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        