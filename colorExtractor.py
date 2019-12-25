# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:06:45 2019

@author: cse.repon
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
from matplotlib import pyplot as plt
import colorsys


def extractSkin(image):
  # Taking a copy of the image
  img =  image.copy()
  # Converting from BGR Colours Space to HSV
  img =  cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
  
  # Defining HSV Threadholds
  lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
  upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
  
  # Single Channel mask,denoting presence of colours in the about threshold
  skinMask = cv2.inRange(img,lower_threshold,upper_threshold)
  
  # Cleaning up mask using Gaussian Filter
  skinMask = cv2.GaussianBlur(skinMask,(3,3),0)
  
  # Extracting skin from the threshold mask
  skin  =  cv2.bitwise_and(img,img,mask=skinMask)
  
  # Return the Skin image
  return cv2.cvtColor(skin,cv2.COLOR_HSV2BGR)

def removeBlack(estimator_labels, estimator_cluster):
  
  
  # Check for black
  hasBlack = False
  
  # Get the total number of occurance for each color
  occurance_counter = Counter(estimator_labels)

  
  # Quick lambda function to compare to lists
  compare = lambda x, y: Counter(x) == Counter(y)
   
  # Loop through the most common occuring color
  for x in occurance_counter.most_common(len(estimator_cluster)):
    
    # Quick List comprehension to convert each of RBG Numbers to int
    color = [int(i) for i in estimator_cluster[x[0]].tolist() ]
    
  
    
    # Check if the color is [0,0,0] that if it is black 
    if compare(color , [0,0,0]) == True:
      # delete the occurance
      del occurance_counter[x[0]]
      # remove the cluster 
      hasBlack = True
      estimator_cluster = np.delete(estimator_cluster,x[0],0)
      break
      
   
  return (occurance_counter,estimator_cluster,hasBlack)

def getColorInformation(estimator_labels, estimator_cluster,hasThresholding=False):
  
  # Variable to keep count of the occurance of each color predicted
  occurance_counter = None
  
  # Output list variable to return
  colorInformation = []
  
  
  #Check for Black
  hasBlack =False
  
  # If a mask has be applied, remove th black
  if hasThresholding == True:
    
    (occurance,cluster,black) = removeBlack(estimator_labels,estimator_cluster)
    occurance_counter =  occurance
    estimator_cluster = cluster
    hasBlack = black
    
  else:
    occurance_counter = Counter(estimator_labels)
 
  # Get the total sum of all the predicted occurances
  totalOccurance = sum(occurance_counter.values()) 
  
 
  # Loop through all the predicted colors
  for x in occurance_counter.most_common(len(estimator_cluster)):
    
    index = (int(x[0]))
    
    # Quick fix for index out of bound when there is no threshold
    index =  (index-1) if ((hasThresholding & hasBlack)& (int(index) !=0)) else index
    
    # Get the color number into a list
    color = estimator_cluster[index].tolist()
    
    # Get the percentage of each color
    color_percentage= (x[1]/totalOccurance)
    
    #make the dictionay of the information
    colorInfo = {"cluster_index":index , "color": color , "color_percentage" : color_percentage }
    
    # Add the dictionary to the list
    colorInformation.append(colorInfo)
    
      
  return colorInformation 


def extractDominantColor(image,number_of_colors=5,hasThresholding=False):
  
  # Quick Fix Increase cluster counter to neglect the black(Read Article) 
  if hasThresholding == True:
    number_of_colors +=1
  
  # Taking Copy of the image
  img = image.copy()
  
  # Convert Image into RGB Colours Space
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  
  # Reshape Image
  img = img.reshape((img.shape[0]*img.shape[1]) , 3)
  
  #Initiate KMeans Object
  estimator = KMeans(n_clusters=number_of_colors, random_state=0)
  
  # Fit the image
  estimator.fit(img)
  
  # Get Colour Information
  colorInformation = getColorInformation(estimator.labels_,estimator.cluster_centers_,hasThresholding)
  '''
  for x in colorInformation:
      (r,g,b) = x['color']
      (h,s,v) = colorsys.rgb_to_hsv(r,g,b)
  '''
  return colorInformation

def plotColorBar(colorInformation):
  #Create a 500x100 black image
  color_bar = np.zeros((100,500,3), dtype="uint8")
  
  top_x = 0
  for x in colorInformation:    
    bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

    color = tuple(map(int,(x['color'])))
  
    cv2.rectangle(color_bar , (int(top_x),0) , (int(bottom_x),color_bar.shape[0]) ,color , -1)
    top_x = bottom_x
  return color_bar

def prety_print_data(color_info):
  for x in color_info:
    print(pprint.pformat(x))
    print()
    colorsys.rgb_to_hsv(0.2, 0.4, 0.4)
'''    
image =  imutils.url_to_image("https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQQNpFRGRl5_THq5LI4I7mCJQjqDVihXWxl8yt5QwTo3U1ht1mD")

# Resize image to a width of 250
image = imutils.resize(image,width=250)

#Show image
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.show()


# Apply Skin Mask
skin = extractSkin(image)

plt.imshow(cv2.cvtColor(skin,cv2.COLOR_BGR2RGB))
plt.show()



# Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors 
dominantColors = extractDominantColor(skin,hasThresholding=True)




#Show in the dominant color information
print("Color Information")
prety_print_data(dominantColors)


#Show in the dominant color as bar
print("Color Bar")
colour_bar = plotColorBar(dominantColors)
plt.axis("off")
plt.imshow(colour_bar)
plt.show()
'''

'''
def convolve(B, r):
    D = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
    cv2.filter2D(B, -1, D, B)
    return B
M = 0
isColorExtractable = True 
isProjAvail = False

def hand_histogram(frame,b):
    global M

    b = (b[0]+30,b[1]+30,b[2]-45,b[3]-45)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = hsv_frame[int(b[1]+b[3]/2):int(b[1]+1.3*b[3]/2), int(b[0]+2*b[2]/3):int(b[0]+2.5*b[2]/3)] # Select ROI

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    M = cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)
    return M


def hist_masking(frame):
    
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], M, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 10, 255, cv2.THRESH_BINARY)

    thresh = cv2.dilate(thresh, None, iterations=3)

    thresh = cv2.merge((thresh, thresh, thresh))

    return cv2.bitwise_and(frame, thresh)



def thread_function(image,b):
    
    
    global projImage
    global isProjAvail
    hand_histogram(image, b)
    projImage = hist_masking(image)
    isProjAvail = True
    
    global M
    #Loading the image and converting to HSV
    #image = cv2.imread('/assets/img/skin-detection/zebra1.jpg')
    image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    model_hsv = image_hsv[int(b[1]+b[3]/2):int(b[1]+1.3*b[3]/2), int(b[0]+2*b[2]/3):int(b[0]+2.5*b[2]/3)] # Select ROI
    
    #Get the model histogram M
    M = cv2.calcHist([model_hsv], channels=[0, 1], mask=None, 
                      histSize=[80, 256], ranges=[0, 180, 0, 256] )
    
    #Backprojection of our original image using the model histogram M
    B = cv2.calcBackProject([image_hsv], channels=[0,1], hist=M, 
                             ranges=[0,180,0,256], scale=1)
    B = convolve(B, r=5)
    
    #Threshold to clean the image and merging to three-channels
    _, projImage = cv2.threshold(B, 30, 255, cv2.THRESH_BINARY)
    isProjAvail = True
    
    #cv2.imshow("backProj",thresh)
    
    dominantColors = ce.extractDominantColor(image,hasThresholding=isThresh)
    #dominantColors = ce.extractDominantColor(img[int(b[1]):int(b[1]+b[3]), int(b[0]):int(b[0]+b[2])],hasThresholding=True)
    #Show in the dominant color information
    global lower_color
    global upper_color
    #low_initial = lower_color
    #up_initial = upper_color
    #lower_color = np.array([179, 255, 255])
    #upper_color = np.array([0, 0, 0])
    for i in range (0,1) :
        (r,g,b) = dominantColors[i]['color']
        rgbcolor = np.uint8([[[b,g,r]]])
        hsvcolor = cv2.cvtColor(rgbcolor,cv2.COLOR_BGR2HSV)
        print(hsvcolor[0][0][0],' ',hsvcolor[0][0][1],' ',hsvcolor[0][0][2],' ',lower_color[0],' ',lower_color[1],' ',lower_color[2])
        if(hsvcolor[0][0][0]<lower_color[0]):
            lower_color[0] =hsvcolor[0][0][0]
        if(hsvcolor[0][0][1]<lower_color[1]):
            lower_color[1] =hsvcolor[0][0][1]
        if(hsvcolor[0][0][2]<lower_color[2]):
            lower_color[2] =hsvcolor[0][0][2]
        
        if(hsvcolor[0][0][0]>upper_color[0]):
            upper_color[0] =hsvcolor[0][0][0]
        if(hsvcolor[0][0][1]>upper_color[1]):
            upper_color[1] =hsvcolor[0][0][1]
        if(hsvcolor[0][0][2]>upper_color[2]):
            upper_color[2] =hsvcolor[0][0][2]
            
    
    #lower_color=np.array([lower_color[0]-10,lower_color[1]-20,lower_color[2]-20])
    #upper_color=np.array([upper_color[0],upper_color[1]+20,upper_color[2]+20])
    
    print( lower_color )
    print( upper_color )
    
    
    (r,g,b) = dominantColors[0]['color']
    bgrcolor = np.uint8([[[b,g,r]]])
    hsvcolor = cv2.cvtColor(bgrcolor,cv2.COLOR_BGR2HSV)
    lower_color=np.array([hsvcolor[0][0][0]-10,hsvcolor[0][0][1]-20,hsvcolor[0][0][2]-20])
    upper_color=np.array([hsvcolor[0][0][0]+10,hsvcolor[0][0][1]+20,hsvcolor[0][0][2]+20])
    
    print( lower_color )
    print( upper_color )
    # [[[ 60 255 255]]]
    
    (r,g,b) = dominantColors[0]['color']
    (h,s,v) = ce.colorsys.rgb_to_hsv(r/float(255),g/float(255),b/float(255))
    lower_color = np.array([h*180, s*255, v*255])
    
    (r,g,b) = dominantColors[4]['color']
    (h,s,v) = ce.colorsys.rgb_to_hsv(r/float(255),g/float(255),b/float(255))
    upper_color = np.array([h*180, s*255, v*255])
    
    
    
    print("lower: ", lower_color) 
    print("upper: ", upper_color) 
    print("Color Information")
    ce.prety_print_data(dominantColors)
    print("Color Bar")
    colour_bar = ce.plotColorBar(dominantColors)
    ce.plt.axis("off")
    ce.plt.imshow(colour_bar)
    ce.plt.show()
    ce.plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    ce.plt.show()
    '''
 '''
    if(isProjAvail):
        #Backprojection of our original image using the model histogram M
       
        
        b = (b[0]+30,b[1]+30,b[2]-45,b[3]-45)
        projImage = hist_masking(img)
        #img_preprocess = cv2.cvtColor(projImage, cv2.COLOR_BGR2GRAY)
        cv2.imshow("projImage",projImage)
        cv2.imshow("for_img",img[int(b[1]+1.7*b[3]/3):int(b[1]+2.3*b[3]/3), int(b[0]+b[2]/3):int(b[0]+1.5*b[2]/3)])
    ''' 