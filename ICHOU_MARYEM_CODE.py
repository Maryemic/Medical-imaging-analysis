# Packages : ***************************
import os
import scipy
import matplotlib as mpl
import skimage
from skimage import exposure
import cv2
import numpy as np
import pydicom as dicom
from PIL import Image
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
import shapely
#*****************************
## Lire l'image I0012.dcm
dataset=dicom.read_file("D:\\1.Master BIO-MSCS\\Traitement d'image II\\rapport\\retina\\I0012.dcm")

#******************************
## Afficher les métadonnées :
print(dataset)
##***********************************

print(dataset . dir ( "patient" ))
print(dataset.PatientName) # show patient name
print(dataset.PatientSex) # show patient gender
#**********************************
# Pour l'affichage du dimension : (Rows,columns)
print(dataset.pixel_array.shape)

#******************************
### Transformer l'image en matrice de pixel.
def dicom_to_array(filename):
    d = dicom.read_file(filename)
    a = d.pixel_array
    return np.array(a)
#***************************
# Taille de l'image
a1 = dicom_to_array("D:\\1.Master BIO-MSCS\\Traitement d'image II\\rapport\\retina\\I0012.dcm")
print(a1.size)
#************************************
#Résolution
# tell matplotlib to display our images as a 6 x 6 inch image, with resolution of 150 dpi
plt.figure(figsize = (6,6), dpi=150)
plt.imshow(a1, 'gray')


# tell matplotlib to display our images as a 6 x 6 inch image, with resolution of 80 dpi
plt.figure(figsize = (6,6), dpi=80)
plt.imshow(a1, 'gray')

#*******************************************
# La matrice des pixels
print(dataset.PixelData [: 200])
print(dataset.pixel_array, dataset.pixel_array.shape)
## On obtient que des zéros
#Il est possible que le début et la fin du tableau soient vraiment des zéros.
#Il est généralement plus significatif de regarder quelque chose au milieu de l'image.
midrow = dataset.Rows // 2
midcol = dataset.Columns // 2
print(dataset.pixel_array[midrow-20:midrow+20, midcol-20:midcol+20])

#****************************************************
### Use numpy.ndarray.min() and
#numpy.ndarray.max() to find the smallest and largest values in the array a1

print( np.ndarray.min(a1) )
print( np.ndarray.max(a1) )
#*************************************************
## Affichage de l'image
plt.imshow(dataset.pixel_array,plt.cm.bone)
plt.show()
#***************************************************

# Affichage d'une partie de l'image :
part=a1[200:400,100:300]
plt.subplot(2,2,1),plt.imshow(a1,cmap = 'gray')
plt.title('image')
plt.subplot(2,2,2),plt.imshow(part,cmap = 'gray')
plt.title('la partie affichée ')

plt.show()
#**********************************

# Use numpy.histogram() to calculate histogram datafor the values in the ndarray a1
hist,bins = np.histogram(a1, bins=256)

plt.figure() # create a new figure
plt.hist(a1) # plot a histogram of the pixel values
#******************************************************


## Histogram of Hounsfield Units of CT mouse image
plt.hist(a1.flatten(), bins=50, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

#*************************************

# grayscale : image aux niveaux de gris
fig1  = plt.figure()
plt.imshow(a1, cmap="gray", interpolation="bicubic")
plt.colorbar()
fig1.suptitle("Original + Gray colormap", fontsize=12)


#***************************************************************

#Améliorer contrast -Egalisation
a1_eq = exposure.equalize_hist(a1)
hist_eq,bins_eq = np.histogram(a1_eq, bins=256)

plt.figure()
plt.hist(a1_eq)

#******************************************************
# Image avec un contrast améliorer
fig2 = plt.figure()
plt.imshow(a1_eq, cmap="gray", interpolation="bicubic")
plt.colorbar()
fig2.suptitle("Histogram equalization + Gray colormap", fontsize=12)


#******************************************************
### Affichage d'une collection d'image
import matplotlib.pyplot as plt
import pydicom
folder = 'D:\\1.Master BIO-MSCS\\Traitement d\'image II\\rapport\\retiina\\'     #folder where photos are stored
fig=plt.figure(figsize=(8, 8))
columns=4
rows=5
# plot 19 images
for i in range(1,19):
    x= folder + str(i) + '.dcm'
    ds = pydicom.filereader.dcmread(x)
    fig.add_subplot(rows,columns,i)
    # load image pixels
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
plt.show()
#**************************************************

# # Transformations morphologiques et filtres


import cv2
import numpy as np
a1 = dicom_to_array("D:\\1.Master BIO-MSCS\\Traitement d'image II\\rapport\\retina\\I0012.dcm")
#*********************************************
#Erosion
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(a1,kernel,iterations = 2)
plt.figure(figsize=(12, 12))
plt.subplot(131)
plt.imshow(a1, 'gray')
plt.title('Original')
plt.subplot(132)
plt.imshow(erosion, 'gray')
plt.title('erosion')
plt.show()


#*************************************************

## Dilatation : C'est  l'opposé de l'érosion.
dilation = cv2.dilate(a1,kernel,iterations = 1)
plt.figure(figsize=(12, 12))
plt.subplot(131)
plt.imshow(a1, 'gray')
plt.title('Original')
plt.subplot(132)
plt.imshow(dilation, 'gray')
plt.title('dilation')
plt.show()


#**********************************************************

## Ouverture :L'ouverture n'est qu'un autre nom d' érosion suivi de dilatation . Il est utile pour supprimer le bruit
opening = cv2.morphologyEx(a1, cv2.MORPH_OPEN, kernel)
plt.figure(figsize=(12, 12))
plt.subplot(131)
plt.imshow(a1, 'gray')
plt.title('Original')
plt.subplot(132)
plt.imshow(opening, 'gray')
plt.title('opening')
plt.show()


#*******************************************************

## La fermeture est l'inverse de l'ouverture, la dilatation suivie de l'érosion
closing = cv2.morphologyEx(a1, cv2.MORPH_CLOSE, kernel)
plt.figure(figsize=(12, 12))
plt.subplot(131)
plt.imshow(a1, 'gray')
plt.title('Original')
plt.subplot(132)
plt.imshow(closing, 'gray')
plt.title('closing')
plt.show()


#*****************************************************

## Gradient morphologique : C'est la différence entre la dilatation et l'érosion d'une image.
gradient = cv2.morphologyEx(a1, cv2.MORPH_GRADIENT, kernel)
plt.figure(figsize=(12, 12))
plt.subplot(131)
plt.imshow(a1, 'gray')
plt.title('Original')
plt.subplot(132)
plt.imshow(gradient, 'gray')
plt.title('gradient')
plt.show()## Le résultat ressemblera au contour de l'objet.


#********************************************************************

## tophat : C'est la différence entre l'image d'entrée et l'ouverture de l'image.
tophat = cv2.morphologyEx(a1, cv2.MORPH_TOPHAT, kernel)
plt.figure(figsize=(12, 12))
plt.subplot(131)
plt.imshow(a1, 'gray')
plt.title('Original')
plt.subplot(132)
plt.imshow(tophat, 'gray')
plt.title('tophat')
plt.show()


#********************************************************

##Blackhat : la différence entre la fermeture de l'image d'entrée et de l'image d'entrée.
blackhat = cv2.morphologyEx(a1, cv2.MORPH_BLACKHAT, kernel)
plt.figure(figsize=(12, 12))
plt.subplot(131)
plt.imshow(a1, 'gray')
plt.title('Original')
plt.subplot(132)
plt.imshow(blackhat, 'gray')
plt.title('blackhat')
plt.show()


#*******************************************************

#Laplacian Derivatives : dérivés laplaciens
import cv2
import numpy as np
from matplotlib import pyplot as plt


laplacian = cv2.Laplacian(a1,cv2.CV_64F)
sobelx = cv2.Sobel(a1,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(a1,cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(a1,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()


#********************************************************************

## filtre Sobel horizontal et la différence des résultats
import cv2
import numpy as np
from matplotlib import pyplot as plt


# Output dtype = cv2.CV_8U
sobelx8u = cv2.Sobel(a1,cv2.CV_8U,1,0,ksize=5)

# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
sobelx64f = cv2.Sobel(a1,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

plt.subplot(1,3,1),plt.imshow(a1,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

plt.show()


#************************************************************************

## créer un masque sur une image
import numpy as np
from skimage import data
import matplotlib.pyplot as plt
type(a1) #Image est un array numpy
mask = a1 < 87
a1[mask]=255

plt.figure(figsize=(12, 12))
plt.subplot(131)
plt.imshow(a1, 'gray')
plt.title('Original')
plt.subplot(132)
plt.imshow(opening, 'gray')
plt.title('Masked')
plt.show()


#*****************************************************

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
import pylab
import matplotlib.cm as cm
from skimage.color import rgb2gray
from scipy import ndimage
from skimage.io import imread
from skimage.color import rgb2gray
from tkinter import Tk, W, E
from tkinter.ttk import Frame, Button, Entry, Style


##**********************************************************************
# Filtre gaussien
blurred_face = ndimage.gaussian_filter(a1, sigma=3)
very_blurred = ndimage.gaussian_filter(a1, sigma=5)
plt.figure(figsize=(11, 6))

plt.subplot(121), plt.imshow(blurred_face, cmap='gray')
plt.title('blurred_face'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(very_blurred, cmap='gray')
plt.title('very_blurred'), plt.xticks([]), plt.yticks([])

plt.show()

########################****************************
## convolution
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(a1,-1,kernel)
plt.subplot(121),plt.imshow(a1,'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst,'gray'),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()