import numpy as np 
from numpy import matlib
import scipy as sp
from scipy import ndimage

def shift_image(img, dir):
    """
    Function that shifts the image vertically or horizontally for data augmentation. 
    input: img the 2D numpy array, dir to specify shifting direction: 0 for up, 1 for right, 2 for left, 3 for down
    output: the shifted img, a 2D numpy array
    """

    img_shifted = np.zeros_like(img)
    img_height = img.shape[0]
    mid_line = int(img_height / 2)
    if (dir == 0):
        # up
        img_shifted[0:mid_line, :] = img[mid_line:,:]
        img_shifted[mid_line:, :] = img[-1] 
    elif (dir == 1):
        # right
        img_shifted[:,mid_line:] = img[:, 0:mid_line]
        img_shifted[:,0:mid_line] = img[:, 0][...,None]
    elif (dir == 2):
        # left
        img_shifted[:,0:mid_line] = img[:,mid_line:]
        img_shifted[:,mid_line:] = img[:,-1][...,None]
    else:
        # down
        img_shifted[mid_line:,:] = img[0:mid_line,:]
        img_shifted[0:mid_line,:] = img[0]

    return img_shifted

def flip_image(img):
    """
    Function that flips the image horizontally for data augmentation
    input: img the 2D numpy array
    """

    return np.fliplr(img)

def rotate_image(img, theta):

   return sp.ndimage.rotate(img, theta, reshape='False')

def change_brightness(img, val):
    """
    Function to change brightness of an image
    """
    res = np.zeros_like(img)

    if val > 0:
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                res[row,col] = min(255, val + img[row,col])
    else:
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                res[row,col] = max(0, img[row,col] + val)

    return res 