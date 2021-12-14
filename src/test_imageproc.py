import numpy as np
from image_proc import *
import pandas as pd 
import matplotlib.image as img
import matplotlib.pyplot as plt 

def testShift(img):

    img_shifted_up = shift_image(img, 0)
    img_shifted_right = shift_image(img, 1)
    img_shifted_left = shift_image(img, 2)
    img_shifted_down = shift_image(img, 3)

    fig = plt.figure()

    fig.add_subplot(3,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    fig.add_subplot(3,2,3)
    plt.imshow(img_shifted_up, cmap='gray')
    plt.title('Shifted up')
    plt.axis('off')
    
    fig.add_subplot(3,2,4)
    plt.imshow(img_shifted_right, cmap='gray')
    plt.title('Shifted right')
    plt.axis('off')

    fig.add_subplot(3,2,5)
    plt.imshow(img_shifted_down, cmap='gray')
    plt.title('Shifted down')
    plt.axis('off')

    fig.add_subplot(3,2,6)
    plt.imshow(img_shifted_left, cmap='gray')
    plt.title('Shifted left')
    plt.axis('off')

    fig.suptitle('Image Shifting', fontsize=16)


    plt.show()

def testFlip(img):
    img_flip = flip_image(img)

    fig = plt.figure()
    fig.suptitle('Image Flipping', fontsize=16)

    fig.add_subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    fig.add_subplot(1,2,2)
    plt.imshow(img_flip, cmap='gray')
    plt.title('Flipped')
    plt.axis('off')

    plt.show()

def testRotate(img):

    img_rotate45 = rotate_image(img, 45)
    img_rotate135 = rotate_image(img, 135)
    img_rotate315 = rotate_image(img, 315)
    
    fig = plt.figure()
    fig.suptitle('Image Rotating', fontsize=16)

    fig.add_subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    fig.add_subplot(2,2,2)
    plt.imshow(img_rotate45, cmap='gray')
    plt.title('45 degrees')
    plt.axis('off')

    fig.add_subplot(2,2,3)
    plt.imshow(img_rotate135, cmap='gray')
    plt.title('135 degrees')
    plt.axis('off')

    fig.add_subplot(2,2,4)
    plt.imshow(img_rotate315, cmap='gray')
    plt.title('315 degrees')
    plt.axis('off')

    plt.show()

def testBrightness(img):
    res1 = change_brightness(img, 100)
    res2 = change_brightness(img, -100)
    
    fig = plt.figure()
    fig.suptitle('Changing Brightness', fontsize=16)

    fig.add_subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    fig.add_subplot(1,3,2)
    plt.imshow(res1, cmap='gray')
    plt.title('Brighter')
    plt.axis('off')

    fig.add_subplot(1,3,3)
    plt.imshow(res2, cmap='gray')
    plt.title('Darker')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':

    raw_data = pd.read_csv('sign_mnist_test.csv')
    one_img = raw_data.iloc[5, 1:].to_numpy()
    img = one_img.reshape(28, 28)
    #testShift(img)
    #testFlip(img)
    #testRotate(img)
    #testBrightness(img)