import numpy as np

def conv2d(img, mask):
    """ 
    Function that convs img, a 2d numpy array, with mask with zero padding.
    Given an m x n img and a p x p mask, the output image will be  (2m - p) + 1 x (2n - p) + 1
    """

    # first flip the kernel 
    mask = np.flipud(np.fliplr(mask))
    output = np.zeros_like(img)

    # zero pad the img 
    padding_size = int(mask.shape[0]/2)
    img_zero_padded = np.zeros((img.shape[0] + 2* padding_size, img.shape[1] + 2*padding_size))
    img_zero_padded[padding_size:-padding_size, padding_size:-padding_size] = img

    # start the convolution by iterating over the image 
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            center_coords_padded_img = [r+padding_size, c+padding_size]
            mask_window_multiply = np.multiply(mask, img_zero_padded[center_coords_padded_img[0] - padding_size:center_coords_padded_img[0] + padding_size+1, center_coords_padded_img[1] - padding_size:center_coords_padded_img[1] + padding_size+1])
            output[r,c] = np.sum(mask_window_multiply, axis=None)

    return output 

def maxpool(img, kernel=2, stride=2):
    """
    Function that performs max pooling on img.
    """
    n, h_old, w_old = img.shape
    h = int((h_old - kernel) / stride) + 1
    w = int((w_old - kernel) / stride) + 1

    output = np.zeros(n, h_old, w_old)

    for i in range(n):
        y = y_out = 0
        while y + kernel <= h_old:
            x = x_out = 0
            while x + kernel <= w_old:
                output[i, y_out, x_out] = np.max(img[i, y:y+kernel, x:x+kernel])
                x += stride
                x_out += 1
            y += stride
            y_out += 1
    return output
