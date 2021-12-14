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
    h_old, w_old = img.shape
    h = int((h_old - kernel) / stride) + 1
    w = int((w_old - kernel) / stride) + 1
    output = np.zeros([h, w])

    y = y_out = 0
    while y + kernel <= h_old:
        x = x_out = 0
        while x + kernel <= w_old:
            output[y_out, x_out] = np.max(img[y:y+kernel, x:x+kernel])
            x += stride
            x_out += 1
        y += stride
        y_out += 1
    return output

def fc(img, weight, bias):
    (dim, _) = img.shape
    fc = img.reshape((dim * dim, 1))
    output = weight.dot(fc) + bias
    return output

def crossEntropyLoss(prediction, label):
    log_predictions = np.log(prediction)
    to_sum = label * log_predictions 
    return -np.sum(to_sum)

def ReLU(x):
    if x < 0:
        return 0 
    return x 

def initializeWeights(N_weights):
    """
    Returns a 1 x N_weights numpy array with randomly sampled points from a Gaussian distribution. 
    """
    return 0.01 * np.random.normal(loc=0.0, scale=1.0, size=N_weights)

def backpropagation_conv(conv_prev, conv_cur, mask, s): 
    """
    Calculates the backprojection through a convolution layer.
    """
    der_output = np.zeros_like(conv_cur)
    der_mask = np.zeros_like(mask)
    der_bias = np.zeros((mask.shape[0], 1))

    for curr_mask in range(mask.shape[0]):
        curr_y = 0
        result_y = 0

        while curr_y + mask.shape[2] <= conv_cur.shape[1]:
            curr_x = 0
            result_x = 0
            while curr_x + mask.shape[2] <= conv_cur.shape[1]:
                der_mask[curr_mask] += conv_prev[curr_mask, result_x, result_y] * conv_cur[:, curr_y:curr_y + mask.shape[2], curr_x:curr_x + mask.shape[2]]
                der_output[:, curr_y:curr_y+mask.shape[2], curr_x:curr_x+mask.shape[2]] += conv_prev[curr_mask, result_y, result_x] * mask[curr_mask] 
                curr_x = curr_x + s 
                output_x = output_x + s 
            
        der_bias[curr_mask] = np.sum(conv_prev[curr_mask])


    return der_output, der_mask, der_bias 

def find_max_index(x):
    """
    Finds the index of the largest valid value in the array
    """
    idx = np.nanargmax(x)
    idxs = np.unravel_index(idx, x.shape)
    return idxs 


def backpropagation_pool(der_pool, input, f, s):
    """
    Backpropagation through a maxpooling layer. 
    """

    der_output = np.zeros(input.shape)
    
    for curr in range(input.shape[0]):
        curr_y = out_y = 0
        while curr_y + f <= input.shape[1]:
            curr_x = out_x = 0
            while curr_x + f <= input.shape[1]:
                # obtain index of largest value in input for current window
                (a, b) = find_max_index(input[curr, curr_y:curr_y+f, curr_x:curr_x+f])
                der_output[curr, curr_y+a, curr_x+b] = der_pool[curr, out_y, out_x]
                
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        
    return der_output

