
### Supporting code for Computer Vision Assignment 1
### See "Assignment 1.ipynb" for instructions

import math

import numpy as np

# import os
import skimage
from skimage import io


from skimage.color import rgb2gray
from skimage import data

def load(img_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.
    HINT: Converting all pixel values to a range between 0.0 and 1.0
    (i.e. divide by 255) will make your life easier later on!

    Inputs:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    
    img = io.imread(img_path)
    
    #we need to convert the int type to float otherwise it will all be 0 
    img = img.astype('float32')
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = img[i][j]/255
    

    

    return img



def print_stats(image):
    """ Prints the height, width and number of channels in an image.
        
    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).
        
    Returns: none
                
    """
    
    # YOUR CODE HERE
#     h,w = image.shape
#     c = len(image.shape)
#     string = "(" + str(h) + "," + str(w) + "," + str(c) + ")"
    
    return image.shape

def crop(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds. Use array slicing.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index 
        start_col (int): The starting column index 
        num_rows (int): Number of rows in our cropped image.
        num_cols (int): Number of columns in our cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """
    
    #using object slicing 
    crop_img = image[start_row:start_row + num_rows, start_col:start_col + num_cols ]



    return crop_img


def change_contrast(image, factor):
    """Change the value of every pixel by following

                        x_n = factor * (x_p - 0.5) + 0.5

    where x_n is the new value and x_p is the original value.
    Assumes pixel values between 0.0 and 1.0 
    If you are using values 0-255, change 0.5 to 128.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        factor (float): contrast adjustment

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    

    ### YOUR CODE HERE
    hi = image.shape[0]
    wi = image.shape[1]
#     image = image.astype('float32')
    for i in range(hi):
        for j in range(wi):
            image[i][j] = factor * (image[i][j] - 0.5 ) + 0.5

    return image



def resize(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.
    i.e. for each output pixel, use the value of the nearest input pixel after scaling

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """

    
    
    
    input_image = skimage.transform.resize(input_image,(output_rows,output_cols))
#     input_image = input_image.resize(output_rows,output_cols)
    return input_image

def greyscale(input_image):
    """Convert a RGB image to greyscale. 
    A simple method is to take the average of R, G, B at each pixel.
    Or you can look up more sophisticated methods online.
    
    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.

    Returns:
        np.ndarray: Greyscale image, with shape `(output_rows, output_cols)`.
    """
    hi = input_image.shape[0]
    wi = input_image.shape[1]
    for i in range(hi):
        for j in range(wi):
            input_image[i][j] = (input_image[i][j][0] + input_image[i][j][1] + input_image[i][j][2])/3

    return input_image



#add function code learnt from Youtube 
def conv_transform(image):
    image_copy = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_copy[i][j] = image[image.shape[0]-i-1][image.shape[1]-j-1]
    return image_copy



def conv2D(image, kernel):
    """ Convolution of a 2D image with a 2D kernel. 
    Convolution is applied to each pixel in the image.
    Assume values outside image bounds are 0.
    
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    #code learnt from Youtube by Akshaty Sharma 
        
    kernel = conv_transform(kernel) 
    
    image_h,image_w = image.shape[0],image.shape[1]
    
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]
    
    h = kernel_h//2 
    w = kernel_w//2 
    
    image_conv = np.zeros(image.shape)
    
    for i in range(h,image_h-h):
        for j in range(w,image_w-w):
            sum = 0 
            
            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum = sum + kernel[m][n]*image[i-h+m][j-w+n]
            image_conv[i][j] = sum   
                
    return image_conv




    

def test_conv2D():
    """ A simple test for your 2D convolution function.
        You can modify it as you like to debug your function.
    
    Returns:
        None
    """

    # Test code written by 
    # Simple convolution kernel.
    kernel = np.array(
    [
        [1,0,1],
        [0,0,0],
        [1,0,0]
    ])

    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    # Run your conv_nested function on the test image
    test_output = conv2D(test_img, kernel)

    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2:7, 2:7] = 1
    expected_output[5:, 5:] = 0
    expected_output[4, 2:5] = 2
    expected_output[2:5, 4] = 2
    expected_output[4, 4] = 3

    # Test if the output matches expected output
    assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."


def conv(image, kernel):
    """Convolution of a RGB or grayscale image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    #code learnt from Youtube by Akshaty Sharma 
        
    kernel = conv_transform(kernel) #similar of using flip  
    
    image_h,image_w = image.shape[0],image.shape[1]
    
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]
    
    h = kernel_h//2 
    w = kernel_w//2 
    
    image_conv = np.zeros(image.shape)
    
    for i in range(h,image_h-h):
        for j in range(w,image_w-w):
            sum = 0 
            
            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum = sum + kernel[m][n]*image[i-h+m][j-w+n]
            image_conv[i][j] = sum   
                
    return image_conv



    
def gauss2D(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function.
       You should not need to edit it.
       
    Args:
        size: filter height and width
        sigma: std deviation of Gaussian
        
    Returns:
        numpy array of shape (size, size) representing Gaussian filter
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def corr(image, kernel):
    """Cross correlation of a RGB image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    kernel = conv_transform(kernel) #similar of using flip  
    
    image_h,image_w = image.shape[0],image.shape[1]
    
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]
    
    h = kernel_h//2 
    w = kernel_w//2 
    
    image_corr = np.zeros(image.shape)
    
    for i in range(h,image_h-h):
        for j in range(w,image_w-w):
            sum = 0 
            
            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum = sum + kernel[m][n]*image[i+h-m][j+w-n]
            image_corr[i][j] = sum   
                
    return image_corr


