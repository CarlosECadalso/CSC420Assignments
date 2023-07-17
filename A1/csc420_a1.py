# -*- coding: utf-8 -*-
"""CSC420 A1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18jz4RaBpNCrxqc7TQv22WmhEuJy62QlQ

#Filters
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import color


def a1_conv(image_to_conv, filter_to_apply):
    image_grayscale = color.rgb2gray(image_to_conv)  # grayscale it
    filter_h, filter_w = filter_to_apply.shape  # get the dimensions of the filter
    padding = filter_h // 2  # calculate the necesary padding

    # create empty matrices for the padded image and output
    padded_waldo = np.pad(image_grayscale, ((padding, padding), (padding, padding)), 'constant')
    output_image = np.zeros(image_grayscale.shape)

    # convolve filter
    for pad_row in range(padding, output_image.shape[0] + padding):
        for pad_col in range(padding, output_image.shape[1] + padding):
            # note: this method of convolving utilizes the property of convolution that says:
            # 2D convolution is equivalent to 2D correlation with the filter flipped over both axis
            output_image[pad_row - padding, pad_col - padding] = np.sum(padded_waldo[pad_row - padding:pad_row + padding + 1, pad_col - padding:pad_col + padding + 1] * np.flip(filter_to_apply))

    return output_image  # return the result


waldo = plt.imread("waldo.png")[..., :3]  # get the waldo image

# create the test filter
test_filter = np.zeros((3, 3))
test_filter[0, 1] = 0.5
test_filter[1, 0] = 0.125
test_filter[1, 1] = 0.5
test_filter[1, 2] = 0.5
test_filter[2, 1] = 0.125

# create the test filter
test_filter = np.zeros((3, 3))
test_filter[0, 0] = 1
test_filter[0, 1] = 0
test_filter[0, 2] = 1
test_filter[1, 0] = 2
test_filter[1, 1] = 0
test_filter[1, 2] = 2
test_filter[2, 0] = 1
test_filter[2, 1] = 0
test_filter[2, 2] = 1

# call the function
output = a1_conv(waldo, test_filter)

print("The original waldo.png image")
plt.axis("off")
plt.imshow(waldo)
plt.show()

print("The waldo.png image after being grayscaled and being convolved by the provided test filter")
plt.axis("off")
plt.imshow(output)
plt.show()

import numpy as np

def is_separable(filter_to_check):
    U, D, VT = np.linalg.svd(filter_to_check, full_matrices=False)
    error_range = D[0] * 1e-10

    for value in D[1:]:
      if value > error_range:
        return False
    return True

# create the test filter
test_filter = np.zeros((3, 3))
test_filter[0, 1] = 0.5
test_filter[1, 0] = 0.125
test_filter[1, 1] = 0.5
test_filter[1, 2] = 0.5
test_filter[2, 1] = 0.125

print(is_separable(test_filter))

import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from math import sqrt

def aux_id_conv(matrix_1d, filter_1d, padding):
  output = np.zeros(matrix_1d.shape)
  for i in range(padding, output.shape[0]-padding):
    output[i] = np.sum(matrix_1d[i-padding:i + padding + 1] * filter_1d)
  return output


def a1_conv_separable(image_to_conv, filter_to_apply):
    image_grayscale = color.rgb2gray(image_to_conv)  # grayscale it
    filter_h, filter_w = filter_to_apply.shape  # get the dimensions of the filter
    padding = filter_h // 2  # calculate the necessary padding
    filter_U, filter_D, filter_VT = np.linalg.svd(filter_to_apply, full_matrices=False)
    filter_sigma_root = sqrt(filter_D[0])
    # note: this method of convolving utilizes the property of convolution that says:
    # 2D convolution is equivalent to 2D correlation with the filter flipped over both axis
    vertical_filter = np.flip(filter_sigma_root * filter_U[:, 0])
    horizontal_filter = np.flip(filter_sigma_root * filter_VT[0, :])

    # create empty matrices for the padded image and output
    padded_waldo = np.pad(image_grayscale, ((padding, padding), (padding, padding)), 'constant')
    filtered_image = np.zeros(padded_waldo.shape)
    output_image = np.zeros(image_grayscale.shape)

    # convolve filter
    for pad_row in range(padding, output_image.shape[0] + padding):
      filtered_image[pad_row, :] = aux_id_conv(padded_waldo[pad_row,:], horizontal_filter, padding)

    for pad_col in range(padding, output_image.shape[1] + padding):
      filtered_image[:, pad_col] = aux_id_conv(filtered_image[:, pad_col], vertical_filter, padding)

    output_image = filtered_image[padding:-padding, padding:-padding]

    return output_image  # return the result

waldo = plt.imread("waldo.png")[..., :3]  # get the waldo image
waldo_grayscale = color.rgb2gray(waldo)

# create the test filter
test_filter = np.zeros((3, 3))
test_filter[0, 0] = 1
test_filter[0, 1] = 0
test_filter[0, 2] = 1
test_filter[1, 0] = 2
test_filter[1, 1] = 0
test_filter[1, 2] = 2
test_filter[2, 0] = 1
test_filter[2, 1] = 0
test_filter[2, 2] = 1

output = a1_conv_separable(waldo, test_filter)

print("The waldo.png image after being grayscaled and being convolved by a separable test filter")
plt.axis("off")
plt.imshow(output)
plt.show()

# print(test_case)
# plt.axis("off")
# plt.imshow(test_case)
# plt.show()

waldo = plt.imread("waldo.png")[..., :3]  # get the waldo image
waldo_grayscale = color.rgb2gray(waldo)

# create the test filter
test_filter = np.zeros((3, 3))
test_filter[0, 0] = 0
test_filter[0, 1] = 0.5
test_filter[0, 2] = 0
test_filter[1, 0] = 0.125
test_filter[1, 1] = 0.5
test_filter[1, 2] = 0.5
test_filter[2, 0] = 0
test_filter[2, 1] = 0.125
test_filter[2, 2] = 0

output = a1_conv_separable(waldo, np.flip(test_filter))

print("The waldo.png image after being grayscaled and being correlated by the given test filter")
plt.axis("off")
plt.imshow(output)
plt.show()

import numpy as np
import math
pi = math.pi

def a1_generate_gaussian(kernel_size, sigma):
    half_size = kernel_size // 2
    x, y = np.meshgrid(np.arange(-half_size, half_size + 1), np.arange(-half_size, half_size + 1))

    base = 1/(2 * pi * sigma**2)
    exponent = np.exp(-(x**2 + y**2) / (sigma**2))

    gaussian_output = base * exponent
    return gaussian_output


waldo = plt.imread("waldo.png")[..., :3]  # get the waldo image
waldo_grayscale = color.rgb2gray(waldo)

a1_gaussian = a1_generate_gaussian(3, 1)

from scipy.ndimage import convolve as conv

print("Gaussian Filter:")
print(a1_gaussian)
print("")

gaus_output = conv(waldo_grayscale, a1_gaussian, mode='constant')
print("The waldo.png image after being grayscaled and being convolved with a gaussian")
plt.axis("off")
plt.imshow(gaus_output)
plt.show()

import numpy as np

def separate_matrix_into_vectors(matrix):
    U, S, VT = np.linalg.svd(matrix)
    S = np.diag(S)
    U = U[:, 0]
    VT = VT[0, :]
    return U, S[0], VT

# create the test filter
test_filter = np.zeros((3, 3))
test_filter[0, 0] = -1
test_filter[0, 1] = 0
test_filter[0, 2] = 1
test_filter[1, 0] = -2
test_filter[1, 1] = 0
test_filter[1, 2] = 2
test_filter[2, 0] = -1
test_filter[2, 1] = 0
test_filter[2, 2] = 1

print(test_filter)

U, S, VT = separate_matrix_into_vectors(test_filter)

filter_sigma_root = sqrt(S[0])
vertical_filter = filter_sigma_root * U
horizontal_filter = filter_sigma_root * VT

np.outer(vertical_filter, horizontal_filter)

"""# Gradients"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from scipy.ndimage import correlate as corr

def gradient_magnitude_and_direction(image):
    Sobel_x = np.array([[-1,0,1]])
    Sobel_y = np.array([[1,2,1]])
    Sobel_kernel_y = Sobel_x.T @ Sobel_y
    Sobel_kernel_x = Sobel_y.T @ Sobel_x

    # calculate gradient in x direction using derivative
    gradient_x = corr(image, Sobel_kernel_x)

    # calculate gradient in y direction using derivative
    gradient_y = corr(image, Sobel_kernel_y)

    # calculate magnitude of gradient
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # calculate gradient direction
    direction = np.arctan2(gradient_y, gradient_x)

    return magnitude, direction

# waldo = plt.imread("waldo.png")[..., :3]  # get the waldo image
# waldo_grayscale = color.rgb2gray(waldo)
# waldo_magnitude, waldo_direction = gradient_magnitude_and_direction(waldo_grayscale)

# print("The magnitude of the gradient of waldo.png")
# plt.axis("off")
# plt.imshow(waldo_magnitude)
# plt.show()
# print("The direction of the gradient of waldo.png")
# print(waldo_direction)
# print(np.max(waldo_direction))
# plt.axis("off")
# plt.imshow(waldo_direction)
# plt.show()

# template = plt.imread("template.png")[..., :3]  # get the waldo image
# template_grayscale = color.rgb2gray(template)
# template_magnitude, template_direction = gradient_magnitude_and_direction(template_grayscale)

# print("The magnitude of the gradient of template.png")
# plt.axis("off")
# plt.imshow(template_magnitude)
# plt.show()
# print("The direction of the gradient of template.png")
# plt.axis("off")
# plt.imshow(template_direction)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from scipy.ndimage import correlate as corr

waldo = plt.imread("waldo.png")[..., :3]  # get the waldo image
waldo_grayscale = color.rgb2gray(waldo)
waldo_grad, dump = gradient_magnitude_and_direction(waldo_grayscale)

def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def find_in_waldo(input_template):
  template_grad, dump = gradient_magnitude_and_direction(input_template)
  output_corr = corr(waldo_grad, template_grad)

  output_corr = normalize_image(output)
  matrix_max = output_corr.max()
  for row in range(output_corr.shape[0]):
      for col in range(output_corr.shape[1]):
        if output_corr[row, col] == matrix_max:
          # note: normally this would be used to remove non-local maxima
          # however, the visualization does not show the maxima without
          # doing this because waldo.png is over 1000 pixels, so the single
          # pixel maxima does not show up without making adjacent pixels
          # stand out more
          output_corr[row-2:row+3, col-2:col+3] = output[row, col] * 10

  return output_corr



template = plt.imread("template.png")[..., :3]  # get the waldo image
template_grayscale = color.rgb2gray(template)
output = find_in_waldo(template_grayscale)


print("Result of attempting to localizes the template.png in the image waldo.png")
plt.axis("off")
plt.imshow(output)
plt.show()

import numpy as np

def canny_edge_detect(img):
  img_gradient_magnitude, img_gradient_direction\
                        = gradient_magnitude_and_direction(img)
  output_img = np.zeros(img.shape)

  degree_convter = 180 / np.pi
  p = np.max(img_gradient_magnitude)
  r = np.max(img_gradient_magnitude)


  for row in range(1, img.shape[0]-1):
    for col in range(1, img.shape[1]-1):
      # convert to degrees between 0 and 180
      # note: we do not need to check all 360 degrees
      # because 180 degrees already has all directions
      angle = (img_gradient_direction[row, col] + np.pi) * degree_convter

      # if direction is 0 degree or 180 degree
      if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
        p = img_gradient_magnitude[row, col+1]
        r = img_gradient_magnitude[row, col-1]

      # if direction is 45 degree
      elif (22.5 <= angle < 67.5):
        p = img_gradient_magnitude[row-1, col+1]
        r = img_gradient_magnitude[row+1, col-1]

      # if direction is 90 degree
      elif (67.5 <= angle < 112.5):
        p = img_gradient_magnitude[row+1, col]
        r = img_gradient_magnitude[row-1, col]

      # if direction is 135 degree
      # this could be an else, but it was left as an elif
      # for the sake of readability
      elif (112.5 <= angle < 157.5):
        p = img_gradient_magnitude[row-1, col-1]
        r = img_gradient_magnitude[row+1, col+1]

      # compare and leave the max in while removing the rest
      if (img_gradient_magnitude[row, col] >= p) and (img[row, col] >= r):
        output_img[row, col] = img_gradient_magnitude[row, col]
      # else, the value of output_img[row, col] will remain as 0

  return output_img # return the resulting matrix

waldo = plt.imread("waldo.png")[..., :3]  # get the waldo image
waldo_grayscale = color.rgb2gray(waldo)

canny_edged = canny_edge_detect(waldo_grayscale)

print("Result of appying Canny Edge Detection")
print("(with non-maxima supression) to waldo.png")
plt.axis("off")
plt.imshow(canny_edged)
plt.show()

import numpy as np
from scipy.ndimage.filters import gaussian_filter

def harris_corner_detection(image, loaclity=20):
    # Note: Since the maxima suppression occurs within a local area, I created
    # a variable "locality" to keep track of how "local" should the maxima
    # suppression be

    img_len, img_width = image.shape

    # 1. Compute gradients Ix and Iy
    Ix, Iy = np.gradient(image)

    # 2. Finding the products of gradients for the matrix:
    # [Ixx  Ixy]
    # [Ixy  Iyy]
    Ixx = Ix**2
    Ixy = Ix*Iy
    Iyy = Iy**2

    # 3. Applying Gaussian filter for weighted average of gradients
    Ixx = gaussian_filter(Ixx, 1)
    Ixy = gaussian_filter(Ixy, 1)
    Iyy = gaussian_filter(Iyy, 1)

    # 3.1. Create M (this step is mostly for clearity)
    M = np.zeros((2, 2))
    M[0,0] = Ixx
    M[1,0] = Ixy
    M[0,1] = Ixy
    M[0,0] = Iyy

    # 4. Finding R
    # 4.1. Finding det(M)
    detM = np.linalg.det(M)
    detM = Ixx*Iyy - Ixy**2

    # 4.2. Finding trace(M)
    traceM = np.trace(M)
    traceM = Ixx + Iyy

    # 4.3. Solving for R
    alpha=0.04
    R = detM - (alpha * (traceM**2))

    # 5. find points with large R
    corners = np.zeros(image.shape)
    corner_list = [0]
    threshold=0.01

    for i in range(img_len):
      for i in range(img_len):
        # Remove everything under threshold
        if R[i,j] <= threshold:
          R[i,j] = 0

    # 6. Non-maxima suppression
    for i in range(loaclity, img_len-loaclity):
        for j in range(loaclity, img_width-loaclity):
          if np.max(R[i-loaclity:i+loaclity+1, j-loaclity:j+loaclity+1]):
            corners[i,j] = 255
            corner_list.append((i,j))

    return corners, corner_list