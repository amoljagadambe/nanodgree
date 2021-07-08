"""
Refer the article: https://setosa.io/ev/image-kernels/
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import os

base_dir = os.getcwd()

image = mpimg.imread(base_dir + "/data/curved_lane.jpg")
# Convert to grayscale for filtering
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def show_image(image_array):
    plt.imshow(image_array, cmap='gray')
    plt.show()


def apply_2d_filter(gery_image, custom_filter, bit_depth=-1):
    """
    # Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)
    :param custom_filter:
    :param gery_image:
    :param bit_depth:
    :return: filtered array
    """
    return cv2.filter2D(gery_image, bit_depth, custom_filter)


def bottom_sobel(image):
    """
    sobel kernels are used to show only the differences in adjacent pixel values in a particular direction.
    :param image:
    :return: filtered image stored in image folder with function name
    """
    # 3x3 array for edge detection
    bottom_sobel_filter = np.array([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]])

    filtered_image_y = apply_2d_filter(image, bottom_sobel_filter)
    return show_image(filtered_image_y)


def top_sobel(image):
    """
    sobel kernels are used to show only the differences in adjacent pixel values in a particular direction.
    :param image:
    :return: filtered image stored in image folder with function name
    """
    # 3x3 array for edge detection
    top_sobel_filter = np.array([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]])

    filtered_image_y = apply_2d_filter(image, top_sobel_filter)
    return show_image(filtered_image_y)


def left_sobel(image):
    """
    sobel kernels are used to show only the differences in adjacent pixel values in a particular direction.
    :param image:
    :return: filtered image stored in image folder with function name
    """
    # 3x3 array for edge detection
    left_sobel_filter = np.array([[1, 0, -1],
                                  [2, 0, -2],
                                  [1, 0, -1]])

    filtered_image_y = apply_2d_filter(image, left_sobel_filter)
    return show_image(filtered_image_y)


def right_sobel(image):
    """
    sobel kernels are used to show only the differences in adjacent pixel values in a particular direction.
    :param image:
    :return: filtered image stored in image folder with function name
    """
    # 3x3 array for edge detection
    right_sobel_filter = np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]])

    filtered_image_y = apply_2d_filter(image, right_sobel_filter)
    return show_image(filtered_image_y)


# TODO: write function for below filter also
"""
# 3x3 array for edge detection
sharpen_filter = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])

blur_filter = np.array([[0.0625, 0.125, 0.0625],
                        [0.125, 0.25, 0.125],
                        [0.0625, 0.125, 0.0625]])
"""

from keras.models import Sequential
from keras.layers import MaxPooling2D

model = Sequential()
model.add(MaxPooling2D(pool_size=(2,2), strides=3, input_shape=(100, 100, 15)))
model.summary()