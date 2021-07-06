import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as Functional

base_dir = os.getcwd()

image_path = base_dir + "/data/udacity_sdc.png"


def show_image(path: str):
    # load color image
    bgr_img = cv2.imread(path)
    # convert to grayscale
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # normalize, rescale entries to lie in [0,1]
    gray_img = gray_img.astype("float32") / 255

    # plot image
    plt.imshow(gray_img, cmap='gray')
    plt.show()


filter_values = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])

# define four filters
filter_1 = filter_values
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])

# For an example, print out the values of filter 1
print('Filter 2: \n', filter_2)
"""
Filter 2:
 [[ 1  1 -1 -1]
 [ 1  1 -1 -1]
 [ 1  1 -1 -1]
 [ 1  1 -1 -1]]

"""


def visualize_filter(no_filters: int):
    """
    This will display the first n number of filters. output of this function is stored in images
    folder with function name
    :param no_filters: int
    :return:
    """
    # visualize all given filters
    fig = plt.figure(figsize=(10, 5))
    for i in range(no_filters):
        ax = fig.add_subplot(1, no_filters, i + 1, xticks=[], yticks=[])
        ax.imshow(filters[i], cmap='gray')
        ax.set_title('Filter %s' % str(i + 1))
        width, height = filters[i].shape
        for x in range(width):
            for y in range(height):
                ax.annotate(str(filters[i][x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='white' if filters[i][x][y] < 0 else 'black')
    plt.show()


# define a neural network with a single convolutional layer with four filters
class Net(torch.nn.Module):
    def __init__(self, weight):
        super(Net, self).__init__()
        # initializes the weights of the convolutional layer to be the weights of the 4 defined filters
        k_height, k_width = weight.shape[2:]
        # assumes there are 4 grayscale filters
        self.conv = torch.nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        # calculates the output of a convolutional layer pre- and post-activation
        conv_x = self.conv(x)
        activated_x = Functional.relu(conv_x)

        # return both layer
        return conv_x, activated_x


# instantiate the model and set the weights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)

# print out the layer in the network
print(model)
"""
Net(
  (conv): Conv2d(1, 4, kernel_size=(4, 4), stride=(1, 1), bias=False)
)
"""


# helper function for visualizing the output of a given layer
# default number of filters is 4
def viz_layer(layer, n_filters=4):
    fig = plt.figure(figsize=(20, 20))

    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filters, i + 1, xticks=[], yticks=[])
        # grab layer outputs
        ax.imshow(np.squeeze(layer[0, i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i + 1))
    plt.show()


# load color image
bgr_img = cv2.imread(image_path)
# convert to grayscale
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

# normalize, rescale entries to lie in [0,1]
gray_img = gray_img.astype("float32") / 255

# convert the image into an input Tensor
gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)

# get the convolutional layer (pre and post activation)
conv_layer, activated_layer = model(gray_img_tensor)

# visualize the output of a conv layer
viz_layer(conv_layer)


# after a ReLu is applied
# visualize the output of an activated conv layer
viz_layer(activated_layer)
