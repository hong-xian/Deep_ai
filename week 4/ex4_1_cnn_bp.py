import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'


# zero padding
def zero_pad(x, pad):
    """
    pad all images in dataset x with zeros, the padding is applied for the height and width of image
    :param x: numpy array with shape(m, n_C, n_H, n_W)
    :param pad: integer
    :return: padded image of shape (m, n_C, n_H+2*pad, n_W+2*pad)

    """
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    return x_pad


# np.random.seed(1)
# a = np.random.randn(4, 2, 3, 3)
# a_pad = zero_pad(a, 2)
# print("a.shape:", a.shape)
# print("a_pad.shape:", a_pad.shape)
# print("a[1, 1] =", a[0, 0])
# print("a_pad[1, 1] =", a_pad[0, 0])
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(a[0, 0])
# axes[0].set_title("x")
# axes[1].imshow(a_pad[0, 0])
# axes[1].set_title("x_pad")

# single step of convolution
def conv_single_step(x_slice_prev, w, b):
    """
    apply one filter defined by parameter w on a single slice of the output activation of the previous layer
    :param x_slice_prev: slice of input data of shape(n_C, f, f)
    :param w: matrix of shape(n_C, f, f)
    :param b: matrix of shape(1, 1, 1)
    :return: a scalar value
    """
    a = x_slice_prev * w
    z = np.sum(a)
    z = z + b
    return z


# convolution forward
def conv_forward(x_prev, weight, b, hparameters):
    """
    implements of forward convolution
    :param x_prev: output of previous layer, numpy array with shape(m, n_C_prev, n_H_prev, n_W_prev)
    :param weight: filter of shape(n_C_prev, n_C, f, f)
    :param b: bias of shape(1, n_C, 1, 1)
    :param hparameters: dict containing "stride" and "padding"
    :return:
    z -- conv output, shape(m, n_C, n_H, n_W)
    cache -- cache of values needed for backward propagation
    """
    (m, n_C_prev, n_H_prev, n_W_prev) = x_prev.shape
    (n_C_prev, n_C, f, f) = weight.shape
    stride = hparameters["stride"]
    padding = hparameters["padding"]
    n_H = int((n_H_prev - f + 2 * padding) / stride) + 1
    n_W = int((n_W_prev - f + 2 * padding) / stride) + 1
    z = np.zeros((m, n_C, n_H, n_W))
    x_prev_pad = zero_pad(x_prev, padding)

    for i in range(m):
        x_prev_pad_in = x_prev_pad[i]
        for c in range(n_C):
            for h in range(n_H):
                for w in range(n_W):
                    vert_start = stride * h
                    vert_end = vert_start + f
                    horiz_start = stride * w
                    horiz_end = horiz_start + f
                    x_slice_prev = x_prev_pad_in[:, vert_start:vert_end, horiz_start:horiz_end]
                    z[i, c, h, w] = conv_single_step(x_slice_prev, weight[:, c, :, :], b[:, c, :, :])

    assert (z.shape == (m, n_C, n_H, n_W))
    cache = (x_prev, weight, b, hparameters)
    return z, cache


# pooling forward
def pool_forward(x_prev, hparameters, mode="max"):
    """
    implements of pooling forward
    :param x_prev: input data of shape(m, n_C_prev, n_H_prev, n_W_prev)
    :param hparameters: dict containing "f" and "stride"
    :param mode: pooling mode you can chose, defined as string "max" or "average"
    :return:
    a -- output of pooling layer of shape(m, n_C, n_H, n_W)
    cache -- used for backward propagation, input and hparameters
    """
    (m, n_C_prev, n_H_prev, n_W_prev) = x_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev
    a = np.zeros((m, n_C, n_H, n_W))
    for i in range(m):
        for c in range(n_C):
            for h in range(n_H):
                for w in range(n_W):
                    vert_start = stride * h
                    vert_end = vert_start + f
                    horiz_start = stride * w
                    horiz_end = horiz_start + f
                    x_slice_prev = x_prev[i, c, vert_start:vert_end, horiz_start:horiz_end]

                    if mode == "max":
                        a[i, c, h, w] = np.max(x_slice_prev)
                    if mode == "average":
                        a[i, c, h, w] = np.average(x_slice_prev)
    cache = (x_prev, hparameters)
    assert(a.shape == (m, n_C, n_H, n_W))
    return a, cache


# convolution layer backward propagation
def conv_backward(dz, cache):
    """
    Implement the backward propagation for a convolution function

    Arguments:
    dz --gradient of the cost with respect to the output of the conv layer (z), numpy array of shape (m,  n_C, n_H, n_W)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    da_prev -- gradient of the cost with respect to the input of the conv layer (a_prev),
               numpy array of shape (m, n_C_prev, n_H_prev, n_W_prev)
    dweight -- gradient of the cost with respect to the weights of the conv layer (weight)
          numpy array of shape (n_C_prev, n_C, f, f)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, n_C, 1, 1)
    """

    # Retrieve information from "cache"
    (a_prev, weight, b, hparameters) = cache

    # Retrieve dimensions from A_prev's shape
    (m, n_C_prev, n_H_prev, n_W_prev) = a_prev.shape

    # Retrieve dimensions from W's shape
    (n_C_prev, n_C, f, f) = weight.shape

    # Retrieve information from "hparameters"
    stride = hparameters['stride']
    padding = hparameters['padding']

    # Retrieve dimensions from dz's shape
    (m, n_C, n_H, n_W) = dz.shape

    # Initialize dA_prev, dW, db with the correct shapes
    da_prev = np.zeros((m, n_C_prev, n_H_prev, n_W_prev))
    dweight = np.zeros((n_C_prev, n_C, f, f))
    db = np.zeros((1, n_C, 1, 1))

    # Pad a_prev and da_prev
    a_prev_pad = zero_pad(a_prev, pad)
    da_prev_pad = zero_pad(da_prev, pad)

    for i in range(m):

        # select ith training example from a_prev_pad and da_prev_pad
        a_prev_pad_in = a_prev_pad[i, :, :, :]
        da_prev_pad_in = da_prev_pad[i, :, :, :]

        for c in range(n_C):
            for h in range(n_H):
                for w in range(n_W):

                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Use the corners to define the slice from a_prev_pad_in
                    a_slice = a_prev_pad_in[:, vert_start:vert_end, horiz_start:horiz_end]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad_in[:, vert_start:vert_end, horiz_start:horiz_end] += weight[:, c, :, :] * dz[i, c, h, w]
                    dweight[:, c, :, :] += a_slice * dz[i, c, h, w]
                    db[:, c, :, :] += dz[i, c, h, w]

        # Set the ith training example's da_prev to the unpaded da_prev_pad_in (Hint: use X[pad:-pad, pad:-pad, :])
        da_prev[i, :, :, :] = da_prev_pad_in[:, padding:-padding, padding:-padding]

    # Making sure your output shape is correct
    assert(da_prev.shape == (m, n_C_prev, n_H_prev, n_W_prev))

    return da_prev, dweight, db


# pooling layer backward propagation
def create_mask_from_window(x):
    mask = (x == np.max(x))
    return mask


def distribute_value(dz, shape):
    (n_H, n_W) = shape
    average = dz / (n_H * n_W)
    a = average * np.ones(shape)
    return a


def pool_backward(da, cache, mode="max"):
    """
    implements the backward propagation of pooling
    :param da: gradient of cost respect to the output of the pooling layer, same shape as a
    :param cache: cache output from the forward propagation of the pooling layer,
                  contains the layer's input and hparameters
    :param mode: "max" or "average"
    :return: da_prev -- gradient of cost w.r.t the input of the pooling layer, same shape as a_prev
    """
    (a_prev, hparameters) = cache
    stride = hparameters["stride"]
    f = hparameters["f"]
    m, n_C, n_H, n_W = da.shape
    da_prev = np.zeros(a_prev.shape)
    for i in range(m):
        a_prev_in = a_prev[i]
        for c in range(n_C):
            for h in range(n_H):
                for w in range(n_W):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    if mode == "max":
                        a_prev_slice = a_prev_in[c, vert_start:vert_end, horiz_start:horiz_end]
                        mask = create_mask_from_window(a_prev_slice)
                        da_prev[i, c, vert_start:vert_end, horiz_start:horiz_end] += mask * da[i, c, h, w]

                    elif mode == "average":
                        descent = da[i, c, h, w]
                        shape = (f, f)
                        da_prev[i, c, vert_start:vert_end, horiz_start:horiz_end] += distribute_value(descent, shape)

    assert (da_prev.shape == a_prev.shape)
    return da_prev


# np.random.seed(1)
# A_prev = np.random.randn(2, 3, 4, 4)
# hparameters = {"stride": 2, "f": 2}
# A, cache = pool_forward(A_prev, hparameters)
# dA = np.random.randn(2, 3, 2, 2)
#
# dA_prev = pool_backward(dA, cache, mode="max")
# print("mode = max")
# print('mean of dA = ', np.mean(dA))
# print('dA_prev[1,1] = ', dA_prev[1, 1])
# print()
# dA_prev = pool_backward(dA, cache, mode="average")
# print("mode = average")
# print('mean of dA = ', np.mean(dA))
# print('dA_prev[1,1] = ', dA_prev[1, 1])











