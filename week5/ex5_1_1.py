import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def initialize_adam(parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["w" + str(l)] = wl
                    parameters["b" + str(l)] = bl

    Returns:
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dw" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dw" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    n = len(parameters) // 2
    # number of layers in the neural networks
    v = {}
    s = {}

    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for i in range(n):
        v["dw" + str(i+1)] = np.zeros(parameters["w" + str(i+1)].shape)
        v["db" + str(i+1)] = np.zeros(parameters["b" + str(i+1)].shape)
        s["dw" + str(i+1)] = np.zeros(parameters["w" + str(i+1)].shape)
        s["db" + str(i+1)] = np.zeros(parameters["b" + str(i+1)].shape)
    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999,  epsilon=1e-8):
    """
    Update parameters using Adam

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['w' + str(l)] = wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dw' + str(l)] = dwl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates
    beta2 -- Exponential decay hyperparameter for the second moment estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """

    n = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary

    # Perform Adam update on all parameters
    for i in range(n):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dw" + str(i+1)] = beta1 * v["dw" + str(i+1)] + (1 - beta1) * grads["dw" + str(i+1)]
        v["db" + str(i+1)] = beta1 * v["db" + str(i+1)] + (1 - beta1) * grads["db" + str(i+1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dw" + str(i+1)] = v["dw" + str(i+1)] / (1 - beta1**t)
        v_corrected["db" + str(i+1)] = v["db" + str(i+1)] / (1 - beta1**t)

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dw" + str(i+1)] = beta2 * s["dw" + str(i+1)] + (1 - beta2) * (grads["dw" + str(i+1)] ** 2)
        s["db" + str(i+1)] = beta2 * s["db" + str(i+1)] + (1 - beta2) * (grads["db" + str(i+1)] ** 2)

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dw" + str(l+1)] = s["dw" + str(i+1)] / (1 - beta2 ** t)
        s_corrected["db" + str(l+1)] = s["db" + str(i+1)] / (1 - beta2 ** t)

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon".
        # Output: "parameters".
        parameters["w" + str(l+1)] = (parameters["w" + str(i+1)] - learning_rate * v_corrected["dw" + str(i+1)]
                                      / np.sqrt(s_corrected["dw" + str(i+1)] + epsilon))
        parameters["b" + str(l+1)] = (parameters["b" + str(i+1)] - learning_rate * v_corrected["db" + str(i+1)]
                                      / np.sqrt(s_corrected["db" + str(i+1)] + epsilon))

    return parameters, v, s


def rnn_cell_forward(x_t, a_prev, parameters):
    """
    implement a single forward step of RNN-cell
    :param x_t: input data at time step t, shape(n_x, m); m denote the sample size
    :param a_prev: hidden state at time step t-1, shape(n_a, m)
    :param parameters: python dictionary
                        w_ax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        w_aa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        w_ya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b_a --  Bias, numpy array of shape (n_a, 1)
                        b_y -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    :return:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at time step "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """
    w_ax = parameters["w_ax"]
    w_aa = parameters["w_aa"]
    w_ya = parameters["w_ya"]
    b_a = parameters["b_a"]
    b_y = parameters["b_y"]

    a_next = np.tanh(np.matmul(w_aa, a_prev) + np.matmul(w_ax, x_t) + b_a)
    yt_pred = softmax(np.matmul(w_ya, a_next) + b_y)

    cache = (a_next, a_prev, x_t, parameters)
    # store values needed for back propagation
    return a_next, yt_pred, cache

# test for rnn_cell_forward
# np.random.seed(1)
# x_t = np.random.randn(3, 10)
# a_prev = np.random.randn(5, 10)
# w_aa = np.random.randn(5, 5)
# w_ax = np.random.randn(5, 3)
# w_ya = np.random.randn(2, 5)
# b_a = np.random.randn(5, 1)
# b_y = np.random.randn(2, 1)
# parameters = {"w_aa": w_aa, "w_ax": w_ax, "w_ya": w_ya, "b_a": b_a, "b_y": b_y}
# a_next, yt_pred, cache = rnn_cell_forward(x_t, a_prev, parameters)
# print("a_next[4] = ", a_next[4])
# print("a_next.shape = ", a_next.shape)
# print("yt_pred[1] =", yt_pred[1])
# print("yt_pred.shape = ", yt_pred.shape)


def rnn_forward(x, a_0, parameters):
    """
        Implement the forward propagation of the recurrent neural network described in Figure (3).

        Arguments:
        x -- Input data for every time-step, of shape (n_x, m, len_x).
        a_0 -- Initial hidden state, of shape (n_a, m)
        parameters -- python dictionary containing:
                            w_aa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                            w_ax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                            w_ya -- Weight matrix relating the hidden-state to the output,
                                    numpy array of shape (n_y, n_a)
                            b_a --  Bias numpy array of shape (n_a, 1)
                            b_y -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

        Returns:
        a -- Hidden states for every time-step, numpy array of shape (n_a, m, len_x)
        y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, len_x)
        caches -- tuple of values needed for the backward pass, contains (list of caches, x)
        """
    caches = []

    # retrieve dim from shape of x and parameters["w_ya"]
    n_x, m, len_x = x.shape
    n_y, n_a = parameters["w_ya"].shape

    # initialize "a" and "y_pred"
    a = np.zeros((n_a, m, len_x))
    y_pred = np.zeros((n_y, m, len_x))

    a_next = a_0
    for t in range(len_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        a[:, :, t] = a_next
        y_pred[:, :, t] = yt_pred
        caches.append(cache)

    caches = (caches, x)
    return a, y_pred, caches


# np.random.seed(1)
# x = np.random.randn(3, 10, 4)
# a0 = np.random.randn(5, 10)
# w_aa = np.random.randn(5, 5)
# w_ax = np.random.randn(5, 3)
# w_ya = np.random.randn(2, 5)
# b_a = np.random.randn(5, 1)
# b_y = np.random.randn(2, 1)
# parameters = {"w_aa": w_aa, "w_ax": w_ax, "w_ya": w_ya, "b_a": b_a, "b_y": b_y}
#
# a, y_pred, caches = rnn_forward(x, a0, parameters)
# print("a[4][1] = ", a[4][1])
# print("a.shape = ", a.shape)
# print("y_pred[1][3] =", y_pred[1][3])
# print("y_pred.shape = ", y_pred.shape)
# print("caches[1][1][3] =", caches[1][1][3])
# print("len(caches) = ", len(caches))

def rnn_cell_backward(da_next, cache):
    """
    implement backward pass for a RNN cell (single time step)
    :param da_next: gradient of loss with respect to a_next
    :param cache: python dictionary contains all values of (out of rnn_forward)
    :return:
    gradients -- python dictionary containing:
                    dx -- gradient of input data, of shape (n_x, m)
                    da_prev -- gradient of previous hidden state, shape of (n_a, m)
                    dw_ax -- gradients of input to hidden weight matrix, shape of (n_x, n_a)
                    dw_aa -- gradients of hidden to hidden weight matrix, shape of (n_a, n_a)
                    db_a --gradients of bias, shape of (n_a, 1)
    """

    a_next, a_prev, x_t, parameters = cache
    w_ax = parameters["w_ax"]
    w_aa = parameters["w_aa"]
    w_ya = parameters["w_ya"]
    b_a = parameters["b_a"]
    b_y = parameters["b_y"]

    dtanh = (1 - np.square(a_next)) * da_next
    dx_t = np.matmul(w_ax.T, dtanh)
    dw_ax = np.matmul(dtanh, x_t.T)

    da_prev = np.matmul(w_aa.T, dtanh)
    dw_aa = np.matmul(dtanh, a_prev.T)
    # b_a involving a batch
    db_a = np.sum(dtanh, keepdims=True, axis=-1)

    gradients = {"dx_t": dx_t, "dw_ax": dw_ax, "da_prev": da_prev, "dw_aa": dw_aa, "db_a": db_a}

    return gradients


# np.random.seed(1)
# x_t = np.random.randn(3, 10)
# a_prev = np.random.randn(5, 10)
# w_ax = np.random.randn(5, 3)
# w_aa = np.random.randn(5, 5)
# w_ya = np.random.randn(2, 5)
# b_a = np.random.randn(5, 1)
# b_y = np.random.randn(2, 1)
# parameters = {"w_ax": w_ax, "w_aa": w_aa, "w_ya": w_ya, "b_a": b_a, "b_y": b_y}
#
# a_next, y_t, cache = rnn_cell_forward(x_t, a_prev, parameters)
#
# da_next = np.random.randn(5, 10)
# gradients = rnn_cell_backward(da_next, cache)
# print("gradients[\"dx_t\"][1][2] =", gradients["dx_t"][1][2])
# print("gradients[\"dx_t\"].shape =", gradients["dx_t"].shape)
# print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
# print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
# print("gradients[\"dw_ax\"][3][1] =", gradients["dw_ax"][3][1])
# print("gradients[\"dw_ax\"].shape =", gradients["dw_ax"].shape)
# print("gradients[\"dw_aa\"][1][2] =", gradients["dw_aa"][1][2])
# print("gradients[\"dw_aa\"].shape =", gradients["dw_aa"].shape)
# print("gradients[\"db_a\"][4] =", gradients["db_a"][4])
# print("gradients[\"db_a\"].shape =", gradients["db_a"].shape)

def rnn_backward(da, caches):
    """
    implement the backward pass for RNN over an entire sequence of input data
    :param da: Upstream gradients of all hidden states, of shape (n_a, m, len_x)
    :param caches: tuple containing information from the forward pass (rnn_forward)
    :return:
        gradients -- python dictionary containing:
                        dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, len_x)
                        da_0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
                        dw_ax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
                        dw_aa -- Gradient w.r.t the hidden state's weight matrix, numpy-array of shape (n_a, n_a)
                        db_a -- Gradient w.r.t the bias, of shape (n_a, 1)
    """
    caches, x = caches
    a_1, a_0, x_1, parameters = caches[0]

    # retrieve dim
    n_a, m, len_x = da.shape
    n_x, m = x_1.shape

    # initialize gradients
    dx = np.zeros([n_x, m, len_x])
    dw_ax = np.zeros([n_a, n_x])
    dw_aa = np.zeros([n_a, n_a])
    db_a = np.zeros([n_a, 1])
    da_prev_t = np.zeros([n_a, m])

    for t in reversed(range(len_x)):
        gradients = rnn_cell_backward(da[:, :, t] + da_prev_t, caches[t])
        dx_t, dw_ax_t, da_prev_t, dw_aa_t, db_a_t = (gradients["dx_t"], gradients["dw_ax"],
                                                     gradients["da_prev"], gradients["dw_aa"], gradients["db_a"])
        dx[:, :, t] = dx_t
        dw_aa += dw_aa_t
        dw_ax += dw_ax_t
        db_a += db_a_t
    da_0 = da_prev_t
    gradients = {"dx": dx, "dw_ax": dw_ax, "da_0": da_0, "dw_aa": dw_aa, "db_a": db_a}
    return gradients


# np.random.seed(1)
# x = np.random.randn(3, 10, 4)
# a_0 = np.random.randn(5, 10)
# w_ax = np.random.randn(5, 3)
# w_aa = np.random.randn(5, 5)
# w_ya = np.random.randn(2, 5)
# b_a = np.random.randn(5, 1)
# b_y = np.random.randn(2, 1)
# parameters = {"w_ax": w_ax, "w_aa": w_aa, "w_ya": w_ya, "b_a": b_a, "b_y": b_y}
# a, y, caches = rnn_forward(x, a_0, parameters)
# da = np.random.randn(5, 10, 4)
# gradients = rnn_backward(da, caches)
#
# print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
# print("gradients[\"dx\"].shape =", gradients["dx"].shape)
# print("gradients[\"da_0\"][2][3] =", gradients["da_0"][2][3])
# print("gradients[\"da_0\"].shape =", gradients["da_0"].shape)
# print("gradients[\"dw_ax\"][3][1] =", gradients["dw_ax"][3][1])
# print("gradients[\"dw_ax\"].shape =", gradients["dw_ax"].shape)
# print("gradients[\"dw_aa\"][1][2] =", gradients["dw_aa"][1][2])
# print("gradients[\"dw_aa\"].shape =", gradients["dw_aa"].shape)
# print("gradients[\"db_a\"][4] =", gradients["db_a"][4])
# print("gradients[\"db_a\"].shape =", gradients["db_a"].shape)




