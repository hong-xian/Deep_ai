# construct long short term memory network
from ex5_1_1 import sigmoid, softmax
import numpy as np


def lstm_cell_forward(x_t, a_prev, c_prev, parameters):
    """
    implement a single time step forward propagation of LSTM
    :param x_t: input data at time step t, shape(n_x, m); m denote the sample size
    :param a_prev: hidden state at time step t-1, shape(n_a, m)
    :param c_prev: memory state at time step t-1, shape(n_a, m)
    :param parameters: python dictionary containing:
                        w_f -- weight matrix of forget gate, shape of (n_a, n_a + n_x)
                        b_f -- bias of forget gate, shape of (n_a, 1)
                        w_u -- weight matrix of update gate, shape of (n_a, n_a + n_x)
                        b_u -- bias of update gate, shape of (n_a, 1)
                        w_c -- weight matrix of first "tanh" for candidate memory state of t, shape of (n_a, n_a + n_x)
                        b_c -- bias of first "tanh" for candidate memory state of t, shape of (n_a, 1)
                        w_o -- weight matrix of output gate, shape of (n_a, n_a + n_x)
                        b_o -- bias of output gate, shape of (n_a, 1)
                        w_y -- weight matrix of prediction, shape of (n_y, n_a)
                        b_y -- bias of prediction, shape of (n_y, 1)

    :return:
        a_next -- next hidden state, shape of (n_a, m)
        c_next -- next memory state, shape of (n_a, m)
        yt_pred -- prediction of time step t, shape of (n_y, m)
        cache -- containing values for back propagation, including(a_next, a_prev, c_prev, x_t, parameters)
    """
    w_f = parameters["w_f"]
    b_f = parameters["b_f"]
    w_u = parameters["w_u"]
    b_u = parameters["b_u"]
    w_o = parameters["w_o"]
    b_o = parameters["b_o"]
    w_c = parameters["w_c"]
    b_c = parameters["b_c"]
    w_y = parameters["w_y"]
    b_y = parameters["b_y"]

    n_x, m = x_t.shape
    m_y, n_a = w_y.shape
    # concatenate a_prev and x_t
    contact = np.zeros([n_a + n_x, m])
    contact[:n_a, :] = a_prev
    contact[n_a:, :] = x_t

    # compute gate coefficient of gates
    f_t = sigmoid(np.matmul(w_f, contact) + b_f)
    u_t = sigmoid(np.matmul(w_u, contact) + b_u)
    o_t = sigmoid(np.matmul(w_o, contact) + b_o)
    c_t_hat = np.tanh(np.matmul(w_c, contact) + b_c)
    c_next = f_t * c_prev + u_t * c_t_hat
    a_next = np.tanh(o_t * c_next)
    yt_pred = softmax(np.matmul(w_y, a_next) + b_y)

    cache = (a_next, c_next, a_prev, c_prev, f_t, u_t, o_t, c_t_hat, x_t, parameters)

    return a_next, c_next, yt_pred, cache


# np.random.seed(1)
# x_t = np.random.randn(3, 10)
# a_prev = np.random.randn(5, 10)
# c_prev = np.random.randn(5, 10)
# w_f = np.random.randn(5, 5+3)
# b_f = np.random.randn(5, 1)
# w_u = np.random.randn(5, 5+3)
# b_u = np.random.randn(5, 1)
# w_o = np.random.randn(5, 5+3)
# b_o = np.random.randn(5, 1)
# w_c = np.random.randn(5, 5+3)
# b_c = np.random.randn(5, 1)
# w_y = np.random.randn(2, 5)
# b_y = np.random.randn(2, 1)
#
# parameters = {"w_f": w_f, "w_u": w_u, "w_o": w_o, "w_c": w_c, "w_y": w_y,
#               "b_f": b_f, "b_u": b_u, "b_o": b_o, "b_c": b_c, "b_y": b_y}
#
# a_next, c_next, y_t, cache = lstm_cell_forward(x_t, a_prev, c_prev, parameters)
# print("a_next[4] = ", a_next[4])
# print("a_next.shape = ", a_next.shape)
# print("c_next[2] = ", c_next[2])
# print("c_next.shape = ", c_next.shape)
# print("y_t[1] =", y_t[1])
# print("y_t.shape = ", y_t.shape)
# print("cache[1][3] =", cache[1][3])
# print("len(cache) = ", len(cache))


# forward propagation
def lstm_forward(x, a_0, parameters):
    """
    implement forward propagation of lstm network
    :param x: input data of all time step, shape of (n_x, m, len_x)
    :param a_0: initial hidden state, shape of (n_a, m)
    :param parameters:python dictionary, containing:
                        w_f -- weight matrix of forget gate, shape of (n_a, n_a + n_x)
                        b_f -- bias of forget gate, shape of (n_a, 1)
                        w_u -- weight matrix of update gate, shape of (n_a, n_a + n_x)
                        b_u -- bias of update gate, shape of (n_a, 1)
                        w_c -- weight matrix of first "tanh" for candidate memory state of t, shape of (n_a, n_a + n_x)
                        b_c -- bias of first "tanh" for candidate memory state of t, shape of (n_a, 1)
                        w_o -- weight matrix of output gate, shape of (n_a, n_a + n_x)
                        b_o -- bias of output gate, shape of (n_a, 1)
                        w_y -- weight matrix of prediction, shape of (n_y, n_a)
                        b_y -- bias of prediction, shape of (n_y, 1)
    :return:
            a -- hidden state of all time step, shape of (n_a, m, len_x)
            y -- prediction of all time step, shape of (n_y, m, len_x)
            caches -- contain all values for backward propagation, (list of all caches, x)
    """
    caches = []
    n_x, m, len_x = x.shape
    n_y, n_a = w_y.shape

    # initialize a, c, y
    a = np.zeros([n_a, m, len_x])
    c = np.zeros([n_a, m, len_x])
    y = np.zeros([n_y, m, len_x])

    # initialize a_next, c_next
    a_next = a_0
    c_next = np.zeros([n_a, m])

    for t in range(len_x):
        a_next, c_next, yt_pred, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)
        a[:, :, t] = a_next
        y[:, :, t] = yt_pred
        c[:, :, t] = c_next
        caches.append(cache)

    caches = (caches, x)
    return a, y, c, caches


# np.random.seed(1)
# x = np.random.randn(3, 10, 7)
# a_0 = np.random.randn(5, 10)
# w_f = np.random.randn(5, 5+3)
# b_f = np.random.randn(5, 1)
# w_u = np.random.randn(5, 5+3)
# b_u = np.random.randn(5, 1)
# w_o = np.random.randn(5, 5+3)
# b_o = np.random.randn(5, 1)
# w_c = np.random.randn(5, 5+3)
# b_c = np.random.randn(5, 1)
# w_y = np.random.randn(2, 5)
# b_y = np.random.randn(2, 1)
#
# parameters = {"w_f": w_f, "w_u": w_u, "w_o": w_o, "w_c": w_c, "w_y": w_y,
#               "b_f": b_f, "b_u": b_u, "b_o": b_o, "b_c": b_c, "b_y": b_y}
# a, y, c, caches = lstm_forward(x, a_0, parameters)
# print("a[4][3][6] = ", a[4][3][6])
# print("a.shape = ", a.shape)
# print("y[1][4][3] =", y[1][4][3])
# print("y.shape = ", y.shape)
# print("caches[1][1[1]] =", caches[1][1][1])
# print("c[1][2][1]", c[1][2][1])
# print("len(caches) = ", len(caches))


def lstm_cell_backward(da_next, dc_next, cache):
    """
    single step backward propagation of lstm cell
    :param da_next: gradient of next hidden state, shape of (n_a, m)
    :param dc_next: gradient of next memory state, shape of (n_a, m)
    :param cache: values from forward propagation
    :return:
    gradient: python dictionary containing:
                dx_t -- gradient of input data, shape of (n_x, m)
                da_prev -- gradient of previous hidden state, shape of (n_a, m)
                dc_prev -- gradient of previous memory state, shape of (n_a, m)
                dw_f -- gradient w.r.t weight of forget gate, shape of (n_a, n_a + n_x)
                dw_u -- gradient w.r.t weight of update gate, shape of (n_a, n_a + n_x)
                dw_c -- gradient w.r.t weight of memory gate, shape of (n_a, n_a + n_x)
                dw_o -- gradient w.r.t weight of output gate, shape of (n_a, n_a + n_x)
                db_f -- gradient w.r.t. bias of the forget gate, of shape (n_a, 1)
                db_u -- gradient w.r.t. bias of the update gate, of shape (n_a, 1)
                db_c -- gradient w.r.t. bias of the memory gate, of shape (n_a, 1)
                db_o -- gradient w.r.t. bias of the output gate, of shape (n_a, 1)
    """
    (a_next, c_next, a_prev, c_prev, f_t, u_t, o_t, c_t_hat, x_t, parameters) = cache
    n_a, m = a_next.shape
    # derivative of gate
    dc_next += da_next * o_t * (1 - np.square(c_next))
    do_t = da_next * np.tanh(c_next)
    dc_t_hat = dc_next * u_t
    df_t = dc_next * c_prev
    du_t = dc_next * c_t_hat

    # derivative of parameters
    concat = np.concatenate((a_prev, x_t), axis=0).T
    dw_o = np.matmul(do_t * o_t * (1-o_t), concat)
    dw_f = np.matmul(df_t * f_t * (1 - f_t), concat)
    dw_u = np.matmul(du_t * u_t * (1 - u_t), concat)
    dw_c = np.matmul(dc_t_hat * u_t * (1 - np.square(c_t_hat)), concat)
    db_o = np.sum(do_t * o_t * (1-o_t), axis=1, keepdims=True)
    db_f = np.sum(df_t * f_t * (1 - f_t), axis=1, keepdims=True)
    db_u = np.sum(du_t * u_t * (1 - u_t), axis=1, keepdims=True)
    db_c = np.sum(dc_t_hat * u_t * (1 - np.square(c_t_hat)), axis=1, keepdims=True)

    dc_prev = dc_next * f_t
    da_prev = (np.matmul(parameters["w_o"][:, :n_a].T, do_t * o_t * (1-o_t)) +
               np.matmul(parameters["w_f"][:, :n_a].T, df_t * f_t * (1-f_t)) +
               np.matmul(parameters["w_u"][:, :n_a].T, du_t * u_t * (1-u_t)) +
               np.dot(parameters["w_c"][:, :n_a].T, dc_t_hat * (1 - np.square(c_t_hat)), ))
    dx_t = (np.matmul(parameters["w_o"][:, n_a:].T, do_t * o_t * (1 - o_t)) +
            np.matmul(parameters["w_f"][:, n_a:].T, df_t * f_t * (1 - f_t)) +
            np.matmul(parameters["w_u"][:, n_a:].T, du_t * u_t * (1 - u_t)) +
            np.dot(parameters["w_c"][:, n_a:].T, dc_t_hat * (1 - np.square(c_t_hat)), ))
    gradients = {"dx_t": dx_t, "da_prev": da_prev, "dc_prev": dc_prev, "dw_f": dw_f, "dw_o": dw_o,
                 "dw_u": dw_u, "dw_c": dw_c, "db_o": db_f, "db_f": db_u, "db_u": db_o, "db_c": db_c}
    return gradients


# np.random.seed(1)
# x_t = np.random.randn(3, 10)
# a_prev = np.random.randn(5, 10)
# c_prev = np.random.randn(5, 10)
# w_f = np.random.randn(5, 5+3)
# b_f = np.random.randn(5, 1)
# w_u = np.random.randn(5, 5+3)
# b_u = np.random.randn(5, 1)
# w_o = np.random.randn(5, 5+3)
# b_o = np.random.randn(5, 1)
# w_c = np.random.randn(5, 5+3)
# b_c = np.random.randn(5, 1)
# w_y = np.random.randn(2, 5)
# b_y = np.random.randn(2, 1)
#
# parameters = {"w_f": w_f, "w_u": w_u, "w_o": w_o, "w_c": w_c, "w_y": w_y,
#               "b_f": b_f, "b_u": b_u, "b_o": b_o, "b_c": b_c, "b_y": b_y}
#
# a_next, c_next, y_t, cache = lstm_cell_forward(x_t, a_prev, c_prev, parameters)
#
# da_next = np.random.randn(5, 10)
# dc_next = np.random.randn(5, 10)
# gradients = lstm_cell_backward(da_next, dc_next, cache)
# print("gradients[\"dxt\"][1][2] =", gradients["dx_t"][1][2])
# print("gradients[\"dxt\"].shape =", gradients["dx_t"].shape)
# print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
# print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
# print("gradients[\"dc_prev\"][2][3] =", gradients["dc_prev"][2][3])
# print("gradients[\"dc_prev\"].shape =", gradients["dc_prev"].shape)
# print("gradients[\"dWf\"][3][1] =", gradients["dw_f"][3][1])
# print("gradients[\"dWf\"].shape =", gradients["dw_f"].shape)
# print("gradients[\"dWu\"][1][2] =", gradients["dw_u"][1][2])
# print("gradients[\"dWu\"].shape =", gradients["dw_u"].shape)
# print("gradients[\"dWc\"][3][1] =", gradients["dw_c"][3][1])
# print("gradients[\"dWc\"].shape =", gradients["dw_c"].shape)
# print("gradients[\"dWo\"][1][2] =", gradients["dw_o"][1][2])
# print("gradients[\"dWo\"].shape =", gradients["dw_o"].shape)
# print("gradients[\"dbf\"][4] =", gradients["db_f"][4])
# print("gradients[\"dbf\"].shape =", gradients["db_f"].shape)
# print("gradients[\"dbu\"][4] =", gradients["db_u"][4])
# print("gradients[\"dbu\"].shape =", gradients["db_u"].shape)
# print("gradients[\"dbc\"][4] =", gradients["db_c"][4])
# print("gradients[\"dbc\"].shape =", gradients["db_c"].shape)
# print("gradients[\"dbo\"][4] =", gradients["db_o"][4])
# print("gradients[\"dbo\"].shape =", gradients["db_o"].shape)


def lstm_backward(da, caches):
    """
    implement backward propagation of lstm
    :param da: gradients of hidden state, shape of (n_a, m, len_x)
    :param caches: values of forward propagation
    :return:
        gradients containing:
                dx -- gradient of input data, shape of (n_x, m, len_x)
                da_0 -- gradient of previous hidden state, shape of (n_a, m)
                dw_f -- gradient w.r.t weight of forget gate, shape of (n_a, n_a + n_x)
                dw_u -- gradient w.r.t weight of update gate, shape of (n_a, n_a + n_x)
                dw_c -- gradient w.r.t weight of memory gate, shape of (n_a, n_a + n_x)
                dw_o -- gradient w.r.t weight of output gate, shape of (n_a, n_a + n_x)
                db_f -- gradient w.r.t. bias of the forget gate, of shape (n_a, 1)
                db_u -- gradient w.r.t. bias of the update gate, of shape (n_a, 1)
                db_c -- gradient w.r.t. bias of the memory gate, of shape (n_a, 1)
                db_o -- gradient w.r.t. bias of the output gate, of shape (n_a, 1)

    """
    caches, x = caches
    (a_1, c_1, a_0, c_0, f_1, u_1, o_1, c_1_hat,  x_1, parameters) = caches[0]
    n_a, m, len_x = da.shape
    n_x, m = x_1.shape

    # initialize the gradient
    dx = np.zeros((n_x, m, len_x))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dw_f = np.zeros((n_a, n_a + n_x))
    dw_u = np.zeros((n_a, n_a + n_x))
    dw_c = np.zeros((n_a, n_a + n_x))
    dw_o = np.zeros((n_a, n_a + n_x))
    db_f = np.zeros((n_a, 1))
    db_u = np.zeros((n_a, 1))
    db_c = np.zeros((n_a, 1))
    db_o = np.zeros((n_a, 1))

    # all time step
    for t in reversed(range(len_x)):
        gradient = lstm_cell_backward(da[:, :, t] + da_prevt, dc_prevt, caches[t])
        dx[:, :, t] = gradient['dx_t']
        dw_f = dw_f + gradient['dw_f']
        dw_u = dw_u + gradient['dw_u']
        dw_c = dw_c + gradient['dw_c']
        dw_o = dw_o + gradient['dw_o']
        db_f = db_f + gradient['db_f']
        db_u = db_u + gradient['db_u']
        db_c = db_c + gradient['db_c']
        db_o = db_o + gradient['db_o']
        da_prevt = gradient["da_prev"]
        dc_prevt = gradient["dc_prev"]
    da_0 = gradient["da_prev"]
    gradients = {"dx": dx, "da_0": da_0, "dw_f": dw_f, "dw_o": dw_o, "dw_u": dw_u,
                 "dw_c": dw_c, "db_o": db_f, "db_f": db_u, "db_u": db_o, "db_c": db_c}
    return gradients


# np.random.seed(1)
# x = np.random.randn(3, 10, 7)
# a_0 = np.random.randn(5, 10)
# w_f = np.random.randn(5, 5+3)
# b_f = np.random.randn(5, 1)
# w_u = np.random.randn(5, 5+3)
# b_u = np.random.randn(5, 1)
# w_o = np.random.randn(5, 5+3)
# b_o = np.random.randn(5, 1)
# w_c = np.random.randn(5, 5+3)
# b_c = np.random.randn(5, 1)
# w_y = np.random.randn(2, 5)
# b_y = np.random.randn(2, 1)
#
# parameters = {"w_f": w_f, "w_u": w_u, "w_o": w_o, "w_c": w_c, "w_y": w_y,
#               "b_f": b_f, "b_u": b_u, "b_o": b_o, "b_c": b_c, "b_y": b_y}
#
# a, y, c, caches = lstm_forward(x, a_0, parameters)
#
# da = np.random.randn(5, 10, 4)
# gradients = lstm_backward(da, caches)
#
# print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
# print("gradients[\"dx\"].shape =", gradients["dx"].shape)
# print("gradients[\"da0\"][2][3] =", gradients["da_0"][2][3])
# print("gradients[\"da0\"].shape =", gradients["da_0"].shape)
# print("gradients[\"dWf\"][3][1] =", gradients["dw_f"][3][1])
# print("gradients[\"dWf\"].shape =", gradients["dw_f"].shape)
# print("gradients[\"dWu\"][1][2] =", gradients["dw_u"][1][2])
# print("gradients[\"dWu\"].shape =", gradients["dw_u"].shape)
# print("gradients[\"dWc\"][3][1] =", gradients["dw_c"][3][1])
# print("gradients[\"dWc\"].shape =", gradients["dw_c"].shape)
# print("gradients[\"dWo\"][1][2] =", gradients["dw_o"][1][2])
# print("gradients[\"dWo\"].shape =", gradients["dw_o"].shape)
# print("gradients[\"dbf\"][4] =", gradients["db_f"][4])
# print("gradients[\"dbf\"].shape =", gradients["db_f"].shape)
# print("gradients[\"dbi\"][4] =", gradients["db_u"][4])
# print("gradients[\"dbi\"].shape =", gradients["db_u"].shape)
# print("gradients[\"dbc\"][4] =", gradients["db_c"][4])
# print("gradients[\"dbc\"].shape =", gradients["db_c"].shape)
# print("gradients[\"dbo\"][4] =", gradients["db_o"][4])
# print("gradients[\"dbo\"].shape =", gradients["db_o"].shape)



















