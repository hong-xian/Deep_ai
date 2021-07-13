import numpy as np
import time
import random
import matplotlib.pyplot as plt

data = open("/Users/xiujing/Desktop/DL/deep_ai/ex5_1/dinos.txt", "r").read()
data = data.lower()
# converting to unordered and no-repeated list
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
# print(chars)
# print("total character: %d ，unique character: %d" % (data_size, vocab_size))

# construct a dictionary for corresponding character and index
char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
# print(char_to_ix)
# print(ix_to_char)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def clip(gradients, maxvalue):
    """
    using for clip the gradients to [-maxvalue, maxvalue]
    :param gradients: dictionary containing: "dw_aa", "dw_ax", "dw_ya", "db", "db_y"
    :param maxvalue: threshold
    :return: clipped gradients
    """
    for gradient in gradients.values():
        np.clip(gradient, -maxvalue, maxvalue, out=gradient)
    return gradients


# np.random.seed(3)
# dw_ax = np.random.randn(5, 3)*10
# dw_aa = np.random.randn(5, 5)*10
# dw_ya = np.random.randn(2, 5)*10
# db = np.random.randn(5, 1)*10
# db_y = np.random.randn(2, 1)*10
# gradients = {"dw_ax": dw_ax, "dw_aa": dw_aa, "dw_ya": dw_ya, "db": db, "db_y": db_y}
# print(gradients["dw_ax"])
# gradients = clip(gradients, 10)
# print("gradients[\"dw_aa\"][1][2] =", gradients["dw_aa"][1][2])
# print("gradients[\"dw_ax\"][3][1] =", gradients["dw_ax"][3][1])
# print("gradients[\"dw_ya\"][1][2] =", gradients["dw_ya"][1][2])
# print("gradients[\"db\"][4] =", gradients["db"][4])
# print("gradients[\"db_y\"][1] =", gradients["db_y"][1])


def sample(parameters, char_index, seed):
    """
    sampling according to the input probabilistic distribution of the rnn output
    :param parameters: dictionary containing "dw_aa", "dw_ax", "dw_ya", "db", "db_y"
    :param char_index: dictionary of character to index
    :param seed: random seed
    :return:
        indices -- a list of length n containing the indices of the sampled characters.
    """
    w_aa, w_ax, w_ya, b_y, b = (parameters['w_aa'], parameters['w_ax'],
                                parameters['w_ya'], parameters['b_y'], parameters['b'])
    vocab_size = b_y.shape[0]
    n_a = w_aa.shape[1]

    # step 1: create the one-hot vector for the first character
    x = np.zeros((vocab_size, 1))
    # initialize a_prev as zeros
    a_prev = np.zeros((n_a, 1))

    # Create an empty list of indices, this is the list which will contain
    # the list of indices of the characters to generate
    indices = []
    # Idx is a flag to detect a newline character, we initialize it to -1
    idx = -1
    counter = 0
    newline_character = char_index["\n"]

    while idx != newline_character and counter < 50:
        # forward propagation
        a = np.tanh(np.matmul(w_ax, x) + np.matmul(w_aa, a_prev) + b)
        z = np.matmul(w_ya, a) + b_y
        y = softmax(z)

        # for grading purposes
        np.random.seed(counter + seed)

        # step 3: Sample the index of a character within the vocabulary from the probability distribution y
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())
        indices.append(idx)

        # Step 4: Overwrite the input character as the one corresponding to the sampled index.
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        a_prev = a
        counter += 1

        if counter == 50:
            indices.append(char_index["\n"])

    return indices


# # sample example
# np.random.seed(3)
# n_a = 100
# w_ax, w_aa, w_ya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
# b, b_y = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
# parameters = {"w_ax": w_ax, "w_aa": w_aa, "w_ya": w_ya, "b": b, "b_y": b_y}
# indices = sample(parameters, char_to_ix, 0)
# print(len(indices))
# print("Sampling:")
# print("list of sampled indices:", indices)
# print("list of sampled characters:", [ix_to_char[i] for i in indices])


def print_sample(sample_ix, index_char):
    txt = "".join(index_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]
    print("%s" % (txt, ), end="")


def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001


def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0/vocab_size) * seq_length


def initialize_parameters(n_a, n_x, n_y):
    np.random.seed(1)
    w_ax = np.random.randn(n_a, n_x) * 0.01  # input to hidden
    w_aa = np.random.randn(n_a, n_a) * 0.01  # hidden to hidden
    w_ya = np.random.randn(n_y, n_a) * 0.01  # hidden to output
    b = np.zeros((n_a, 1))  # hidden bias
    b_y = np.zeros((n_y, 1))  # output bias

    parameters = {"w_ax": w_ax, "w_aa": w_aa, "w_ya": w_ya, "b": b, "b_y": b_y}

    return parameters


def rnn_step_forward(parameters, a_prev, x):
    w_aa, w_ax, w_ya, b_y, b = parameters['w_aa'], parameters['w_ax'], parameters['w_ya'], \
                               parameters['b_y'], parameters['b']
    a_next = np.tanh(np.dot(w_ax, x) + np.dot(w_aa, a_prev) + b)
    p_t = softmax(np.dot(w_ya, a_next) + b_y)
    return a_next, p_t


def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    gradients['dw_ya'] += np.matmul(dy, a.T)
    gradients['db_y'] += dy
    da = np.matmul(parameters['w_ya'].T, dy) + gradients['da_next']  # backprop into h
    da_raw = (1 - a * a) * da  # backprop through tanh
    gradients['db'] += da_raw
    gradients['dw_ax'] += np.matmul(da_raw, x.T)
    gradients['dw_aa'] += np.matmul(da_raw, a_prev.T)
    gradients['da_next'] = np.matmul(parameters['w_aa'].T, da_raw)
    return gradients


def update_parameters(parameters, gradients, lr):
    parameters['w_ax'] += -lr * gradients['dw_ax']
    parameters['w_aa'] += -lr * gradients['dw_aa']
    parameters['w_ya'] += -lr * gradients['dw_ya']
    parameters['b'] += -lr * gradients['db']
    parameters['b_y'] += -lr * gradients['db_y']

    return parameters


def rnn_forward(X, Y, a_0, parameters, vocab_size=27):
    """
    implement rnn forward propagation
    :param X: input data, list
    :param Y: output data, list, len(Y)=len(X)
    :param a_0: initial hidden state
    :param parameters: python dictionary containing parameters
    :param vocab_size: size of vocal size
    :return:
        loss -- loss value
        cache -- tuple of values needed for backward propagation
    """
    x, a, y_hat = {}, {}, {}
    a[-1] = np.copy(a_0)
    loss = 0

    for t in range(len(X)):
        # Set x[t] to be the one-hot vector representation of the t-th character in X
        # if X[t] == None, we just have x[t]=0. This is used to set the input for the first time-step to the zero vector
        x[t] = np.zeros((vocab_size, 1))
        if X[t]:
            x[t][X[t]] = 1

        a[t], y_hat[t] = rnn_step_forward(parameters, a[t-1], x[t])
        loss += -np.log(y_hat[t][Y[t]])

    cache = (y_hat, a, x)
    return loss, cache


def rnn_backward(X, Y, parameters, cache):
    """
    implement rnn backward propagation
    :param X: input data, list
    :param Y: output data, list, len(Y)=len(X)
    :param parameters: python dictionary containing parameters
    :param cache: values for backward pass
    :return:
        gradients -- gradient dictionary
        a -- dictionary of hidden state
    """
    gradients = {}
    (y_hat, a, x) = cache
    w_aa, w_ax, w_ya, b, b_y = (parameters['w_aa'], parameters['w_ax'], parameters['w_ya'],
                                parameters['b'], parameters['b_y'])
    # initialize gradient values
    gradients['dw_ax'], gradients['dw_aa'], gradients['dw_ya'] = (np.zeros_like(w_ax), np.zeros_like(w_aa),
                                                                  np.zeros_like(w_ya))
    gradients['db'], gradients['db_y'] = np.zeros_like(b), np.zeros_like(b_y)
    gradients['da_next'] = np.zeros_like(a[0])
    for t in reversed(range(len(X))):
        # softmax derivative with cross entropy
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])

    return gradients, a


def optimize(X, Y, a_prev, parameters, learning_rate=0.01):
    """
    Execute one step of the optimization to train the model.
    :param X: list of integers, where each integer is a number that maps to a character in the vocabulary
    :param Y: list of integer, exactly the same as X but shifted one index to the left
    :param a_prev: previous hidden state
    :param parameters: python dictionary containing:
                        w_ax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        w_aa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        w_ya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        b_y -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    :param learning_rate: learning rate for model
    :return:
            loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dw_ax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dw_aa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dw_ya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        db_y -- Gradients of output bias vector, of shape (n_y, 1)
                        a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    gradients = clip(gradients, 5)
    parameters = update_parameters(parameters, gradients, learning_rate)
    return loss, parameters, a[len(X)-1]


# np.random.seed(1)
# n_a = 100
# a_prev = np.random.randn(n_a, 1)
# w_ax, w_aa, w_ya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
# b, b_y = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
# parameters = {"w_ax": w_ax, "w_aa": w_aa, "w_ya": w_ya, "b": b, "b_y": b_y}
# X = [12, 3, 5, 11, 22, 3]
# Y = [4, 14, 11, 22, 25, 26]
# loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate=0.01)
# print("Loss =", loss)
# print("gradients[\"dWaa\"][1][2] =", gradients["dw_aa"][1][2])
# print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dw_ax"]))
# print("gradients[\"dWya\"][1][2] =", gradients["dw_ya"][1][2])
# print("gradients[\"db\"][4] =", gradients["db"][4])
# print("gradients[\"dby\"][1] =", gradients["db_y"][1])
# print("a_last[4] =", a_last[4])


def model(index_char, char_index, num_iteration=10000, n_a=100, dino_name=7, vocab_size=27):
    """
    Trains the model and generates dinosaur names
    :param index_char: dictionary that maps the index to a character
    :param char_index: dictionary that maps a character to an index
    :param num_iteration: number of iterations to train the model
    :param n_a: number of units of the RNN cell
    :param dino_name: number of dinosaur names you want to sample at each iteration
    :param vocab_size: number of unique characters found in the text, size of the vocabulary
    :return:
        parameters -- learned parameters
    """
    n_x, n_y = vocab_size, vocab_size
    parameters = initialize_parameters(n_a, n_x, n_y)
    losses = []
    loss = get_initial_loss(vocab_size, dino_name)

    # Build list of all dinosaur names (training examples)
    with open("/Users/xiujing/Desktop/DL/deep_ai/ex5_1/dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)

    # Initialize the hidden state of rnn
    a_prev = np.zeros((n_a, 1))

    for j in range(num_iteration):
        # define one training example (X,Y)
        index = j % len(examples)
        X = [None] + [char_index[ch] for ch in examples[index]]
        Y = X[1:] + [char_index["\n"]]
        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        curr_loss, parameters, a_prev = optimize(X, Y, a_prev, parameters)
        a_prev = np.zeros((n_a, 1))

        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training
        loss = smooth(loss, curr_loss)
        losses.append(loss)
        if j % 2000 == 0:
            print("Iteration: %d, Loss: %f" % (j, loss))
            seed = 0
            for name in range(dino_name):
                sample_indices = sample(parameters, char_index, seed)
                print_sample(sample_indices, index_char)
                seed += 1
            print("\n")
    return parameters, losses


start_time = time.time()
num_iterations = 30000
parameters, loss = model(ix_to_char, char_to_ix, num_iteration=num_iterations)
end_time = time.time()
train_time = end_time - start_time
print("Execute：" + str(int(train_time)) + " seconds ")
plt.plot(range(num_iterations), loss)
plt.show()


# 没有理解optimize函数返回最后隐藏状态的作用，每次将隐藏状态初始化为0，结果仍然合理。

