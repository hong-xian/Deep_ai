import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import sys
import io


class LstmPoem(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer):
        super(LstmPoem, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z):
        out, (hn, cn) = self.lstm(z)
        output = out[:, -1, :]
        output = self.linear(output)
        return output


def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))


def build_data(text, seq_len=40, stride=3):
    """
    Create a training set by scanning a window of size seq_len over the text corpus, with stride 3.
    Arguments:
    text -- string, corpus of Shakespearian poem
    seq_len -- sequence length, number of time-steps (or characters) in one training example
    stride -- how much the window shifts itself while scanning
    Returns:
        train_set -- list of training examples
        train_labels -- list of training labels
    """
    train_set = []
    train_labels = []
    for i in range(0, len(text) - seq_len, stride):
        train_set.append(text[i: i + seq_len])
        train_labels.append(text[i + seq_len])
    print('number of training examples:', len(train_set))
    return train_set, train_labels


def vectorization(sets, labels, n_x, char_indices, seq_len=40):
    """
    Convert sets and labels (lists) into arrays to be given to a recurrent neural network.
    Arguments:
    sets --
    labels --
    seq_len -- integer, sequence length
    Returns:
        vec_x -- array of shape (m, seq_len, len(chars))
        vec_y -- array of shape (m,)
    """
    m = len(sets)
    vec_x = np.zeros((m, seq_len, n_x), dtype=np.float32)
    vec_y = np.zeros((m,), dtype=np.int64)
    for i, sentence in enumerate(sets):
        for t, char in enumerate(sentence):
            vec_x[i, t, char_indices[char]] = 1
        vec_y[i] = char_indices[labels[i]]
    return vec_x, vec_y


class MyDataset(Dataset):
    def __init__(self, train_x, train_y):
        self.x_data = torch.from_numpy(train_x)
        self.y_data = torch.from_numpy(train_y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.shape[0]


def sample(pred, temperature=1.0):
    # helper function to sample an index from a probability array
    pred = pred.detach().numpy()
    pred = np.asarray(pred).astype('float64')
    pred = softmax(pred)
    # use temperature to control randomness, the smaller temp is, the lower entropy is
    pred = np.log(pred+1e-10) / temperature
    exp_pred = np.exp(pred)
    pred = exp_pred / np.sum(exp_pred)
    out = np.random.choice(range(len(chars)), p=pred.ravel())
    return out


def generate_sample(temperature=1.0):
    generated = ""
    usr_input = input("Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: ")
    sentence = usr_input.zfill(seq_length).lower()
    generated += usr_input

    print("\n\nHere is your poem: \n")
    sys.stdout.write(usr_input)

    for i in range(400):
        x_pred = np.zeros((1, seq_length, vocab_size), dtype=np.float32)
        for t, char in enumerate(sentence):
            if char != '0':
                x_pred[0, t, char_to_index[char]] = 1.

        x_tensor = torch.from_numpy(x_pred)
        prediction = model(x_tensor)
        next_index = sample(prediction, temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

        if next_char == '\n':
            continue


def train(epoch=10):
    model.train()
    for i in range(epoch):
        for j, (texts, targets) in enumerate(train_loader):
            outputs = model(texts)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (j+1) % 50 == 0:
                print("Epoch: %d, Batch: %d, Loss is: %.6f" % (i+1, j+1, loss.item()))

        scheduler.step()
        lr = scheduler.get_last_lr()
        print("Epoch: %d, lr: %f \n" % (i + 1, lr[0]))


print("Loading text data...")
raw_data = io.open('/Users/xiujing/Desktop/DL/deep_ai/ex5_1/shakespeare.txt', encoding='utf-8').read().lower()
print('corpus length:', len(raw_data))
seq_length = 40
chars = sorted(list(set(raw_data)))
vocab_size = len(chars)
char_to_index = dict((c, i) for i, c in enumerate(chars))
index_to_char = dict((i, c) for i, c in enumerate(chars))
print(char_to_index)
print("Creating training set...")
X, Y = build_data(raw_data, seq_length, stride=3)
print("Vectorizing training set...")
x, y = vectorization(X, Y, n_x=vocab_size, char_indices=char_to_index)

training_set = MyDataset(x, y)
train_loader = torch.utils.data.DataLoader(dataset=training_set,
                                           batch_size=128,
                                           shuffle=True)


model = LstmPoem(vocab_size, 100, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print("Loading model...")
train(epoch=20)
# torch.save(model.state_dict(), "/Users/xiujing/Desktop/DL/deep_ai/ex5_1/poem.pt")
# model.load_state_dict(torch.load("/Users/xiujing/Desktop/DL/deep_ai/ex5_1/poem.pt"))


print("\n Now you can generate your example:")
generate_sample()
