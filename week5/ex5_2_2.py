import csv
import emoji
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchsummary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("**********", device)
train_path = '/home/liushuang/PycharmProjects/lab/mydata/ex5_2/train_emoji.csv'
test_path = '/home/liushuang/PycharmProjects/lab/mydata/ex5_2/test.csv'
emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}


def read_csv(filename):
    phrase = []
    emojis = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            phrase.append(row[0])
            emojis.append(row[1])
    x = np.asarray(phrase)
    y = np.asarray(emojis, dtype=np.int64)
    return x, y


def label_to_emoji(label):
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)


def read_glove_vec(glove_file):
    with open(glove_file, "r") as f:
        words = []
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.append(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float32)

    i = 1
    word_to_index = {}
    index_to_word = {}
    for w in sorted(words):
        word_to_index[w] = i
        index_to_word[i] = w
        i += 1
    return word_to_index, index_to_word, word_to_vec_map


X_train, Y_train = read_csv(train_path)
# (132, )
X_test, Y_test = read_csv(test_path)
# (56, )
max_len = len(max(X_train, key=len).split())
# index = 3
# print(X_train[index], label_to_emoji(Y_train[index]))
word_to_indexes, index_to_words, word_to_vec_maps = read_glove_vec('/home/liushuang/PycharmProjects'
                                                                   '/lab/mydata/ex5_2/glove.6B.50d.txt')
# word = "cucumber"
# index = 113317
# print("the index of", word, "in the vocabulary is:", word_to_index[word])
# print("the", str(index) + "th word in the vocabulary is:", index_to_word[index])


def sentence_to_avg(sentence, word_to_vec_map):
    """
    converting words into glove vectors, and take average of these vectors
    :param sentence: string, one training example from X
    :param word_to_vec_map: dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    :return:
        avg -- average vector encoding information of sentence, shape of (50, )
    """
    words = sentence.lower().split()
    avg = np.zeros((50,), dtype=np.float32)
    for w in words:
        avg += word_to_vec_map[w]
    avg = avg / len(words)
    return avg


# average = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_maps)
# print("avg = ", average)


class Linear(nn.Module):
    def __init__(self, num_class):
        super(Linear, self).__init__()
        self.linear = nn.Linear(50, num_class)

    def forward(self, x):
        return self.linear(x)


basic_model = Linear(num_class=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(basic_model.parameters(), lr=0.1, weight_decay=1e-3)


def vectorization(x):
    avg_list = []
    for i in range(len(x)):
        x_avg = sentence_to_avg(x[i], word_to_vec_maps).reshape(1, 50)
        avg_list.append(x_avg)
    x_vec = np.concatenate(avg_list, axis=0)
    return x_vec


X_train_vec, X_test_vec = vectorization(X_train), vectorization(X_test)
# (132, 50), (56, 50)
x_train = torch.from_numpy(X_train_vec).to(device)
y_train = torch.from_numpy(Y_train).to(device)
x_test = torch.from_numpy(X_test_vec).to(device)
y_test = torch.from_numpy(Y_test).to(device)


def get_accuracy(outputs, label):
    _, prediction = torch.max(outputs, dim=1)
    correct = (prediction == label).sum()
    accuracy = correct / len(label)
    return accuracy


def train(x, y, model, epoch=1000):
    for i in range(epoch):
        outputs = model(x)
        loss = criterion(outputs, y)
        accuracy = get_accuracy(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print("Epoch [{} / {}], loss: {:.8f}, accuracy is:{:.8f}".format(i+1, epoch, loss.item(), accuracy))


def print_pred(x, prediction):
    for i in range(len(x)):
        print(x[i], label_to_emoji(int(prediction[i])))


def plot_confusion_matrix(y_true, pred, title='Confusion matrix', cmap=plt.cm.gray_r):
    df_confusion = pd.crosstab(y_true, pred.reshape(pred.shape[0], ),
                               rownames=['Actual'], colnames=['Prediction'], margins=True)
    plt.matshow(df_confusion, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)


# train(x_train, y_train, basic_model)
# with torch.no_grad():
#     test_accuracy = get_accuracy(basic_model(x_test), y_test)
#     print("Test accuracy is: %.8f" % test_accuracy, "\n")
# # Test accuracy is: 0.91071433
# X_my_sentences = np.array(["i adore you", "i love you", "funny lol",
#                            "lets play with a ball", "food is ready", "you are not happy"])
# Y_my_labels = np.array([[0], [0], [2], [1], [4], [3]]).reshape(6, )
# X_my = torch.from_numpy(vectorization(X_my_sentences)).to(device)
# y_pred = torch.max(basic_model(X_my), dim=1)[1].cpu().data.numpy()
# print_pred(X_my_sentences, y_pred)
# print(pd.crosstab(Y_my_labels, y_pred.reshape(y_pred.shape[0], ),
#                   rownames=['Actual'], colnames=['Prediction'], margins=True))
# y_test_pred = torch.max(basic_model(x_test), dim=1)[1].cpu().data.numpy()
# plot_confusion_matrix(Y_test, y_test_pred)
# plt.show()


def sentences_to_indices(x, word_to_index, max_length):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.

    :param x: array of sentences (strings), of shape (m, 1)
    :param word_to_index: a dictionary containing the each word mapped to its index
    :param max_length: maximum number of words in a sentence. You can assume every sentence in X is no longer than this.
    :return:
      X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    m = x.shape[0]
    x_indices = np.zeros((m, max_length), dtype=int)
    for i in range(m):
        sentence = x[i].lower().split()
        j = 0
        for w in sentence:
            x_indices[i][j] = word_to_index[w]
            j += 1
    return x_indices


# X1 = np.array(["funny lol", "lets play baseball", "food is ready for you ok"])
# X1_indices = sentences_to_indices(X1, word_to_indexes, max_length=6)
# print("X1 =", X1)
# print("X1_indices =", X1_indices)


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    create a embedding layer and load in pre-trained Glove 50-dim vector
    :param word_to_vec_map: dictionary mapping words to their GloVe vector representation
    :param word_to_index: dictionary mapping from words to their indices in the vocabulary (400,001 words)
    :return:
         embedding_layer -- pretrained layer
    """
    vocab_len = len(word_to_index) + 1
    emb_dim = 50
    emd_matrix = torch.zeros((vocab_len, emb_dim))
    for word, index in word_to_index.items():
        emd_matrix[index, :] = torch.from_numpy(word_to_vec_map[word])
    embedding_layer = nn.Embedding.from_pretrained(emd_matrix)
    return embedding_layer


# embedding = pretrained_embedding_layer(word_to_vec_maps, word_to_indexes)
# print("weights[1][3] =", embedding.weight[1][3])
# print(embedding.weight.shape)
# # (400001, 50)
# print(embedding(torch.from_numpy(X1_indices)).shape)
# # (3, 6, 50)


class EmojiLstm(nn.Module):
    def __init__(self, word_to_vec_map, word_to_index):
        super(EmojiLstm, self).__init__()
        self.embedding = pretrained_embedding_layer(word_to_vec_map, word_to_index)
        self.lstm = nn.LSTM(50, 128, 2, batch_first=True)
        self.linear = nn.Linear(128, 5)

    def forward(self, x):
        x = self.embedding(x)
        out = self.lstm(x)[0]
        out = nn.ReLU()(out)
        out = self.linear(out[:, -1, :])
        return out


def train_lstm(x, y, model, epoch=1000):
    x = torch.from_numpy(x).to(device)
    model.train()
    model.embedding.weight.require_grad = False
    for i in range(epoch):
        outputs = model(x)
        loss = criterion(outputs, y)
        accuracy = get_accuracy(outputs, y)

        optimizer_lstm.zero_grad()
        loss.backward()
        optimizer_lstm.step()
        # scheduler.step()
        if (i+1) % 10 == 0:
            print("Epoch [{} / {}], loss: {:.8f}, accuracy is:{:.8f}".format(i+1, epoch, loss.item(), accuracy))


lstm_model = EmojiLstm(word_to_vec_maps, word_to_indexes).to(device)
x_train_indexes = sentences_to_indices(X_train, word_to_indexes, max_len)
x_test_indexes = sentences_to_indices(X_test, word_to_indexes, max_len)
optimizer_lstm = torch.optim.Adam([param for param in lstm_model.parameters() if param.requires_grad],
                                  lr=0.001, weight_decay=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer_lstm, step_size=20, gamma=0.5, last_epoch=-1)
train_lstm(x_train_indexes, y_train, lstm_model, epoch=200)
lstm_model.eval()
output = lstm_model(torch.from_numpy(x_test_indexes).to(device))
print("test accuracy is: %.8f " % get_accuracy(output, y_test))


y_test_pred = torch.max(lstm_model(torch.from_numpy(x_test_indexes).to(device)), dim=1)[1].cpu().data.numpy()
print(pd.crosstab(Y_test, y_test_pred.reshape(y_test_pred.shape[0], ),
                  rownames=['Actual'], colnames=['Prediction'], margins=True))


x_sample = np.array(['you are so beautiful'])
x_sample_indices = sentences_to_indices(x_sample, word_to_indexes, max_len)
x_tensor = torch.from_numpy(x_sample_indices).to(device)
pred_index = np.argmax(lstm_model(x_tensor).cpu().detach().numpy())
print(x_sample[0] + ' ' + label_to_emoji(pred_index))
# 未能调出好的参数，测试集准确率只在80%左右，过拟合
