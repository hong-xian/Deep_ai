import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from tensorflow.keras.utils import to_categorical

fake = Faker()
Faker.seed(12345)
random.seed(12345)
FORMATS = ['short', 'medium', 'long',
           'full', 'full', 'full',
           'full', 'full', 'full',
           'full', 'full', 'full',
           'full', 'd MMM YYY', 'd MMMM YYY',
           'dd MMM YYY', 'd MMM, YYY', 'd MMMM, YYY',
           'dd, MMM YYY', 'd MM YY', 'd MMMM YYY',
           'MMMM d YYY', 'MMMM d, YYY', 'dd.MM.YY']


def load_date():
    """
        Loads some fake dates
        :returns: tuple containing human readable string, machine readable string, and date object
    """
    dt = fake.date_object()

    try:
        human_readable = format_date(dt, format=random.choice(FORMATS),  locale='en_US')
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',', '')
        machine_readable = dt.isoformat()

    except AttributeError:
        return None, None, None

    return human_readable, machine_readable, dt


def load_dataset(num):
    """
        Loads a dataset with m examples and vocabularies
        :m: the number of examples to generate
    """
    human_vocab = set()
    machine_vocab = set()
    dataset = []

    for i in tqdm(range(num)):
        h, m, _ = load_date()
        if h is not None:
            dataset.append((h, m))
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))

    human_vocab.add("<unk>")
    human_vocab.add("<pad>")
    human = dict(zip(sorted(human_vocab), list(range(len(human_vocab) + 2))))
    inv_machine = dict(enumerate(sorted(machine_vocab)))
    machine = {v: k for k, v in inv_machine.items()}

    return dataset, human, machine, inv_machine


def preprocess_data(dataset, human_vocab, machine_vocab, len_x, len_y):
    x, y = zip(*dataset)

    x = np.array([string_to_int(i, len_x, human_vocab) for i in x])
    y = np.array([string_to_int(t, len_y, machine_vocab) for t in y])

    x_oh = np.array(list(map(lambda s: to_categorical(s, num_classes=len(human_vocab)), x)))
    y_oh = np.array(list(map(lambda s: to_categorical(s, num_classes=len(machine_vocab)), y)))

    return x, y,  x_oh, y_oh


def string_to_int(string, length, vocab):
    """
    Converts all strings in the vocabulary into a list of integers representing the positions of the
    input string's characters in the "vocab"

    Arguments:
    string -- input string, e.g. 'Wed 10 Jul 2007'
    length -- the number of time steps you'd like, determines if the output will be padded or cut
    vocab -- vocabulary, dictionary used to index every character of your "string"

    Returns:
    rep -- list of integers (or '<unk>') (size = length) representing the position
            of the string's character in the vocabulary
    """

    # make lower to standardize
    string = string.lower()
    string = string.replace(',', '')

    if len(string) > length:
        string = string[:length]

    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))

    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))

    return rep


datasets, human_vocabs, machine_vocabs, inv_machine_vocabs = load_dataset(num=10000)
# print(datasets[:10])
print(machine_vocabs)
print(human_vocabs)
Tx = 30
Ty = 10
X, Y, X_oh, Y_oh = preprocess_data(datasets, human_vocabs, machine_vocabs, Tx, Ty)
print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("X_oh.shape:", X_oh.shape)
print("Y_oh.shape:", Y_oh.shape)
# index = 0
# print("\nSource date:", datasets[index][0])
# print("Target date:", datasets[index][1])
# print()
# print("Source after preprocessing (indices):", X[index])
# print("Target after preprocessing (indices):", Y[index])
# print()
# print("Source after preprocessing (one-hot):", X_oh[index])
# print("Target after preprocessing (one-hot):", Y_oh[index])


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bi_lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, x):
        outputs, (h, c) = self.bi_lstm(x)
        h = h.transpose(1, 2).reshape(-1, 1, self.hidden_dim*2)
        return outputs, h


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim*2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        self.v.data.normal_(mean=0, std=1. / np.sqrt(self.v.size(0)))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        h = hidden.repeat(1, max_len, 1)
        attn_energies = self.score(h, encoder_outputs)
        return nn.Softmax(dim=1)(attn_energies)

    def score(self, h, encoder_outputs):
        energy = nn.Tanh()(self.attn(torch.cat([h, encoder_outputs], 2)))
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(2)
        energy = torch.bmm(energy, v)
        return energy.squeeze(2)


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim+output_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, decoder_input, hidden, encoder_outputs):
        attn_weight = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attn_weight.unsqueeze(1), encoder_outputs)
        context_hidden = torch.cat([context, decoder_input], dim=2)
        output, (post_hidden, c) = self.lstm(context_hidden)
        output = self.out(output)
        post_hidden = post_hidden.transpose(0, 1)
        return output, post_hidden, attn_weight


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seqs):
        batch_size, max_len, output_dim = input_seqs.size(0), Ty, self.decoder.output_dim
        outputs = torch.zeros(batch_size, max_len, output_dim).to(device)
        encoder_outputs, hidden = self.encoder(input_seqs)
        output = torch.zeros(batch_size, 1, output_dim).to(device)
        attn_weights = torch.zeros(batch_size, Ty, Tx)
        for t in range(1, max_len+1):
            output, hidden, attn = self.decoder(output, hidden, encoder_outputs)
            outputs[:, t-1, :] = output.squeeze(1)
            attn_weights[:, t-1, :] = attn
        return outputs, attn_weights


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("*********", device)
input_tensor = torch.from_numpy(X_oh).to(device)
target_tensor = torch.from_numpy(Y_oh).to(device)
Encoder = Encoder(input_dim=37, hidden_dim=32)
Decoder = Decoder(hidden_dim=64, output_dim=11)
model = Seq2Seq(Encoder, Decoder).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
model.train()
loss_list = []
for j in range(3000):
    predictions, _ = model(input_tensor)
    loss = criterion(predictions, target_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())
    if (j+1) % 100 == 0:
        print("%d iteration loss is: %.8f" % (j+1, loss.item()))

torch.save(model.state_dict(), "/home/liushuang/PycharmProjects/lab/mymodel/date_trans.pt")
plt.plot(loss_list)

model.eval()
EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001',
            'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
    source = string_to_int(example, Tx, human_vocabs)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocabs)), source)))
    source_tensor = torch.from_numpy(source).unsqueeze(0).to(device)
    pred, _ = model(source_tensor)
    pred = pred.cpu().detach().numpy()
    pred = np.argmax(pred, axis=-1)[0]
    trans_out = [inv_machine_vocabs[int(i)] for i in pred]
    print("source:", example)
    print("output:", ''.join(trans_out))


def show_attention(input_words, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    ax.set_xticklabels([''] + list(input_words))
    ax.set_yticklabels([''] + list(output_words))

    ax.xaxis.set_major_locator(ticker.MultipleLocator())
    ax.yaxis.set_major_locator(ticker.MultipleLocator())


def evaluate(input_words):
    model.eval()
    with torch.no_grad():
        data = string_to_int(input_words, Tx, human_vocabs)
        processed_data = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocabs)), data)))
        data_tensor = torch.from_numpy(processed_data).unsqueeze(0).to(device)
        outputs, attention_weights = model(data_tensor)
        outputs = outputs.cpu().detach().numpy().squeeze(0)
        prediction = np.argmax(outputs, axis=-1)
        attn_weights = attention_weights.cpu().detach().numpy().squeeze(0)
        output_words = [inv_machine_vocabs[int(i)] for i in prediction]
        show_attention(input_words, output_words, attn_weights)


evaluate('3 May 1979')
plt.show()
