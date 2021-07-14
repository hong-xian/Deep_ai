# from sin predict cos
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary


class Rnn(nn.Module):
    def __init__(self, inputsize):
        super(Rnn, self).__init__()

        self.rnn = nn.RNN(input_size=inputsize,
                          hidden_size=32,
                          num_layers=1,
                          batch_first=True)
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time, :]))
        return torch.stack(outs, dim=1), h_state


time_step = 50
inputsize = 1
lr = 0.001
model = Rnn(inputsize)
print(model)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

h_state = None
losses = []
for step in range(300):
    start, end = step * np.pi, (step+1) * np.pi
    steps = np.linspace(start, end, time_step, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = model(x, h_state)
    h_state = h_state.data
    loss = loss_func(prediction, y)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step == 299:
        with torch.no_grad():
            plt.plot(steps, y_np.flatten(), "r-")
            plt.plot(steps, prediction.data.numpy().flatten(), 'b-')

plt.figure()
plt.plot(losses)
plt.show()
