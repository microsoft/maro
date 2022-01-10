import time

from torch import nn
from torch.nn import NLLLoss
from torch.optim import Adam

SLEEP = 5


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self._fc1 = nn.Linear(input_dim, 3)
        self._relu = nn.ReLU()
        self._fc2 = nn.Linear(3, output_dim)
        self._net = nn.Sequential(self._fc1, self._relu, self._fc2)

    def forward(self, inputs):
        return self._net(inputs)


class TrainOps:
    def __init__(self, name, model):
        self.name = name
        self._loss_fn = NLLLoss()
        self._model = model
        self._optim = Adam(self._model.parameters(), lr=0.001)

    def step(self, X, y):
        y_pred = self._model(X)
        loss = self._loss_fn(y_pred, y)
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()
        time.sleep(SLEEP)


class TrainOps2:
    def __init__(self, name, model):
        self.name = name
        self._loss_fn = NLLLoss()
        self._model = model
        self._optim = Adam(self._model.parameters(), lr=0.001)

    def step(self, X, y):
        y_pred = self._model(X)
        loss = self._loss_fn(y_pred, y)
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()
        time.sleep(SLEEP)


ops_creator = {
    "single.ops": lambda name: TrainOps(name, Model(5, 2)),
    "multi.ops0": lambda name: TrainOps2(name, Model(7, 3)),
    "multi.ops1": lambda name: TrainOps2(name, Model(7, 3)),
    "multi.ops2": lambda name: TrainOps2(name, Model(7, 3))
}
