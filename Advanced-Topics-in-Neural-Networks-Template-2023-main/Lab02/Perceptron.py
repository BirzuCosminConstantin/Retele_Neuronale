import torch
from torch import Tensor


def sigmoid(z):
    # y = σ(z) =1 / (1 + exp(−z))
    return 1 / (1 + torch.exp(-z))


def train_perceptron(X: Tensor, W: Tensor, b: Tensor, y_true: Tensor, mu: float):
    # z = w·x + b
    z = torch.mm(X, W) + b
    y = sigmoid(z)

    # Error = Target − y
    error = y_true - y

    # w := w + µ × Error × x
    w_temp = torch.matmul(X.t(), error)
    W += mu * w_temp
    # b := b + µ × Error
    b_temp = error.sum(dim=0)
    print(b_temp)
    b += mu * b_temp

    return W, b


m = 100
n_features = 784
n_perceptrons = 10

# A 2D PyTorch tensor of shape (m, 784) containing the input features.
X = torch.randn(m, n_features)
# A 2D PyTorch tensor of shape (784, 10) containing the initial weights for the 10 perceptrons.
W = torch.randn(n_features, n_perceptrons)
# biases for the 10 perceptrons.
b = torch.randn(n_perceptrons)
# A 2D PyTorch tensor of shape (m, 10) containing the true labels for each of the m examples.
y_true = torch.randint(0, 2, (m, n_perceptrons)).float()
# A float representing the learning rate.
mu = 0.1
for epoch in range(30):
    updated_W, updated_b = train_perceptron(X, W, b, y_true, mu)

print('Updated weights:', W)
print('Updated biases:', b)
