import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torchviz import make_dot
from tqdm import tqdm

a = np.array([3, -7, 11, -7, -4, 7], dtype="float64")
p_star = len(a) -1

N = 100
SIGMA = 0.4

LEARNING_RATE = 0.1
NUM_EPOCHS = 4000

np.random.seed(42)

# Original function
def f_star(x):
    ret = 0

    for k in range(p_star+1):
        ret += a[k] / (1 + 2 * k * np.exp(- (x - k + 4)))

    return ret

def generate_data(x):
    return f_star(x) + SIGMA * np.random.normal(loc=0.0, scale=1.0, size=len(x))

def generate():
    X = np.random.uniform(-5, 5, N)
    return X

###################
# Predictor Model #
###################

class Predictor(nn.Module):
    def __init__(self, p_star):
        super(Predictor, self).__init__()

        self.p_star = p_star
        theta_list= np.random.uniform(-100, 100, size=(p_star + 1))
        self.theta = nn.Parameter(torch.tensor(theta_list)) # Model parameters

    def forward(self, x):
        result = torch.zeros_like(x)
        for k in range(self.p_star + 1):
            result += self.theta[k] / (1 + 2 * k * torch.exp(- (x - k + 3)))
        return result

#################
# Loss Function #
#################
def loss_fn(pred_y, train_y):
    loss = 0.5 * (pred_y - train_y) ** 2
    loss = torch.mean(loss)
    return loss


def main():

    #################
    # Training data #
    #################
    X = generate()
    train_X = torch.tensor(X)

    train_y = generate_data(X)
    train_y = torch.tensor(train_y)

    ##############
    # Prediction #
    ##############
    model = Predictor(p_star)
    pred_y = model(train_X)

    #############
    # Optimizer #
    #############
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    ############
    # Training #
    ############
    loss_history = list()
    for epoch in tqdm(range(NUM_EPOCHS)):
        optimizer.zero_grad()
        pred_y = model(train_X)
        loss = loss_fn(pred_y, train_y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    ############
    # Plotting #
    ############
    plt.plot(np.arange(NUM_EPOCHS), loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training Loss")
    plt.savefig("loss.png")





if __name__ == "__main__":
    main()
