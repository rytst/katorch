import matplotlib.pyplot as plt
import numpy as np

a = np.array([3, -7, 11, -7, -4, 7], dtype="float64")
p_star = len(a) -1

N = 100
SIGMA = 0.4

np.random.seed(42)
THETA = np.random.uniform(-100, 100, len(a))


# Original function
def f_star(x):
    ret = 0

    for k in range(p_star+1):
        ret += a[k] / (1 + 2 * k * np.exp(- (x - k + 4)))

    return ret


def predictor(x):
    ret = 0

    for k in range(p_star+1):
        ret += THETA[k] / (1 + 2 * k * np.exp(- (x - k + 3)))

    return ret


def generate_data(x):
    return f_star(x) + SIGMA * np.random.normal(loc=0.0, scale=1.0, size=len(x))


def plot_data(X, train_y, pred_y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Dataset")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    ax.scatter(X, train_y, s=8.0, label="Training Data")
    ax.scatter(X, pred_y, s=8.0, label="Pred Data")
    plt.legend()
    plt.savefig("./pred.png")


def generate():
    X = np.random.uniform(-5, 5, N)
    return X

def main():

    X = generate()
    train_y = generate_data(X)
    pred_y = predictor(X)
    plot_data(X, train_y, pred_y)



if __name__ == "__main__":
    main()
