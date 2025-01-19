import matplotlib.pyplot as plt
import numpy as np

a = np.array([3, -7, 11, -7, -4, 7], dtype="float64")
p_star = len(a) -1

N = 100
SIGMA = 0.4

np.random.seed(42)


# Original function
def f_star(x):
    ret = 0

    for k in range(p_star+1):
        ret += a[k] / (1 + 2 * k * np.exp(- (x - k + 4)))

    return ret

def generate_data(x):
    return f_star(x) + SIGMA * np.random.normal(loc=0.0, scale=1.0, size=len(x))


def plot_f_star():
    X = np.linspace(-5, 5, 1000)
    y = f_star(X)
    plt.title(r"$y = f_*(x)$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.plot(X, y)
    plt.savefig("./f_star.png")
    plt.clf()

def plot_data(X, y):
    plt.title("Dataset")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.scatter(X, y, s=8.0, label="Training Data")
    plt.legend()
    plt.savefig("./dataset.png")
    plt.clf()


def generate():
    X = np.random.uniform(-5, 5, N)
    return X

def main():

    # Plot original function
    plot_f_star()

    X = generate()
    y = generate_data(X)
    plot_data(X, y)



if __name__ == "__main__":
    main()
