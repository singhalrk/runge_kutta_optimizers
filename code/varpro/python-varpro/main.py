import numpy as np
from scipy import *
import matplotlib.pyplot as plt


def test_data1():
    t = np.arange(0, 1, 0.05)
    alpha = np.array([3, -4, 6, 1])
    c = np.arange(1,5)

    f_t = np.array([c.dot(np.cos(-alpha * t_)) for t_ in t])
    return (t, f_t)

def main():
    t, f = test_data()
    plt.plot(t,f)
    plt.show()

    # alpha =
    return None



if __name__ == '__main__':
    main()
