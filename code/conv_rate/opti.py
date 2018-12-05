import numpy as np
from tqdm import tqdm

def rk2_heun(x, df, h, n=100, beta=0.5):
    path = [x.copy()]
    y_prev = x.copy()
    y_next = x.copy()
    x0 = x.copy()

    for i in tqdm(range(n)):

        y_next = x0 - (h/1) * (df(x0) + df(x0 - (h/2) * df(x0)))
        x0 = (1 - beta)*y_next + beta*y_prev

        y_prev = y_next
        path.append(x0)

    return np.array(path)

def adam_bashforth(x, df, h, x_prev):
    return x - 1.5 * h * df(x) + 0.5 * h * df(x_prev)

# def nesterov()

def euler(x, df, h, n=100):
    path = [x]
    x0 = x.copy()

    for i in tqdm(range(n)):
        x0 = x0 - h * df(x0)
        path.append(x0)

    return np.array(path)

def rk2_ralston(x, df, h):
    k1 = df(x)
    k2 = df(x - 0.75*k1)
    return x - (h/4) * (k1 + 2*k2)

def rk4(x, df, h, n=100):
    path = [x.copy()]
    x0 = x.copy()

    for i in tqdm(range(n)):
        k1 = df(x0)
        k2 = df(x0 - h*0.5*k1)
        k3 = df(x0 - h*0.5*k2)
        k4 = df(x0 - h * k3)
        x0 = x0 - (h / 6)*(k1 + 2*k2 + 2*k3 + k4)
        path.append(x0)

    return np.array(path)
