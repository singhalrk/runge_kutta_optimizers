import numpy as np

def rk2_heun(x, df, lr=1e-2) :
    k1 = df(x)
    k2 = df(x - lr * k1)

    return x - 0.5 * lr * (k1 + k2)

def rk2_ralston(x, df, lr=1e-2) :
    k1 = df(x)
    k2 = df(x - (2/3) * k1)

    return x - 0.25 * lr * (k1 + 3 * k2)

def rk4(x, df, lr=1e-2) :
    k1 = df(x)
    k2 = df(x - (lr/2) * k1)
    k3 = df(x - (lr/2) * k2)
    k4 = df(x - lr * k3)

    return x - (k1 + 2 * k2 + 2 * k3 + k4) / 6

