import numpy as np
import pandas as pd

# Loading data
dataset = pd.read_csv('data (2).csv', header=None)
X = dataset[0]
y = dataset[1]
x_bar = np.mean(X)
y_bar = np.mean(y)

# Part (a) - Normal equation
def line_of_best_fit(x,y, x_bar, y_bar):
    num = np.sum(x*y - y_bar*x)
    denom = np.sum(x**2 - x_bar*x)
    B = num/denom
    a = y_bar - B*x_bar 
    return a, B

a, B = line_of_best_fit(X,y, x_bar, y_bar)
print('Value of a:', np.round(a,3))
print('Value of B:', np.round(B,3))

# Part (b) - Gradient descent
def gradient_descent(x,y):
    lr = 0.001
    m,c = 0,0
    n = len(y)
    
    for i in range(100):
        y_pred = m*x + c
        dm = -2/n * np.sum(x*(y - y_pred))
        dc = -2/n * np.sum(y - y_pred)

        m = m - (lr*dm)
        c = c - (lr*dc)

    return m,c

m,c = gradient_descent(X, y)
print("Value of m:", np.round(m,3))
print("Value of c:", np.round(c,3))