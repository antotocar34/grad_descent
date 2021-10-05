import numpy as np

#Boring in class 3 point OLS by hand donen in python instead. 

Y  = np.array([1.4, 1.9, 3.2])
x1 = np.array([0.05, 2.3, 2.9])
x0 = np.ones_like(x1)
X = np.stack([x0, x1], axis = 1)
beta = np.array([1., 3.]) #, b0, b1
t = 0.001

def mse(x, y, beta):
    res = y - np.dot(x, beta)
    return np.dot(np.transpose(res), res)
points_log = [beta]
mse_log = [mse(X, Y, beta)]
def grad_descent(x, y, t, mse, beta, steps):
    while len(points_log) < steps:
        f = y - np.dot(x, beta)
        beta[0] -= t * -2 * np.dot(f, x0)
        beta[1] -= t * -2 * np.dot(f, x1)
        points_log.append(beta)
        mse_log.append(mse(x, y, beta))
    print(f"Estimates for beta from gradient descent: {points_log[-1]}")
    print(f" MSE from Gradient Descent: {mse_log[-1]}")
def ols(x, y):
    beta_hat = np.dot(np.linalg.inv(np.dot(np.transpose(x), x)),np.dot(np.transpose(x), y))
    return beta_hat
beta_ols = ols(X,Y)
mse_ols = mse(X,Y, beta_ols)
print(f"Beta estimated by OLS: {beta_ols}. MSE by OLS: {mse_ols}")

grad_descent(X, Y, t, mse, beta, 1000)

threshold = 0.01
print(f"Achieved MSE with given threshold of {threshold}: {mse_log[-1] < mse_ols* (1 + threshold)}")
print(f"Steps taken: {len(points_log)}")
