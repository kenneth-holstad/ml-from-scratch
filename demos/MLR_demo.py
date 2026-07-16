# first part gradient descent for LR instead of closed-form

from nn import Dense, Sequential, MSELoss
import numpy as np

# data
X = np.random.randn(100, 3)
y = 3 * X[:,0] + 1.5 * X[:,1] - 2 * X[:,2] + 2 + np.random.randn(100) * 0.5

model = Sequential([Dense(3, 1)])
loss_fn = MSELoss()
lr = 0.01

for epoch in range(501):
    pred = model.forward(X)
    loss = loss_fn.forward(pred.squeeze(), y)
    grad = loss_fn.backward()
    model.backward(grad.reshape(-1, 1))

    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b
            layer.zero_grad()

    if (epoch < 100 and epoch % 25 == 0) or (epoch < 25 and epoch % 5 == 0) \
        or epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

slopes = model.layers[0].W[:, 0]
intercept = model.layers[0].b[0]

terms = ' + '.join([f"{s:.4f}x{i+1}" for i, s in enumerate(slopes)])
print(f"GD:  y = {terms} + {intercept:.4f}")

# fit closed-form OLS on same data
from classic_algorithms.ols import OLS

ols = OLS()
ols.fit(X, y)

print(f"OLS intercept: {ols.beta[0]:.4f}")
print(f"OLS slopes: {ols.beta[1:]}")

# if you want - in future could try to do a 3-D visualization