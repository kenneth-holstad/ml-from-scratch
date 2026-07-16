# first part gradient descent for LR instead of closed-form

from nn import Dense, Sequential, MSELoss
import numpy as np
import matplotlib.pyplot as plt

# data
X = np.random.randn(100, 1)
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.5

model = Sequential([Dense(1, 1)])
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

slope = model.layers[0].W[0, 0]      # W is shape (1,1), so index in to get the scalar
intercept = model.layers[0].b[0]     # b is shape (1,), so index to get the scalar

print(f"GD: y = {slope:.4f}x + {intercept:.4f}")

# fit closed-form OLS on same data
from classic_algorithms.ols import OLS

ols = OLS()
ols.fit(X, y)

ols_slope = ols.beta[1]
ols_intercept = ols.beta[0]

print(f"OLS:  y = {ols_slope:.4f}x + {ols_intercept:.4f}")


# plot results
plt.scatter(X, y, color='steelblue', alpha=0.6, label='Data points')

# regression line -- generate evenly spaced x values across the range, plot predicted y
x_line = np.linspace(X.min(), X.max(), 100)
y_line = slope * x_line + intercept
y_line_ols = ols_slope * x_line + ols_intercept
plt.plot(x_line, y_line, color='tomato', linewidth=2, 
         label=f'GD: y = {slope:.4f}x + {intercept:.4f}')
plt.plot(x_line, y_line_ols, color='green', linewidth=2, 
         linestyle='--', label=f'OLS: y = {ols_slope:.4f}x + {ols_intercept:.4f}')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression - Gradient Descent')
plt.legend()
plt.tight_layout()
plt.show()