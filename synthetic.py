from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Array = np.ndarray


def rmse(y: Array, y_pred: Optional[Array] = None) -> float:
    """Calculate the root mean squared error between y and y_pred.

    If y_pred is not provided, y is treated as the residual vector.
    """

    if y_pred is None:
        rmse = np.sqrt(np.mean(np.power(y, 2)))
    else:
        rmse = np.sqrt(np.mean(np.power((y - y_pred), 2)))

    return rmse


def mae(y: Array, y_pred: Optional[Array] = None) -> float:
    """Calculate the mean squared error between y and y_pred.

    If y_pred is not provided, y is treated as the residual vector.
    """

    if y_pred is None:
        mae = np.mean(np.abs(y))
    else:
        mae = np.mean(np.abs((y - y_pred)))

    return mae


def get_energy(nheavy: int, nsamples: int) -> Array:
    raw = np.random.normal(size=(nheavy, nsamples))
    return np.sum(raw, 0)


#%%

xplot = []
yplot = []
for nh in range(1, 9):
    method1 = get_energy(nh, 1_000_000)
    method2 = get_energy(nh, 1_000_000)
    rms1 = rmse(method1, method2)
    xplot.append(nh)
    yplot.append(rms1)

#%%
xplot = np.array(xplot)
yplot = np.array(yplot)
plt.plot(xplot, yplot, "ro")
plt.plot(xplot, yplot / np.sqrt(xplot), "bo")
plt.show()

# %%

# Generate a synthetic dataset
xplot = []
yplot = []
for nh in range(1, 9):
    method1 = get_energy(nh, 1_000_000)
    method2 = get_energy(nh, 1_000_000)

    residual = method1 - method2

    data = {
        "RMSE(E)": rmse(residual),
        "RMSE(E) / sqrt(nh)": rmse(residual) / np.sqrt(nh),
        "RMSE(E/nh)": rmse(residual / nh),
        "MAE(E)": mae(residual),
        "MAE(E) / sqrt(nh)": mae(residual) / np.sqrt(nh),
        "MAE(E/nh)": mae(residual / nh),
    }

    yplot.append(data)
    xplot.append(nh)

df = pd.DataFrame(yplot, index=xplot)

# %%

# Plotting
cols = [
    "RMSE(E)",
    "RMSE(E) / sqrt(nh)",
    "RMSE(E/nh)",
    "MAE(E)",
    "MAE(E) / sqrt(nh)",
    "MAE(E/nh)",
]

fig, ax = plt.subplots(figsize=(11, 8.5))

ax.set_prop_cycle(
    color=["b", "r", "g", "b", "r", "g"],
    linestyle=["-", "-", "-", "-.", "-.", "-."],
    marker=["o", "o", "o", "o", "o", "o"],
)

plt.plot(xplot, df[cols].to_numpy())
plt.legend(cols, loc="upper left", fancybox=True)

# %%
