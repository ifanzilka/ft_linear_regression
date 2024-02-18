from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR
from utils import r2score_

model = MyLR(thetas=np.array([[0], [0]]).astype('float64'), alpha=1e-3, max_iter=1000)

data = pd.read_csv('../data/data.csv')
X = np.array(data['km']).reshape(-1, 1)
y = np.array(data['price']).reshape(-1, 1)

fig, ax = plt.subplots()
scatter = ax.scatter(X, y, color='blue', label='Data')

x_line = np.array([np.min(X), np.max(X)])
y_line = model.thetas[1] * x_line + model.thetas[0]
line, = ax.plot([], [], color='red', label='Linear Regression')

text = ax.text(0.3, 0.9, '', transform=ax.transAxes, ha='center')

ax.set_ylim(np.min(y)-1000, np.max(y)+1000)

def init():
    line.set_data([], [])
    text.set_text('')
    return line, text

i = 0

def update(frame):
    global i
    if i >= 50:
        # Stop the animation after 50 iterations
        animation.event_source.stop()
    normalized, not_normalized = model.fit_(X, y)
    y_pred = model.predict_(X)
    mse = model.mse_(y, y_pred)
    r2 = r2score_(y, y_pred)
    model.thetas = not_normalized
    y_line = normalized[1] * x_line + normalized[0]
    line.set_data(x_line, y_line)


    text.set_text(f'MSE: {mse:.2f}, R2 Score: {r2:.2f}\n i: {i*1000}th iteration')
    i += 1

    return line, text

animation = FuncAnimation(fig, update, frames=50, init_func=init, blit=True)

plt.title('Linear Regression Animation')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()