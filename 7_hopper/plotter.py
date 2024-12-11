import numpy as np
from matplotlib import pyplot as plt

plot_rewards = np.load("plot_rewards.npy")

print(np.shape(plot_rewards))

plt.plot(plot_rewards)
plt.show()