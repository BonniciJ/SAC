import numpy as np
from matplotlib import pyplot as plt

# Code written by me, optimised by chatGPT 4o

def movAvg(x, w=100):
    """Calculate the moving average with window size w."""
    return np.convolve(x, np.ones(w) / w, mode='valid')


def moving_std(data, w=100):
    """Calculate the moving standard deviation with window size w."""
    moving_avg = movAvg(data, w)
    # Ensure the standard deviation aligns with the moving average
    return np.sqrt(np.convolve((data - np.pad(moving_avg, (w-1, 0), 'constant'))**2, np.ones(w) / w, mode='valid'))


# Load data
plot_log = np.load("plot_log.npy")

# Extract metrics
episode_reward = plot_log[:, 0]
q1_loss = plot_log[:, 1] / np.arange(1, len(plot_log[:, 1]) + 1)
q2_loss = plot_log[:, 2] / np.arange(1, len(plot_log[:, 2]) + 1)
pi_loss = plot_log[:, 3] / np.arange(1, len(plot_log[:, 3]) + 1)

# Moving averages and standard deviations
window_size = 150
episode_reward_avg = movAvg(episode_reward, window_size)
episode_reward_std = moving_std(episode_reward, window_size)

q1_loss_avg = movAvg(q1_loss, window_size)
q1_loss_std = moving_std(q1_loss, window_size)

q2_loss_avg = movAvg(q2_loss, window_size)
q2_loss_std = moving_std(q2_loss, window_size)

pi_loss_avg = movAvg(pi_loss, window_size)
pi_loss_std = moving_std(pi_loss, window_size)

# Create a single figure and three subplots in one row
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Training Metrics for Hopper Environment', fontsize=16)

# Plot learning curve
x_range = range(window_size - 1, len(episode_reward))  # Align x-axis
axes[0].plot(x_range, episode_reward_avg, color="blue", label="Moving Average")
axes[0].fill_between(
    x_range,
    episode_reward_avg - episode_reward_std,
    episode_reward_avg + episode_reward_std,
    color="blue",
    alpha=0.2,
    label="Standard Deviation"
)
axes[0].set_title('Learning Curve')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Return')
axes[0].legend()

# Plot Q losses
q_x_range = range(window_size - 1, len(q1_loss))  # Align x-axis for Q losses
axes[1].plot(q_x_range, q1_loss_avg, label='Q1 Loss', color='orange')
axes[1].fill_between(
    q_x_range,
    q1_loss_avg - q1_loss_std,
    q1_loss_avg + q1_loss_std,
    color="orange",
    alpha=0.2
)
axes[1].plot(q_x_range, q2_loss_avg, label='Q2 Loss', color='green')
axes[1].fill_between(
    q_x_range,
    q2_loss_avg - q2_loss_std,
    q2_loss_avg + q2_loss_std,
    color="green",
    alpha=0.2
)
axes[1].set_title('Q Losses')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Loss')
axes[1].legend()

# Plot policy loss
pi_x_range = range(window_size - 1, len(pi_loss))  # Align x-axis for policy loss
axes[2].plot(pi_x_range, pi_loss_avg, label="Policy Loss", color="purple")
axes[2].fill_between(
    pi_x_range,
    pi_loss_avg - pi_loss_std,
    pi_loss_avg + pi_loss_std,
    color="purple",
    alpha=0.2
)
axes[2].set_title('Policy Loss')
axes[2].set_xlabel('Episode')
axes[2].set_ylabel('Loss')
axes[2].legend()

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.85)

# Save the figure
plt.savefig('plot.png')
plt.show()
