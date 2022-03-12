import matplotlib.pyplot as plt
import pickle

# Load the reward and loss pickle files
all_rewards = pickle.load(open("all_rewards.pkl", "rb"))
losses = pickle.load(open("losses.pkl", "rb"))

# Rewards and losses are a list of tuples: (frame, reward), (frame, loss)
rewards_list = list(zip(*all_rewards))
reward_frames = rewards_list[0]
reward_y = rewards_list[1]

losses_list = list(zip(*losses))
losses_frames = losses_list[0]
losses_y = losses_list[1]

# Subplot function
figure, axis = plt.subplots(2)

# Plot the Rewards
axis[0].plot(reward_frames, reward_y)
axis[0].set_title("Rewards")

# Plot the Losses
axis[1].plot(losses_frames, losses_y)
axis[1].set_title("Losses")

plt.show()
