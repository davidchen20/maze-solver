from maze_env_registration import *  # Make sure env is registered
import gymnasium as gym
from stable_baselines3 import DQN
import pygame
import matplotlib.pyplot as plt
import time

# Load environment and model
env = gym.make("MazeEnv-clear", grid_size=10, render_mode="rgb_array")
model = DQN.load("no_walls/dqn_maze_agent")

obs, _ = env.reset()

plt.ion()
fig, ax = plt.subplots()
img = ax.imshow(env.render())  # Get first frame
plt.axis("off")

# Run a test episode

done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    print(reward)
    frame = env.render()
    
    img.set_data(frame)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)  # Controls speed of playback

plt.ioff()
plt.show()