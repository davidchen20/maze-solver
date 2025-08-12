from maze_env_registration import *  # Registers env
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import pygame
import matplotlib.pyplot as plt
import time
import pandas as pd

# Create environment
env = gym.make("MazeEnv-walls", fixed_layout=True, grid_size=7, render_mode="rgb_array")

# Train a DQN agent
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001)
model.learn(total_timesteps=50_000)

model.save("wall_generation/dqn_maze_agent")

# df = pd.read_csv("./logs/monitor.csv", skiprows=1)
# print(df.tail())

plt.ion()
fig, ax = plt.subplots()
img = ax.imshow(env.render())  # Get first frame
plt.axis("off") 


# Watch it play
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    print(reward)
    frame = env.render()  # Get the RGB array

    img.set_data(frame)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)  # Controls speed of playback

plt.ioff()
plt.show()
