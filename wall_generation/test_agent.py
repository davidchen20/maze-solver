from maze_env_registration import * 
import gymnasium as gym
from stable_baselines3 import DQN
import pygame
import matplotlib.pyplot as plt
import time

env = gym.make("MazeEnv-walls", fixed_layout=True, grid_size=7, render_mode="rgb_array")
model = DQN.load("wall_generation/dqn_maze_agent")

obs, _ = env.reset()

plt.ion()
fig, ax = plt.subplots()
img = ax.imshow(env.render()) 
plt.axis("off")

done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    print(reward)
    frame = env.render()
    
    img.set_data(frame)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)

plt.ioff()
plt.show()