from maze_env_registration import *  # Registers env
import gymnasium as gym
from stable_baselines3 import DQN
import pygame
import matplotlib.pyplot as plt
import time

env = gym.make("MazeEnv-walls", fixed_layout=True, grid_size=7, render_mode="rgb_array")

model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001)
model.learn(total_timesteps=50_000)

model.save("wall_generation/dqn_maze_agent")

plt.ion()
fig, ax = plt.subplots()
img = ax.imshow(env.render()) 
plt.axis("off") 

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    print(reward)
    frame = env.render()

    img.set_data(frame)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1) 

plt.ioff()
plt.show()
