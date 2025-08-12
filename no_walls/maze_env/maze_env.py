# maze_env/maze_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, grid_size, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # up, right, down, left
        self.reset()
        self.render_mode = render_mode
        self.window_size = 500
        self.cell_size = self.window_size // self.grid_size
        self.window = None
        self.clock = None
        

    def reset(self, seed=None, options=None):
        self.agent_pos = [0, 0]
        self.goal_pos = [9, 8]
        self.previous_distance_squared = (self.agent_pos[1] - self.goal_pos[1])**2 + (self.agent_pos[0] - self.goal_pos[0])**2
        self.total_timesteps = 0
        self.visited_nodes = {tuple(self.agent_pos): 1}

        return self._get_obs(), {}
    
    def step(self, action):
        reward = 0
        done = False
        x, y = self.agent_pos
        if action == 0:
            if y > 0: 
                y -= 1
            else:
                reward -= 0.8
        elif action == 1: 
            if x < self.grid_size - 1: 
                x += 1
            else:
                reward -= 0.8
        elif action == 2: 
            if y < self.grid_size - 1: 
                y += 1
            else:
                reward -= 0.8
        elif action == 3:
            if x > 0: 
                x -= 1
            else:
                reward -= 0.8
        self.agent_pos = [x, y]

        distance_squared = (self.agent_pos[1] - self.goal_pos[1])**2 + (self.agent_pos[0] - self.goal_pos[0])**2
        delta = self.previous_distance_squared - distance_squared
    
        reward -= 1

        reward += 0.5 * delta

        if self.agent_pos == self.goal_pos:
            done = True
            reward += 10

        if self.total_timesteps >= self.grid_size ** 2 and not done:
            done = True
            reward -= 50

        self.previous_distance_squared = distance_squared

        self.total_timesteps += 1
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        obs[self.agent_pos[1], self.agent_pos[0]] = 1.0
        obs[self.goal_pos[1], self.goal_pos[0]] = 0.5
        return obs

    def _surface_to_array(self, surface):
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        ).copy()

    def render(self):
        surface = None
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
                self.clock = pygame.time.Clock()
            surface = self.window
        elif self.render_mode == "rgb_array":
            surface = pygame.Surface((self.window_size, self.window_size))

        if surface is None:
            return

        surface.fill((255, 255, 255))  # white background

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                color = (200, 200, 200)

                if [x, y] == self.agent_pos:
                    color = (0, 0, 255)  # blue agent
                elif [x, y] == self.goal_pos:
                    color = (0, 255, 0)  # green goal

                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (0, 0, 0), rect, 1)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return self._surface_to_array(surface)
