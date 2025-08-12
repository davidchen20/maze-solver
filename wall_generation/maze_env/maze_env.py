import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, grid_size, fixed_layout=False, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.fixed_layout = fixed_layout
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # up, right, down, left
        self.reset()
        self.render_mode = render_mode
        self.window_size = 500
        self.cell_size = self.window_size // self.grid_size
        self.window = None
        self.clock = None
        
    def _generate_maze(self):
        # Start with a grid full of walls
        self.walls = np.ones((self.grid_size, self.grid_size), dtype=np.uint8)
        
        stack = []

        start_x, start_y = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        self.walls[start_y, start_x] = 0 
        stack.append((start_x, start_y))

        while stack:
            current_x, current_y = stack[-1]
            neighbors = []

            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = current_x + dx, current_y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.walls[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                next_x, next_y = random.choice(neighbors)
                
                self.walls[next_y, next_x] = 0
                self.walls[current_y + (next_y - current_y) // 2, current_x + (next_x - current_x) // 2] = 0
                
                stack.append((next_x, next_y))
            else:
                stack.pop()

        np.save("maze.npy", self.walls)

    def _place_agent_and_goal(self):
        valid_positions = np.argwhere(self.walls == 0).tolist()
        
        if len(valid_positions) < 2:
            self.reset() 
            return

        # self.agent_pos = random.choice(valid_positions)
        self.agent_pos = [4, 4]
        # self.goal_pos = random.choice(valid_positions)
        self.goal_pos = [0, 0]

        # while np.array_equal(self.agent_pos, self.goal_pos):
        #     self.goal_pos = random.choice(valid_positions)
        
        # Convert from [row, col] (numpy format) to [x, y]
        self.agent_pos = [self.agent_pos[1], self.agent_pos[0]]
        self.goal_pos = [self.goal_pos[1], self.goal_pos[0]]


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.fixed_layout:
            self.walls = np.load("maze.npy")
        else:
            self._generate_maze()
        self._place_agent_and_goal()

        self.previous_distance_squared = (self.agent_pos[0] - self.goal_pos[0])**2 + (self.agent_pos[1] - self.goal_pos[1])**2
        self.total_timesteps = 0

        self.visited_nodes = set()

        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), {}
    
    def step(self, action):
        reward = 0
        done = False
        x, y = self.agent_pos
        if action == 0:
            if y > 0 and self.walls[y-1, x] != 1: 
                y -= 1
            else:
                reward -= 10
        elif action == 1: 
            if x < self.grid_size - 1 and self.walls[y, x+1] != 1: 
                x += 1
            else:
                reward -= 10
        elif action == 2: 
            if y < self.grid_size - 1 and self.walls[y+1, x] != 1: 
                y += 1
            else:
                reward -= 10
        elif action == 3:
            if x > 0 and self.walls[y, x-1] != 1: 
                x -= 1
            else:
                reward -= 10
        self.agent_pos = [x, y]

        distance_squared = (self.agent_pos[1] - self.goal_pos[1])**2 + (self.agent_pos[0] - self.goal_pos[0])**2
        delta = self.previous_distance_squared - distance_squared

        reward -= 1

        reward += delta

        if self.agent_pos == self.goal_pos:
            done = True
            reward += 100

        if self.total_timesteps >= self.grid_size ** 2 and not done:
            done = True
            reward -= 0.5*distance_squared
            # reward -= 50
            

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
                elif self.walls[y, x] == 1: 
                    color = (0, 0, 0) # wall

                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (0, 0, 0), rect, 1)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return self._surface_to_array(surface)
