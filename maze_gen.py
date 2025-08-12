import numpy as np
import random
import time
import sys

def generate_maze(grid_size=15, delay=0.03):
    """
    Generate and visualize a maze using the 2-cell jump recursive-backtracker.
    Prints pixel-art frames to the terminal.
    """
    # Ensure odd size for proper wall/cell alignment
    if grid_size % 2 == 0:
        grid_size += 1

    # 1 = wall, 0 = passage
    walls = np.ones((grid_size, grid_size), dtype=np.uint8)

    # Pick a random starting cell at odd coordinates
    sx = random.randrange(1, grid_size, 2)
    sy = random.randrange(1, grid_size, 2)
    walls[sy, sx] = 0
    stack = [(sx, sy)]

    def print_frame(current=None):
        WALL = "█"
        PATH = " "
        CURR = "▓"
        out_lines = []
        for y in range(grid_size):
            row_chars = []
            for x in range(grid_size):
                if current is not None and (x, y) == current:
                    row_chars.append(CURR)
                elif walls[y, x] == 1:
                    row_chars.append(WALL)
                else:
                    row_chars.append(PATH)
            out_lines.append("".join(row_chars))
        sys.stdout.write("\033[H\033[J")  # clear screen
        sys.stdout.write("\n".join(out_lines) + "\n")
        sys.stdout.flush()

    print_frame(stack[-1])
    time.sleep(delay)

    while stack:
        cx, cy = stack[-1]
        neighbors = []
        # Look for unvisited neighbors 2 cells away
        for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and walls[ny, nx] == 1:
                neighbors.append((nx, ny))

        if neighbors:
            nx, ny = random.choice(neighbors)
            walls[ny, nx] = 0
            mx, my = (cx + nx) // 2, (cy + ny) // 2
            walls[my, mx] = 0
            stack.append((nx, ny))
        else:
            stack.pop()

        print_frame(stack[-1] if stack else None)
        time.sleep(delay)

    return walls

if __name__ == "__main__":
    generate_maze(grid_size=15, delay=0.05)