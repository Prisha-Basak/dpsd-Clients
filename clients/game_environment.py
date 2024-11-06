import numpy as np
from copy import deepcopy

class PacmanEnvironment:
    def __init__(self, width=28, height=31):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        """Initialize/reset the game board"""
        # Create basic board layout
        self.board = [[' ' for _ in range(self.width)] for _ in range(self.height)]

        # Add walls (basic maze structure)
        for i in range(self.height):
            self.board[i][0] = '#'
            self.board[i][-1] = '#'
        for j in range(self.width):
            self.board[0][j] = '#'
            self.board[-1][j] = '#'

        # Add some internal walls (simplified maze)
        for i in range(5, self.height-5, 5):
            for j in range(5, self.width-5, 5):
                self.board[i][j] = '#'

        # Add food dots
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i][j] == ' ':
                    self.board[i][j] = '.'

        # Place Pacman
        self.pacman_pos = (self.height//2, self.width//2)
        self.board[self.pacman_pos[0]][self.pacman_pos[1]] = 'P'

        # Place ghosts
        self.ghost_positions = [
            (1, 1), (1, self.width-2),
            (self.height-2, 1), (self.height-2, self.width-2)
        ]
        for i, pos in enumerate(self.ghost_positions):
            self.board[pos[0]][pos[1]] = chr(ord('a') + i)

        self.score = 0
        self.food_count = sum(row.count('.') for row in self.board)
        return self.get_board()

    def step(self, action):
        """Take a step in the environment"""
        # Action: 0=up, 1=down, 2=right, 3=left
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        dx, dy = directions[action]

        # Calculate new position
        new_x = self.pacman_pos[0] + dx
        new_y = self.pacman_pos[1] + dy

        # Check if move is valid
        if self.board[new_x][new_y] == '#':
            return self.get_board(), -1, False  # Hit wall penalty

        # Update Pacman position
        self.board[self.pacman_pos[0]][self.pacman_pos[1]] = ' '
        reward = 0

        # Check for food
        if self.board[new_x][new_y] == '.':
            reward = 1
            self.score += 1
            self.food_count -= 1

        # Check for ghosts
        elif self.board[new_x][new_y] in 'abcd':
            self.board[new_x][new_y] = 'P'
            self.pacman_pos = (new_x, new_y)
            return self.get_board(), -50, True  # Game over

        self.board[new_x][new_y] = 'P'
        self.pacman_pos = (new_x, new_y)

        # Move ghosts (simple movement towards Pacman)
        self._move_ghosts()

        # Check if Pacman was caught after ghost movement
        if self._check_ghost_collision():
            return self.get_board(), -50, True

        # Check win condition
        if self.food_count == 0:
            return self.get_board(), 100, True  # Win bonus

        return self.get_board(), reward, False

    def _move_ghosts(self):
        """Move ghosts towards Pacman"""
        for i, pos in enumerate(self.ghost_positions):
            ghost_char = chr(ord('a') + i)
            self.board[pos[0]][pos[1]] = ' ' if self.board[pos[0]][pos[1]] == ghost_char else self.board[pos[0]][pos[1]]

            # Simple ghost AI: move towards Pacman
            dx = np.sign(self.pacman_pos[0] - pos[0])
            dy = np.sign(self.pacman_pos[1] - pos[1])

            # Try horizontal movement first
            new_x, new_y = pos[0], pos[1] + dy
            if self.board[new_x][new_y] not in '#abcd':
                self.ghost_positions[i] = (new_x, new_y)
            # Try vertical movement if horizontal failed
            else:
                new_x, new_y = pos[0] + dx, pos[1]
                if self.board[new_x][new_y] not in '#abcd':
                    self.ghost_positions[i] = (new_x, new_y)

        # Update ghost positions on board
        for i, pos in enumerate(self.ghost_positions):
            if self.board[pos[0]][pos[1]] == 'P':
                continue  # Don't overwrite Pacman
            self.board[pos[0]][pos[1]] = chr(ord('a') + i)

    def _check_ghost_collision(self):
        """Check if Pacman collided with a ghost"""
        return any(pos == self.pacman_pos for pos in self.ghost_positions)

    def get_board(self):
        """Return the current board state"""
        return deepcopy(self.board)

    def get_valid_moves(self):
        """Get valid moves for current position"""
        valid = [0, 0, 0, 0]  # up, down, right, left
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

        for i, (dx, dy) in enumerate(directions):
            new_x = self.pacman_pos[0] + dx
            new_y = self.pacman_pos[1] + dy
            if self.board[new_x][new_y] != '#':
                valid[i] = 1

        return valid
