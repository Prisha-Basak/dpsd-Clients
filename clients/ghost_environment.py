import numpy as np
import torch
from typing import List, Tuple

class GhostEnvironment:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.board_size = (31, 28)
        self.ghost_positions = []
        self.pacman_pos = None
        self.reset()

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _convert_board_to_numerical(self, board):
        """Convert board to numerical representation."""
        import torch
        numerical_board = np.zeros((31, 28), dtype=np.float32)

        # Only print debug info if debug mode is enabled
        if self.debug_mode:
            print(f"\nBoard dimensions: {len(board)}x{len(board[0])}")

        for i in range(len(board)):
            for j in range(len(board[0])):
                cell = board[i][j]
                if cell == '#':  # Wall
                    numerical_board[i][j] = 0.0
                elif cell == '.':  # Food
                    numerical_board[i][j] = 0.5
                elif cell == ' ':  # Empty space
                    numerical_board[i][j] = 0.1
                elif cell == 'P':  # Pacman
                    numerical_board[i][j] = 1.0
                    self.pacman_pos = (i, j)
                elif cell in ['a', 'b', 'c', 'd']:  # Ghosts
                    numerical_board[i][j] = -0.5
                    self.ghost_positions.append((i, j))

        # Convert to PyTorch tensor and add batch and channel dimensions
        tensor_board = torch.FloatTensor(numerical_board)
        tensor_board = tensor_board.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 31, 28)

        if self.debug_mode:
            print(f"Numerical board shape: {tensor_board.shape}\n")

        return tensor_board

    def reset(self):
        """Reset the environment to initial state."""
        # Initialize empty board
        self.board = [[' ' for _ in range(28)] for _ in range(31)]

        # Add walls (simplified maze structure)
        for i in range(31):
            self.board[i][0] = '#'
            self.board[i][27] = '#'
        for j in range(28):
            self.board[0][j] = '#'
            self.board[30][j] = '#'

        # Place ghosts in corners
        ghost_positions = [
            (1, 1),    # Ghost a - top left
            (1, 26),   # Ghost b - top right
            (29, 1),   # Ghost c - bottom left
            (29, 26)   # Ghost d - bottom right
        ]
        self.ghost_positions = []  # Track ghost positions

        for idx, (x, y) in enumerate(ghost_positions):
            ghost_char = chr(ord('a') + idx)
            if self.board[x][y] == ' ':  # Ensure position is empty
                self.board[x][y] = ghost_char
                self.ghost_positions.append((x, y))
            else:
                print(f"Warning: Ghost {ghost_char} position occupied, finding alternative")
                # Find nearest empty space
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        new_x, new_y = x + dx, y + dy
                        if (0 <= new_x < 31 and 0 <= new_y < 28 and
                            self.board[new_x][new_y] == ' '):
                            self.board[new_x][new_y] = ghost_char
                            self.ghost_positions.append((new_x, new_y))
                            break

        # Place Pacman in center
        center_x, center_y = 15, 13
        self.board[center_x][center_y] = 'P'
        self.pacman_pos = (center_x, center_y)

        # Initialize previous distances for reward calculation
        self.ghost_previous_distances = [
            self._manhattan_distance(pos, self.pacman_pos)
            for pos in self.ghost_positions
        ]

        # Convert board to numerical state
        return self._convert_board_to_numerical(self.board)

    def _calculate_ghost_rewards(self):
        """Calculate rewards for ghost positions."""
        total_reward = 0

        # Base reward for each ghost based on distance to Pacman
        for i, ghost_pos in enumerate(self.ghost_positions):
            distance = self._manhattan_distance(ghost_pos, self.pacman_pos)
            prev_distance = self.ghost_previous_distances[i]

            # Reward for getting closer to Pacman
            if distance < prev_distance:
                total_reward += 0.5
            elif distance > prev_distance:
                total_reward -= 0.3

            # Reward inverse to distance (closer is better)
            if distance == 0:  # Caught Pacman
                total_reward += 100
            elif distance <= 2:  # Very close to Pacman
                total_reward += 5
            elif distance <= 5:  # Moderately close
                total_reward += 2

            # Update previous distance
            self.ghost_previous_distances[i] = distance

        # Strategic positioning rewards
        ghost_positions_set = set(self.ghost_positions)
        pacman_surroundings = [
            (self.pacman_pos[0] + dx, self.pacman_pos[1] + dy)
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]
        ]

        # Reward for surrounding Pacman
        surrounded_count = sum(1 for pos in pacman_surroundings
                             if pos in ghost_positions_set)
        total_reward += surrounded_count * 2

        # Coordination rewards - encourage ghosts to maintain optimal spacing
        for i, pos1 in enumerate(self.ghost_positions):
            for j, pos2 in enumerate(self.ghost_positions):
                if i < j:  # Only check each pair once
                    distance = self._manhattan_distance(pos1, pos2)
                    # Optimal distance between ghosts is 3-5 spaces
                    if 3 <= distance <= 5:
                        total_reward += 0.5
                    elif distance < 2:  # Too close
                        total_reward -= 1.0
                    elif distance > 8:  # Too far
                        total_reward -= 0.5

        return total_reward

    def step(self, actions):
        """Take a step in the environment with the given actions."""
        if len(actions) != 4:
            raise ValueError(f"Expected 4 ghost actions, got {len(actions)}")

        # Reset ghost positions for this step
        self.ghost_positions = []
        reward = 0

        # Move each ghost
        for ghost_idx, action in enumerate(actions):
            if not isinstance(action, list) or len(action) != 4:
                raise ValueError(f"Invalid action format for ghost {ghost_idx}")

            # Get action index (convert one-hot to index)
            action_idx = action.index(1)

            # Get current ghost position
            ghost_char = chr(ord('a') + ghost_idx)
            ghost_pos = None
            for i in range(len(self.board)):
                for j in range(len(self.board[0])):
                    if self.board[i][j] == ghost_char:
                        ghost_pos = (i, j)
                        break
                if ghost_pos:
                    break

            if ghost_pos:
                # Get valid moves for this ghost
                valid_moves = self.get_valid_moves(ghost_pos)

                # Map action index to direction
                directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # left, right, up, down
                move = directions[action_idx]

                # Apply move if valid
                new_pos = (ghost_pos[0] + move[0], ghost_pos[1] + move[1])
                if new_pos in valid_moves:
                    # Update board
                    self.board[ghost_pos[0]][ghost_pos[1]] = ' '
                    self.board[new_pos[0]][new_pos[1]] = ghost_char
                    self.ghost_positions.append(new_pos)
                else:
                    # If invalid move, stay in place
                    self.ghost_positions.append(ghost_pos)

        # Calculate rewards based on new positions
        reward = self._calculate_ghost_rewards()

        # Convert board to numerical representation and ensure it's a tensor
        next_state = self._convert_board_to_numerical(self.board)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state)
        next_state = next_state.detach().clone()  # Ensure we have a copy

        # Check if episode is done (Pacman caught or all dots eaten)
        done = self._check_done()

        return next_state, reward, done

    def _move_pacman(self):
        """Simple Pacman AI for training ghosts"""
        if not self.pacman_pos:
            return

        # Get valid moves for Pacman
        valid_moves = self.get_valid_moves(self.pacman_pos)
        if not valid_moves:
            return

        # Simple movement: try to move away from ghosts
        best_move = None
        max_distance = -1

        for move in valid_moves:
            min_ghost_distance = float('inf')
            for ghost_pos in self.ghost_positions:
                distance = self._manhattan_distance(move, ghost_pos)
                min_ghost_distance = min(min_ghost_distance, distance)

            if min_ghost_distance > max_distance:
                max_distance = min_ghost_distance
                best_move = move

        if best_move:
            # Update board
            self.board[self.pacman_pos[0]][self.pacman_pos[1]] = ' '
            self.board[best_move[0]][best_move[1]] = 'P'
            self.pacman_pos = best_move

    def _check_done(self):
        """Check if episode is done."""
        # Check if Pacman is caught
        if any(pos == self.pacman_pos for pos in self.ghost_positions):
            return True

        # Check if all dots are eaten
        for row in self.board:
            if '.' in row:
                return False
        return True


    def get_valid_moves(self, position):
        """Get valid moves for a given position."""
        valid_moves = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # left, right, up, down

        for dx, dy in directions:
            new_x = position[0] + dx
            new_y = position[1] + dy

            # Check if move is within bounds and not into a wall
            if (0 <= new_x < len(self.board) and
                0 <= new_y < len(self.board[0]) and
                self.board[new_x][new_y] != '#'):

                # For ghosts, also check collision with other ghosts
                if position in self.ghost_positions:
                    if not any((new_x, new_y) == ghost_pos for ghost_pos in self.ghost_positions):
                        valid_moves.append((new_x, new_y))
                else:
                    valid_moves.append((new_x, new_y))

        return valid_moves
