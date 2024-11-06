import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from model_architecture import PacmanCNN, convert_move_to_array, get_valid_moves
import json
import os

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 10000
LEARNING_RATE = 0.001
NUM_EPISODES = 1000

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def preprocess_board(board):
    """Convert board characters to numerical values"""
    processed = np.zeros((31, 28))
    for i in range(len(board)):
        for j in range(len(board[i])):
            cell = board[i][j]
            if cell == '#':  # Wall
                processed[i][j] = -1
            elif cell == '.':  # Food
                processed[i][j] = 0.5
            elif cell == 'P':  # Pacman
                processed[i][j] = 1
            elif cell in 'abcd':  # Ghosts
                processed[i][j] = -0.5
    return processed

def calculate_reward(board, next_board, points_gained):
    """Calculate reward based on state transition"""
    reward = points_gained  # Base reward from points

    # Additional rewards/penalties
    if 'P' not in str(next_board):  # Pacman died
        reward -= 50
    elif points_gained > 0:  # Successfully ate food
        reward += 5

    # Check for ghost proximity
    pacman_pos = None
    ghost_positions = []

    for i in range(len(next_board)):
        for j in range(len(next_board[i])):
            if next_board[i][j] == 'P':
                pacman_pos = (i, j)
            elif next_board[i][j] in 'abcd':
                ghost_positions.append((i, j))

    if pacman_pos:
        # Penalize being too close to ghosts
        for ghost_pos in ghost_positions:
            distance = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
            if distance < 3:
                reward -= (3 - distance) * 2

    return reward

def train_model():
    model = PacmanCNN()
    target_model = PacmanCNN()  # Target network for stable learning
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)
    epsilon = EPSILON_START

    # Training statistics
    episode_rewards = []
    episode_lengths = []

    for episode in range(NUM_EPISODES):
        # Reset environment (implement game reset logic here)
        board = initialize_game()  # You'll need to implement this
        state = preprocess_board(board)
        total_reward = 0
        steps = 0

        while True:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 3)  # Random move
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = q_values.argmax().item()

            # Convert action to move format and apply it
            move = [0] * 4
            move[action] = 1

            # Get next state and reward (implement game step logic here)
            next_board, points_gained, done = step_game(board, move)  # You'll need to implement this
            next_state = preprocess_board(next_board)
            reward = calculate_reward(board, next_board, points_gained)

            # Store transition in memory
            memory.push(state, action, reward, next_state, done)

            # Train model if enough samples are available
            if len(memory) > BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                state_batch = torch.tensor([s for s, _, _, _, _ in batch])
                action_batch = torch.tensor([a for _, a, _, _, _ in batch])
                reward_batch = torch.tensor([r for _, _, r, _, _ in batch])
                next_state_batch = torch.tensor([s for _, _, _, s, _ in batch])
                done_batch = torch.tensor([d for _, _, _, _, d in batch])

                # Compute Q values
                current_q_values = model(state_batch).gather(1, action_batch.unsqueeze(1))
                next_q_values = target_model(next_state_batch).max(1)[0].detach()
                target_q_values = reward_batch + GAMMA * next_q_values * (1 - done_batch)

                # Compute loss and update model
                loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_reward += reward
            steps += 1
            state = next_state
            board = next_board

            if done:
                break

        # Update target network periodically
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Record statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Print progress
        if episode % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            avg_length = sum(episode_lengths[-10:]) / 10
            print(f"Episode {episode}/{NUM_EPISODES}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Length: {avg_length:.2f}")
            print(f"Epsilon: {epsilon:.2f}")
            print("------------------------")

    # Save the trained model
    torch.save(model.state_dict(), 'pacman_model_weights.pth')

    # Save training statistics
    stats = {
        'rewards': episode_rewards,
        'lengths': episode_lengths
    }
    with open('training_stats.json', 'w') as f:
        json.dump(stats, f)

if __name__ == "__main__":
    train_model()
