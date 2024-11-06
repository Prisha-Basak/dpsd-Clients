import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model_architecture import PacmanCNN
from replay_memory import PrioritizedReplayMemory, process_experience_batch
from game_environment import PacmanEnvironment
import json
import os

# Hyperparameters
BATCH_SIZE = 128  # Increased batch size for stability
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.999  # Slower decay
MEMORY_SIZE = 100000
LEARNING_RATE = 0.00005  # Reduced learning rate
TARGET_UPDATE = 10
NUM_EPISODES = 10000

class DQNTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PacmanCNN().to(self.device)
        self.target_net = PacmanCNN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = PrioritizedReplayMemory(MEMORY_SIZE)
        self.epsilon = EPSILON_START

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(4)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()

    def optimize_model(self):
        """Perform one step of optimization"""
        if len(self.memory) < BATCH_SIZE:
            return

        experiences, indices, weights = self.memory.sample(BATCH_SIZE)
        if experiences is None:
            return

        # Process batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            process_experience_batch(experiences, self.device)

        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + GAMMA * next_q_values * (1 - done_batch)

        # Compute loss with importance sampling weights
        td_errors = target_q_values.unsqueeze(1) - current_q_values
        loss = (weights.unsqueeze(1) * F.smooth_l1_loss(current_q_values,
                                                       target_q_values.unsqueeze(1),
                                                       reduction='none')).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Update priorities in memory
        self.memory.update_priorities(indices, td_errors.detach())

        return loss.item()

    def train_episode(self, env_state, max_steps=1000):
        """Train for one episode"""
        total_reward = 0
        steps = 0
        state = self.policy_net.preprocess_board(env_state).numpy()
        done = False

        print(f"\nStarting episode with epsilon: {self.epsilon:.3f}")

        while not done and steps < max_steps:
            # Select and perform action
            action = self.select_action(state)

            # Get next state and reward from environment
            next_state, reward, done = self.step_environment(env_state, action)
            next_state = self.policy_net.preprocess_board(next_state).numpy()

            # Store transition in memory
            self.memory.push(state, action, reward, next_state, done)

            # Move to next state
            state = next_state
            total_reward += reward
            steps += 1

            # Perform optimization step
            if steps % 4 == 0:  # Update every 4 steps
                loss = self.optimize_model()
                if loss is not None:
                    self.losses.append(loss)
                    if steps % 100 == 0:  # Print every 100 steps
                        print(f"Step {steps}, Loss: {loss:.4f}, Total Reward: {total_reward}")

        print(f"Episode finished after {steps} steps with total reward: {total_reward}")
        return total_reward, steps

    def train(self, num_episodes=NUM_EPISODES):
        """Main training loop"""
        print("\nStarting training...")
        print(f"Device: {self.device}")
        print(f"Episodes: {num_episodes}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Learning rate: {LEARNING_RATE}")
        print("------------------------")

        best_reward = float('-inf')
        episode_count = 0
        no_improvement_count = 0

        for episode in range(num_episodes):
            # Initialize environment
            env_state = self.initialize_environment()

            # Train one episode
            total_reward, steps = self.train_episode(env_state)

            # Record statistics
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)

            # Update target network
            if episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print("Target network updated")

            # Decay epsilon
            self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

            # Track best model
            if total_reward > best_reward:
                best_reward = total_reward
                self.save_model(filename='best_pacman_model.pth')
                print(f"New best model saved with reward: {best_reward:.2f}")
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Print detailed progress every 10 episodes
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                print(f"\nEpisode {episode}/{num_episodes}")
                print(f"Average Reward (last 10): {avg_reward:.2f}")
                print(f"Average Length (last 10): {avg_length:.2f}")
                print(f"Average Loss (last 100): {avg_loss:.4f}")
                print(f"Current Epsilon: {self.epsilon:.3f}")
                print(f"Best Reward: {best_reward:.2f}")
                print("------------------------")

            # Early stopping if no improvement for a while
            if no_improvement_count >= 50:
                print("\nStopping early due to no improvement in rewards")
                break

            episode_count = episode + 1

        print(f"\nTraining completed after {episode_count} episodes")
        print(f"Best reward achieved: {best_reward:.2f}")

        # Save final model and statistics
        self.save_model(filename='final_pacman_model.pth')
        self.save_statistics()
        print("Model and statistics saved")

    def save_model(self, filename='pacman_model_weights.pth'):
        """Save the trained model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_reward': max(self.episode_rewards) if self.episode_rewards else float('-inf')
        }, filename)
        print(f"Model saved to {filename}")

    def save_statistics(self):
        """Save training statistics"""
        stats = {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths,
            'losses': self.losses
        }
        with open('training_stats.json', 'w') as f:
            json.dump(stats, f)

    def initialize_environment(self):
        """Initialize the game environment"""
        if not hasattr(self, 'env'):
            self.env = PacmanEnvironment()
        return self.env.reset()

    def step_environment(self, state, action):
        """Take a step in the environment"""
        next_state, reward, done = self.env.step(action)
        return next_state, reward, done

if __name__ == "__main__":
    trainer = DQNTrainer()
    trainer.train()
