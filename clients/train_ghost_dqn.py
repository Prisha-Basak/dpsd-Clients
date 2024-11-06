import os
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model_architecture import GhostCNN
from ghost_environment import GhostEnvironment
from replay_memory import PrioritizedReplayMemory, process_experience_batch, Experience

# Create weights directory if it doesn't exist
os.makedirs('weights', exist_ok=True)

class GhostTrainer:
    """Trainer for ghost agents using DQN."""

    def __init__(self, device='cpu', batch_size=64, gamma=0.99, lr=0.0001, target_update=10):
        """Initialize the trainer."""
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.patience = 1000  # Episodes without improvement before early stopping

        # Initialize environment
        self.env = GhostEnvironment()

        # Initialize networks
        self.policy_net = GhostCNN().to(device)
        self.target_net = GhostCNN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Initialize optimizer with gradient clipping
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.max_grad_norm = 1.0  # For gradient clipping

        # Initialize replay memory
        self.memory = PrioritizedReplayMemory(100000)

        # Initialize epsilon values for exploration
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.9995
        self.epsilon = self.epsilon_start

        # Initialize rewards tracking
        self.rewards_window = deque(maxlen=100)
        self.best_reward = float('-inf')

    def select_actions(self, state, eval_mode=False):
        """Select actions for all ghosts."""
        # Ensure state is a tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        state = state.to(self.device)

        # No exploration during evaluation
        if eval_mode:
            self.policy_net.eval()
            with torch.no_grad():
                q_values = self.policy_net(state)
                # Select best actions for each ghost
                actions = []
                for ghost_idx in range(4):
                    ghost_q_values = q_values[0, ghost_idx]
                    action_idx = ghost_q_values.argmax().item()
                    # Convert to one-hot
                    action = [0, 0, 0, 0]
                    action[action_idx] = 1
                    actions.append(action)
            self.policy_net.train()
            return actions

        # Epsilon-greedy during training
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                actions = []
                for ghost_idx in range(4):
                    ghost_q_values = q_values[0, ghost_idx]
                    action_idx = ghost_q_values.argmax().item()
                    # Convert to one-hot
                    action = [0, 0, 0, 0]
                    action[action_idx] = 1
                    actions.append(action)
        else:
            # Random actions
            actions = []
            for _ in range(4):
                action = [0, 0, 0, 0]
                action[random.randrange(4)] = 1
                actions.append(action)

        return actions

    def optimize_model(self):
        """Perform one step of optimization."""
        if len(self.memory) < self.batch_size:
            return None

        # Sample from replay memory
        experiences, indices, weights = self.memory.sample(self.batch_size)
        if experiences is None:
            return None

        # Process experience batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = process_experience_batch(experiences, self.device)

        # Compute current Q values
        current_q_values = self.policy_net(state_batch)  # Shape: [batch_size, 4, 4, 1]

        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)  # Shape: [batch_size, 4, 4, 1]
            # Get max Q-value for each ghost
            next_q_values = next_q_values.max(2)[0]  # Shape: [batch_size, 4, 1]

        # Reshape tensors for broadcasting
        done_batch = done_batch.unsqueeze(1).unsqueeze(2)  # Shape: [batch_size, 1, 1]
        reward_batch = reward_batch.unsqueeze(1).unsqueeze(2)  # Shape: [batch_size, 1, 1]

        # Compute expected Q values
        expected_q_values = (next_q_values * self.gamma * (1 - done_batch) +
                           reward_batch)  # Shape: [batch_size, 4, 1]

        # Expand expected_q_values to match current_q_values shape
        expected_q_values = expected_q_values.unsqueeze(2)  # Shape: [batch_size, 4, 1, 1]
        expected_q_values = expected_q_values.expand(-1, -1, 4, -1)  # Shape: [batch_size, 4, 4, 1]

        # Compute loss with importance sampling weights
        weights = torch.FloatTensor(weights).to(self.device)
        weights = weights.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Shape: [batch_size, 1, 1, 1]

        loss = (weights * torch.nn.functional.smooth_l1_loss(current_q_values,
                                                           expected_q_values.detach(),
                                                           reduction='none')).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # Update priorities in replay memory
        td_errors = torch.abs(current_q_values - expected_q_values).detach().cpu().numpy().mean(axis=(1,2,3))
        self.memory.update_priorities(indices, td_errors)

        return loss.item()

    def train(self, num_episodes=10000, max_steps_per_episode=1000):
        """Train the ghost agents."""
        from tqdm import tqdm
        import json
        from datetime import datetime

        # Create training log file
        log_file = f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        training_log = []

        print("Starting training...")
        best_eval_reward = float('-inf')
        episodes_without_improvement = 0
        eval_frequency = 100  # Evaluate every 100 episodes

        # Create progress bar for episodes
        progress_bar = tqdm(range(num_episodes), desc="Training Progress")

        for episode in progress_bar:
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0

            # Training phase
            self.policy_net.train()

            # Progress bar for steps
            step_bar = tqdm(range(max_steps_per_episode),
                          desc=f"Episode {episode}",
                          leave=False)

            for step in step_bar:
                # Select actions
                actions = self.select_actions(state)

                # Take step in environment
                next_state, reward, done = self.env.step(actions)

                # Store transition in memory - unpack experience tuple
                self.memory.push(state, actions, reward, next_state, done)

                # Move to next state
                state = next_state
                episode_reward += reward
                steps += 1  # Increment steps counter

                # Perform optimization step
                loss = self.optimize_model()
                if loss is not None:
                    episode_loss += loss

                # Update step progress bar
                step_bar.set_postfix({
                    'Reward': f'{episode_reward:.2f}',
                    'Loss': f'{loss:.4f}' if loss else 'N/A'
                })

                if done:
                    break

            # Close step progress bar
            step_bar.close()

            # Decay epsilon
            self.epsilon = max(self.epsilon_end,
                             self.epsilon * self.epsilon_decay)

            # Update target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Calculate metrics
            avg_loss = episode_loss / steps if steps > 0 else 0
            metrics = {
                'episode': episode,
                'steps': steps,
                'reward': episode_reward,
                'avg_loss': avg_loss,
                'epsilon': self.epsilon
            }

            # Evaluation phase
            if episode > 0 and episode % eval_frequency == 0:
                eval_reward = self._evaluate(num_episodes=5)
                metrics['eval_reward'] = eval_reward

                # Save best model
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    episodes_without_improvement = 0
                    print("\nNew best model! Saving...")
                    torch.save(self.policy_net.state_dict(),
                             'weights/ghost_model_best.pth')
                else:
                    episodes_without_improvement += eval_frequency

                # Early stopping
                if episodes_without_improvement >= self.patience:
                    print("\nEarly stopping triggered!")
                    break

            # Regular checkpointing
            if episode % 1000 == 0:
                torch.save(self.policy_net.state_dict(),
                         f'weights/ghost_model_checkpoint_{episode}.pth')

            # Update progress bar
            progress_bar.set_postfix({
                'Reward': f'{episode_reward:.2f}',
                'Loss': f'{avg_loss:.4f}',
                'Epsilon': f'{self.epsilon:.4f}',
                'Best': f'{best_eval_reward:.2f}'
            })

            # Log metrics
            training_log.append(metrics)
            with open(log_file, 'w') as f:
                json.dump(training_log, f, indent=2)

        print("\nTraining completed!")
        return self.policy_net

    def _evaluate(self, num_episodes=5):
        """Evaluate the current policy."""
        self.policy_net.eval()
        total_reward = 0

        with torch.no_grad():
            for _ in range(num_episodes):
                state = self.env.reset()
                episode_reward = 0
                done = False
                steps = 0

                while not done and steps < 1000:
                    actions = self.select_actions(state, eval_mode=True)
                    next_state, reward, done = self.env.step(actions)
                    state = next_state
                    episode_reward += reward
                    steps += 1

                total_reward += episode_reward

        self.policy_net.train()
        return total_reward / num_episodes

if __name__ == "__main__":
    trainer = GhostTrainer()
    trainer.train()
