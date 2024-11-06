import random
import numpy as np
import torch
from collections import namedtuple

# Define experience tuple structure
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling weight
        self.beta_increment = 0.001
        self.epsilon = 1e-6  # Small constant to avoid zero priority

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        max_priority = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(Experience(state, action, reward, next_state, done))
        else:
            self.memory[self.position] = Experience(state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of experiences based on their priorities."""
        if len(self.memory) < batch_size:
            return None, None, None

        # Calculate sampling probabilities
        total_priority = np.sum(self.priorities[:len(self.memory)])
        if total_priority == 0:
            probs = np.ones(len(self.memory)) / len(self.memory)
        else:
            probs = self.priorities[:len(self.memory)] / total_priority

        # Sample indices based on priorities
        indices = np.random.choice(len(self.memory), batch_size, p=probs)

        # Calculate importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** -0.4
        weights = weights / weights.max()  # Normalize weights

        # Get experiences for the sampled indices
        experiences = [self.memory[idx] for idx in indices]

        return experiences, indices, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities for sampled experiences."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error.item()) + self.epsilon

    def __len__(self):
        return len(self.memory)

def process_experience_batch(experiences, device):
    """Process a batch of experiences."""
    # Unpack experiences into separate batches
    batch = Experience(*zip(*experiences))

    # Process states and next_states - handle both tensor and array inputs
    state_batch = torch.stack([s.clone() if isinstance(s, torch.Tensor) else torch.FloatTensor(s) for s in batch.state]).to(device)
    next_state_batch = torch.stack([s.clone() if isinstance(s, torch.Tensor) else torch.FloatTensor(s) for s in batch.next_state]).to(device)

    # Remove extra dimension if present (ensure shape is [batch_size, channels, height, width])
    if state_batch.dim() == 5:
        state_batch = state_batch.squeeze(2)
    if next_state_batch.dim() == 5:
        next_state_batch = next_state_batch.squeeze(2)

    # Process actions - convert one-hot lists to indices and reshape
    action_batch = []
    for action_set in batch.action:
        ghost_actions = []
        for ghost_action in action_set:  # action_set is list of 4 one-hot arrays
            # Convert one-hot to index
            ghost_actions.append(ghost_action.index(1))
        action_batch.append(ghost_actions)

    # Convert to tensor with shape [batch_size, num_ghosts]
    action_batch = torch.tensor(action_batch, dtype=torch.long).to(device)

    # Process rewards and done flags
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
    done_batch = torch.tensor(batch.done, dtype=torch.float32).to(device)

    return state_batch, action_batch, reward_batch, next_state_batch, done_batch
