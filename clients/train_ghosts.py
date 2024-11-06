import torch
from train_ghost_dqn import GhostTrainer
import os

def main():
    # Create weights directory if it doesn't exist
    os.makedirs('weights', exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize trainer with optimized hyperparameters
    trainer = GhostTrainer(
        device=device,
        batch_size=64,
        gamma=0.99,
        lr=0.0001,
        target_update=10
    )

    # Train the model
    print("Starting ghost training...")
    trainer.train(
        num_episodes=5000,  # Reduced episodes but we'll use early stopping
        max_steps_per_episode=1000
    )

    print("Training completed!")

if __name__ == "__main__":
    main()
