import torch
from model_architecture import PacmanCNN, convert_move_to_array
import os

class PacmanModelLoader:
    def __init__(self, model_dir='saved_models'):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.epsilon = 0.01  # Fixed exploration rate for evaluation

    def load_model(self, model_type='best'):
        """Load either 'best' or 'final' model"""
        filename = 'best_pacman_model.pth' if model_type == 'best' else 'final_pacman_model.pth'
        filepath = os.path.join(self.model_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Initialize model
        self.model = PacmanCNN().to(self.device)

        # Load weights
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['policy_net_state_dict'])
        self.model.eval()  # Set to evaluation mode

        print(f"Loaded {model_type} model from {filepath}")
        print(f"Best reward achieved during training: {checkpoint.get('best_reward', 'N/A')}")
        return self.model

    def get_action(self, board_state):
        """Get action for a given board state"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        with torch.no_grad():
            # Preprocess board state
            state_tensor = self.model.preprocess_board(board_state).unsqueeze(0).to(self.device)

            # Get Q-values
            q_values = self.model(state_tensor)

            # Select action with highest Q-value
            action = q_values.max(1)[1].item()

            # Convert to move array format
            return convert_move_to_array(action)

def load_and_test_model(model_type='best'):
    """Utility function to load and test the model"""
    loader = PacmanModelLoader()
    try:
        model = loader.load_model(model_type)
        print("Model loaded successfully")
        print("Model architecture:", model)
        return loader
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

if __name__ == "__main__":
    # Test loading both models
    print("Testing best model:")
    best_loader = load_and_test_model('best')

    print("\nTesting final model:")
    final_loader = load_and_test_model('final')
