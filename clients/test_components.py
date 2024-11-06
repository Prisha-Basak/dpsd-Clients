import torch
from model_architecture import PacmanCNN
from game_environment import PacmanEnvironment
from replay_memory import PrioritizedReplayMemory

def test_environment():
    print("\nTesting Environment...")
    env = PacmanEnvironment()
    board = env.reset()
    print("Initial board shape:", len(board), "x", len(board[0]))
    print("Valid moves:", env.get_valid_moves())

    # Test step function
    action = 2  # Try moving right
    next_state, reward, done = env.step(action)
    print("Step result - Reward:", reward, "Done:", done)
    return env

def test_model():
    print("\nTesting Model...")
    model = PacmanCNN()
    env = PacmanEnvironment()
    board = env.reset()

    # Test forward pass
    state_tensor = model.preprocess_board(board)
    print("Preprocessed state shape:", state_tensor.shape)

    # Test model output
    with torch.no_grad():
        q_values = model(state_tensor.unsqueeze(0))
        print("Q-values shape:", q_values.shape)
        print("Q-values:", q_values)
    return model

def test_memory():
    print("\nTesting Replay Memory...")
    memory = PrioritizedReplayMemory(1000)
    env = PacmanEnvironment()

    # Add some experiences
    state = env.reset()
    for _ in range(10):
        action = 2
        next_state, reward, done = env.step(action)
        memory.push(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

    print("Memory size:", len(memory))
    if len(memory) >= 8:
        experiences, indices, weights = memory.sample(8)
        print("Sample batch size:", len(experiences))
        print("Weights shape:", weights.shape)
    return memory

if __name__ == "__main__":
    print("Starting component tests...")

    try:
        env = test_environment()
        print("✓ Environment test completed")
    except Exception as e:
        print("✗ Environment test failed:", str(e))

    try:
        model = test_model()
        print("✓ Model test completed")
    except Exception as e:
        print("✗ Model test failed:", str(e))

    try:
        memory = test_memory()
        print("✓ Memory test completed")
    except Exception as e:
        print("✗ Memory test failed:", str(e))

    print("\nComponent tests completed")
