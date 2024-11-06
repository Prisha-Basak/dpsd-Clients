import torch
from model_architecture import GhostCNN, convert_move_to_array
import numpy as np

def create_test_board():
    """Create a test board with ghosts, walls, and Pacman"""
    board = [[' ' for _ in range(28)] for _ in range(31)]
    # Add some walls
    for i in range(28):
        board[0][i] = '#'
        board[30][i] = '#'
    # Add ghosts
    board[15][14] = 'a'
    board[15][15] = 'b'
    board[16][14] = 'c'
    board[16][15] = 'd'
    # Add Pacman
    board[20][14] = 'P'
    # Add some food
    board[18][14] = '.'
    board[18][15] = '.'
    return board

def test_ghost_model():
    # Initialize model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GhostCNN().to(device)
    weights_path = 'weights/ghost_model_best.pth'

    print(f"Loading weights from {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Create test board
    board = create_test_board()

    print("\nTest board configuration:")
    for row in board:
        print(''.join(row))

    # Process board
    with torch.no_grad():
        # Preprocess and create batch
        board_tensor = model.preprocess_board(board).unsqueeze(0).to(device)

        # Get model predictions
        ghost_moves_probs = model(board_tensor)

        # Convert to CPU for processing
        ghost_moves_probs = ghost_moves_probs.squeeze(-1).cpu()

        # Convert probabilities to moves
        moves = [convert_move_to_array(ghost_move) for ghost_move in ghost_moves_probs[0]]

        # Print moves for each ghost
        ghost_names = ['a', 'b', 'c', 'd']
        print("\nPredicted moves:")
        for ghost, move in zip(ghost_names, moves):
            direction = ['up', 'down', 'right', 'left'][move.index('1')]
            print(f"Ghost {ghost}: {direction} {move}")

if __name__ == "__main__":
    test_ghost_model()
