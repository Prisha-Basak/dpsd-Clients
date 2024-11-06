import socketio
import requests
import time
import os
import torch
from model_architecture import PacmanCNN, convert_move_to_array, get_valid_moves

# Initialize the model
model = PacmanCNN()
# Pre-trained weights would be loaded here if available
model.eval()

link = "http://127.0.0.1:5000"

# Get authentication details
token = input('Enter your token: ')
name = input('Enter your name: ')

# Initialize Socket.IO client
sio = socketio.Client()
connected = False

@sio.event
def connect():
    global connected
    connected = True
    print("Connected to server")
    sio.emit('request')

@sio.event
def disconnect():
    global connected
    connected = False
    print("Disconnected from server")
    os._exit(0)

@sio.event
def reconnect():
    global connected
    connected = True
    print("Reconnected to server")

@sio.on('reset')
def reset():
    print("Resetting")
    os._exit(0)

@sio.on('board')
def handle_server_message(data):
    if not connected:
        print("Not connected yet, ignoring message")
        return
    board, points = data
    process(board, points)

def find_pacman(board):
    """Find Pacman's position on the board"""
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 'P':
                return (i, j)
    return None

def process(board, points):
    """Process the current game state and make a move decision"""
    with torch.no_grad():
        # Get model's move prediction
        move_probs = model(board)

        # Convert probabilities to required move format
        move = convert_move_to_array(move_probs[0])  # [0] to get first batch item

        # Safety checks for Pacman movement
        pacman_pos = find_pacman(board)
        if pacman_pos:
            valid_moves = get_valid_moves(board, pacman_pos)
            # If the chosen move is invalid, select the first valid move
            if valid_moves[move.index(1)] == 0:
                for i, is_valid in enumerate(valid_moves):
                    if is_valid:
                        move = [0, 0, 0, 0]
                        move[i] = 1
                        break

            # Additional safety: avoid ghosts if possible
            if move.index(1) >= 0:  # Ensure we have a valid move
                next_pos = list(pacman_pos)
                directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # up, down, right, left
                dir_idx = move.index(1)
                next_pos[0] += directions[dir_idx][0]
                next_pos[1] += directions[dir_idx][1]

                # Check if next position contains a ghost
                if (0 <= next_pos[0] < len(board) and
                    0 <= next_pos[1] < len(board[0]) and
                    board[next_pos[0]][next_pos[1]] in 'abcd'):
                    # Try to find a safe move
                    for i, is_valid in enumerate(valid_moves):
                        if not is_valid:
                            continue
                        test_pos = [pacman_pos[0] + directions[i][0],
                                  pacman_pos[1] + directions[i][1]]
                        if (0 <= test_pos[0] < len(board) and
                            0 <= test_pos[1] < len(board[0]) and
                            board[test_pos[0]][test_pos[1]] not in 'abcd'):
                            move = [0, 0, 0, 0]
                            move[i] = 1
                            break

    # Send the chosen move
    send_move(move)

def send_move(move):
    url = f"{link}/move/player"
    payload = move
    headers = {'content-type': 'application/json', 'Authorization': f'Bearer {token}'}
    response = requests.post(url, json=payload, headers=headers)
    print(response.text)

# Connect to the server
sio.connect(link, headers={'Authorization': f'Bearer {token}', 'Name': name})

# Wait for events
sio.wait()
