import socketio
import requests
import time
import os
import torch
from model_architecture import GhostCNN, convert_move_to_array, get_valid_moves

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GhostCNN().to(device)
weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'ghost_model_best.pth')
if os.path.exists(weights_path):
    print(f"Loading pre-trained weights from {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
else:
    print("Warning: Pre-trained weights not found!")
model.eval()

link = "http://127.0.0.1:5000"

token = input('Enter your token: ')
name = input('Enter your name: ')

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

def find_ghosts(board):
    ghost_positions = {}
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] in ['a', 'b', 'c', 'd']:
                ghost_positions[board[i][j]] = (i, j)
    return ghost_positions

def process(board, points):
    with torch.no_grad():
        if not isinstance(board, torch.Tensor):
            board_tensor = model.preprocess_board(board).unsqueeze(0).to(device)
        else:
            board_tensor = board.to(device)

        ghost_moves_probs = model(board_tensor)
        ghost_moves_probs = ghost_moves_probs.squeeze(-1).cpu()
        moves = [convert_move_to_array(ghost_move) for ghost_move in ghost_moves_probs[0]]

        ghost_positions = find_ghosts(board)
        if ghost_positions:
            for i, (ghost, pos) in enumerate(sorted(ghost_positions.items())):
                valid_moves = get_valid_moves(board, pos)
                if valid_moves[moves[i].index('1')] == 0:
                    for j, is_valid in enumerate(valid_moves):
                        if is_valid:
                            moves[i] = ['0', '0', '0', '0']
                            moves[i][j] = '1'
                            break

    send_move(moves)

def send_move(moves):
    url = f"{link}/move/ghost"
    payload = moves
    headers = {'content-type': 'application/json', 'Authorization': f'Bearer {token}'}
    response = requests.post(url, json=payload, headers=headers)
    print(response.text)

sio.connect(link, headers={'Authorization': f'Bearer {token}', 'Name': name})
sio.wait()
