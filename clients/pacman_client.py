import socketio
import requests
import time
import os
import torch
from model_architecture import PacmanCNN, convert_move_to_array

class PacmanClient:
    def __init__(self, token, name="PacmanAI", server_url="http://127.0.0.1:5000"):
        self.token = token
        self.name = name
        self.link = server_url
        self.connected = False

        # Initialize the model
        self.model = PacmanCNN()
        self.model.eval()

        # Initialize socketio client
        self.sio = socketio.Client()
        self.setup_socket_events()

    def setup_socket_events(self):
        @self.sio.event
        def connect():
            self.connected = True
            print("Pacman connected to server")
            self.sio.emit('request')

        @self.sio.event
        def disconnect():
            self.connected = False
            print("Pacman disconnected from server")

        @self.sio.event
        def reconnect():
            self.connected = True
            print("Pacman reconnected to server")

        @self.sio.on('reset')
        def reset():
            print("Resetting Pacman")
            self.disconnect()

        @self.sio.on('board')
        def handle_server_message(data):
            if not self.connected:
                print("Pacman not connected yet, ignoring message")
                return
            board, points = data
            self.process(board, points)

    def find_pacman(self, board):
        """Find Pacman's position on the board"""
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == 'P':
                    return (i, j)
        return None

    def process(self, board, points):
        """Process the current game state and make a move decision"""
        with torch.no_grad():
            # Get model's move prediction
            move_probs = self.model(board)

            # Convert probabilities to required move format
            move = convert_move_to_array(move_probs[0])  # [0] to get first batch item

            # Optional: Add safety checks
            pacman_pos = self.find_pacman(board)
            if pacman_pos:
                # TODO: Implement additional safety checks here
                pass

        # Send the chosen move
        self.send_move(move)

    def send_move(self, move):
        """Send move to the server"""
        url = f"{self.link}/move/player"
        headers = {
            'content-type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }
        response = requests.post(url, json=move, headers=headers)
        print(f"Pacman move response: {response.text}")

    def connect(self):
        """Connect to the game server"""
        self.sio.connect(
            self.link,
            headers={
                'Authorization': f'Bearer {self.token}',
                'Name': self.name
            }
        )

    def disconnect(self):
        """Disconnect from the game server"""
        if self.sio.connected:
            self.sio.disconnect()

    def wait(self):
        """Wait for events"""
        self.sio.wait()

# For backwards compatibility when running as standalone
if __name__ == "__main__":
    token = input('Enter your token: ')
    name = input('Enter your name: ')
    client = PacmanClient(token=token, name=name)
    client.connect()
    client.wait()
