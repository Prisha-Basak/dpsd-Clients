import subprocess
import time
import threading
from pacman_model_client import process as pacman_process
from ghost_model_client import process as ghost_process
import socketio
import sys
import os

def start_dev_server():
    """Start the development server in a separate process"""
    server_process = subprocess.Popen(
        ["python", "../Server/dev_main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for server to start and get tokens
    tokens = {'pacman': None, 'ghost': None}
    while True:
        line = server_process.stdout.readline()
        if 'Pacman Token:' in line:
            tokens['pacman'] = line.split(': ')[1].strip()
        elif 'Ghost Token:' in line:
            tokens['ghost'] = line.split(': ')[1].strip()
        if tokens['pacman'] and tokens['ghost']:
            break

    return server_process, tokens

def run_pacman_client(token):
    """Run the Pacman client"""
    sio = socketio.Client()

    @sio.event
    def connect():
        print("Pacman connected to server")

    @sio.event
    def disconnect():
        print("Pacman disconnected from server")

    @sio.on('board')
    def handle_board(data):
        board, points = data
        pacman_process(board, points)

    sio.connect('http://localhost:5000',
                headers={'Authorization': f'Bearer {token}',
                        'Name': 'PacmanAI'})
    return sio

def run_ghost_client(token):
    """Run the Ghost client"""
    sio = socketio.Client()

    @sio.event
    def connect():
        print("Ghosts connected to server")

    @sio.event
    def disconnect():
        print("Ghosts disconnected from server")

    @sio.on('board')
    def handle_board(data):
        board, points = data
        ghost_process(board, points)

    sio.connect('http://localhost:5000',
                headers={'Authorization': f'Bearer {token}',
                        'Name': 'GhostAI'})
    return sio

def main():
    print("Starting model testing...")

    # Start development server
    server_process, tokens = start_dev_server()
    print(f"Server started with tokens: {tokens}")

    try:
        # Wait for server to fully initialize
        time.sleep(2)

        # Start Pacman and Ghost clients
        pacman_client = run_pacman_client(tokens['pacman'])
        ghost_client = run_ghost_client(tokens['ghost'])

        print("Test running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping test...")
    finally:
        # Cleanup
        try:
            pacman_client.disconnect()
            ghost_client.disconnect()
        except:
            pass
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main()
