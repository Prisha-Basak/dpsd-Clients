import socketio
import time
import torch
from model_architecture import PacmanCNN, GhostCNN, convert_move_to_array

def run_pacman_client(token):
    sio = socketio.Client()
    model = PacmanCNN()
    model.eval()

    @sio.event
    def connect():
        print('Pacman connected')
        sio.emit('request')

    @sio.event
    def disconnect():
        print('Pacman disconnected')

    @sio.on('board')
    def on_board(data):
        board, points = data
        print(f'Pacman received board update. Points: {points}')
        with torch.no_grad():
            move_probs = model(board)
            move = convert_move_to_array(move_probs[0])
            sio.emit('move', move)

    sio.connect('http://localhost:5000',
                headers={'Authorization': f'Bearer {token}', 'Name': 'PacmanTest'})
    return sio

def run_ghost_client(token):
    sio = socketio.Client()
    model = GhostCNN()
    model.eval()

    @sio.event
    def connect():
        print('Ghost connected')
        sio.emit('request')

    @sio.event
    def disconnect():
        print('Ghost disconnected')

    @sio.on('board')
    def on_board(data):
        board, points = data
        print('Ghost received board update')
        with torch.no_grad():
            ghost_moves_probs = model(board)
            moves = [convert_move_to_array(ghost_move) for ghost_move in ghost_moves_probs]
            sio.emit('move', moves)

    sio.connect('http://localhost:5000',
                headers={'Authorization': f'Bearer {token}', 'Name': 'GhostTest'})
    return sio

def main():
    # Use the tokens from the server output
    pacman_token = '2ZyGQ6NumcwvWHv8'
    ghost_token = 'dhP2rA7J6Ul1BMt0'

    try:
        print("Starting test clients...")
        pacman_client = run_pacman_client(pacman_token)
        time.sleep(1)  # Small delay between connections
        ghost_client = run_ghost_client(ghost_token)

        print("Both clients connected. Running game...")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping clients...")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        try:
            pacman_client.disconnect()
            ghost_client.disconnect()
        except:
            pass

if __name__ == "__main__":
    main()
