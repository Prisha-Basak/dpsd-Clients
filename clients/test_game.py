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

    @sio.event
    def connect_error(data):
        print(f'Pacman connection error: {data}')

    @sio.on('board')
    def on_board(data):
        try:
            board, points = data
            print(f'\nPacman received board update. Points: {points}')
            print("Current board state:")
            for row in board:
                print(row)

            # Convert board strings to list of lists
            board_list = [list(row) for row in board]
            with torch.no_grad():
                move_probs = model(board_list)
                move = convert_move_to_array(move_probs[0])
                print(f'Pacman move probabilities: {move_probs[0].tolist()}')
                print(f'Pacman selected move: {move}')
                sio.emit('move', move)
        except Exception as e:
            print(f'Error processing Pacman move: {e}')
            print(f'Board data type: {type(board)}')
            print(f'Board sample: {board[:2] if board else None}')

    @sio.on('playerdocked')
    def on_player_docked():
        print('Pacman move docked')

    @sio.on('undock')
    def on_undock():
        print('Move cycle complete')

    sio.connect('http://localhost:5000',
                headers={
                    'Authorization': f'Bearer {token}',
                    'Name': 'PacmanTest'
                },
                wait_timeout=10)
    return sio

def main():
    # Use the tokens from the server output
    pacman_token = 'enn9uYKsb0v999gJ'
    ghost_token = 'xpfFTnjuam54h7un'

    try:
        print("Starting test clients...")
        pacman_client = run_pacman_client(pacman_token)
        print("Pacman client started")
        time.sleep(2)  # Increased delay between connections
        ghost_client = run_ghost_client(ghost_token)
        print("Ghost client started")

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

def main():
    # Use the tokens from the server output
    pacman_token = 'HGYToL5XlHGGt8Rp'
    ghost_token = 'uJ69ohavE04jGXTt'

    try:
        print("Starting test clients...")
        pacman_client = run_pacman_client(pacman_token)
        print("Pacman client started")
        time.sleep(2)  # Increased delay between connections
        ghost_client = run_ghost_client(ghost_token)
        print("Ghost client started")

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
