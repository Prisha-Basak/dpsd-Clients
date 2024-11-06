import socketio
import requests
import os
import time 

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

@sio.on('board')
def handle_server_message(data):
   
    if not connected:
        print("Not connected yet,ignoring message")
        return
    board,points = data
    process(board,points)

@sio.on('reset')
def reset():

    print("Resetting")
    os._exit(0)   

def process(board,points):
    move = [[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1]]
    time.sleep(5)
    send_move(move)

def send_move(move):
   url = f"{link}/move/ghost"
   payload = move
   headers = {'content-type': 'application/json', 'Authorization': f'Bearer {token}'}
   response = requests.post(url, json=payload, headers=headers)
   print(response.text)


sio.connect(link,headers={'Authorization': f'Bearer {token}','Name':name})

sio.wait()

