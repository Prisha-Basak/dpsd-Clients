import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PacmanCNN(nn.Module):
    def __init__(self):
        super(PacmanCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.advantage_hidden = nn.Linear(64 * 31 * 28, 256)
        self.value_hidden = nn.Linear(64 * 31 * 28, 256)

        self.advantage = nn.Linear(256, 4)
        self.value = nn.Linear(256, 1)

        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.preprocess_board(x)
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
        x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        advantage = F.relu(self.ln1(self.advantage_hidden(x)))
        value = F.relu(self.ln2(self.value_hidden(x)))

        advantage = self.advantage(advantage)
        value = self.value(value)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def preprocess_board(self, board):
        mapping = {
            'P': 1.0,
            '.': 0.5,
            '#': -1.0,
            ' ': 0.1,
            'a': -0.8,
            'b': -0.8,
            'c': -0.8,
            'd': -0.8
        }
        return torch.tensor([[mapping.get(cell, 0.0) for cell in row] for row in board], dtype=torch.float32)

class GhostCNN(nn.Module):
    def __init__(self):
        super(GhostCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 31 * 28, 256)

        self.advantage_hidden = nn.ModuleList([
            nn.Linear(256, 64) for _ in range(4)
        ])
        self.value_hidden = nn.ModuleList([
            nn.Linear(256, 64) for _ in range(4)
        ])

        self.advantage = nn.ModuleList([
            nn.Linear(64, 4) for _ in range(4)
        ])
        self.value = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(4)
        ])

        self.ln_adv = nn.ModuleList([
            nn.LayerNorm(64) for _ in range(4)
        ])
        self.ln_val = nn.ModuleList([
            nn.LayerNorm(64) for _ in range(4)
        ])

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.preprocess_board(x)
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            x = x.unsqueeze(1)
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 5:
            x = x.squeeze(0)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        ghost_q_values = []
        for i in range(4):
            advantage = F.relu(self.ln_adv[i](self.advantage_hidden[i](x)))
            value = F.relu(self.ln_val[i](self.value_hidden[i](x)))

            advantage = self.advantage[i](advantage)
            value = self.value[i](value)

            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            ghost_q_values.append(q_values)

        result = torch.stack(ghost_q_values, dim=1)
        result = result.unsqueeze(-1)
        return result

    def preprocess_board(self, board):
        mapping = {
            'P': -0.7,
            '.': 0.3,
            '#': -1.0,
            ' ': 0.0,
            'a': 0.8,
            'b': 0.8,
            'c': 0.8,
            'd': 0.8
        }
        return torch.tensor([[mapping.get(cell, 0.0) for cell in row] for row in board], dtype=torch.float32)

def convert_move_to_array(move):
    move_array = ['0', '0', '0', '0']
    if isinstance(move, torch.Tensor):
        move_idx = torch.argmax(move).item()
    else:
        move_idx = move
    move_array[move_idx] = '1'
    return move_array

def get_valid_moves(board, pos):
    valid = [0, 0, 0, 0]
    x, y = pos
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    for i, (dx, dy) in enumerate(directions):
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < 31 and 0 <= new_y < 28 and board[new_x][new_y] != '#':
            valid[i] = 1

    return valid
