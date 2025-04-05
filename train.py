import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

class ACTIONS:
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    def __hash__(self):
        return hash((self.x, self.y))

class GameAi2048:
    def __init__(self):
        self.size = 4
        self.reset()

    def reset(self):
        self.score = 0
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.add_random_tile()
        self.add_random_tile()
        self.add_random_tile()
        self.add_random_tile()
        self.add_random_tile()
        self.add_random_tile()
        self.done = False
        return self.board

    def add_random_tile(self):
    # เช็คหาตำแหน่งที่ว่างในบอร์ด และเก็บในเซ็ต
        empty_cells = []
        for i in range(4):
            for j in range(4):
                if self.board[i, j] == 0:
                    empty_cells.append((i, j))

        if len(empty_cells) > 0:
            # เลือกตำแหน่งแบบสุ่มจากตำแหน่งที่ว่าง
            chosen_cell = random.choice(empty_cells)
            # สุ่มค่าใหม่ โดยมีโอกาส 10% เป็น 4 และ 90% เป็น 2
            if random.random() < 0.1:
                self.board[chosen_cell[0], chosen_cell[1]] = 4
            else:
                self.board[chosen_cell[0], chosen_cell[1]] = 2

    def _compress(self, row):
        new_row = [tile for tile in row if tile != 0]#ลบ 0 ออกจากแถว
        for i in range(len(new_row) - 1): #ตรวจสอบแถว โดยมีแค่ตัวเลขที่ไม่ใช่0 ลบ1เพราะ (4,2,2,0) เราจะตรวจแค่3ตัวเพราะถ้าตรวจ4ตัว ที่สี4จะไม่มีตต่อไปเทียบ
            if new_row[i] == new_row[i + 1]: # ตรวจที่ตัวนั้นและตัวถัดไป ถ้าตัวนั้นเท่ากัน
                new_row[i] *= 2               # รวมแผ่นที่ i โดยคูณ 2
                self.score += new_row[i]      # อัปเดตคะแนนด้วยค่าที่รวมได้
                new_row.pop(i + 1)            # ลบแผ่นที่ i+1 ออกจากลิสต์
                new_row.append(0)             # เติม 0 ลงท้ายลิสต์เพื่อรักษาความยาว
        # เติมค่า 0 ให้ครบขนาดแถวที่กำหนด (self.size)
        return new_row + [0] * (self.size - len(new_row))

env = GameAi2048()
initial_state = env.reset()

print("Initial State:")
print(initial_state)
print(env._compress(initial_state[0]))


class LinearQNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, hiddenSize)
        self.linear3 = nn.Linear(hiddenSize, outputSize)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x