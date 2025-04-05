import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

class ACTIONS:
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

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

    def _move(self, direction):
        original_grid = self.board.copy()

        if direction == 0:
            self.board = np.array([self._compress(row) for row in self.board]) #ใช้ _compress แบบปกติ
        elif direction == 1:
            self.board = np.array([self._compress(row[::-1])[::-1] for row in self.board]) #ใช้ _compress แบบกลับด้าน
        elif direction == 2:
            self.board = np.array([self._compress(col) for col in self.board.T]).T #self.board.T คือการเปลี่ยนแถวเป็นคอลัมน์ ต้องเป็น 2D Array เมธอด .T ใช้ได้กับ NumPy Array โดยเฉพาะ แต่ละแถวต้องมีความยาวเท่ากัน
        elif direction == 3:
            # พลิกคอลัมน์, _compress แล้วพลิกกลับและ transpose กลับ
            self.board = np.array([self._compress(col[::-1])[::-1] for col in self.board.T]).T

        return not np.array_equal(original_grid, self.board)

    def _get_state(self):
            state = np.where(self.board == 0, 0,np.log2(self.board)).astype(int) #เช็คว่าเป็น 0 หรือไม่ ถ้าใช่ให้เป็น 0 ถ้าไม่ใช่ให้ log2 ของตัวเลขในบอร์ด และทำเป้น int
            return state.flatten() #แปลงเป็น 1D Array

    def _get_state2(self):
        state = np.where(self.board == 0, 0,np.log2(self.board)).astype(int) #เช็คว่าเป็น 0 หรือไม่ ถ้าใช่ให้เป็น 0 ถ้าไม่ใช่ให้ log2 ของตัวเลขในบอร์ด และทำเป้น int
        return state

    def step(self, action):
        moved = self._move(action)
        if moved:
            self.add_random_tile()
        else:
            self.done = True
        return self.board, self.score, self.done

    def _is_game_over(self):
        return not any(
            np.any(self.board[:-1,:] == self.board[1:,:]) or
            np.any(self.board[:,:-1] == self.board[:,1:]))

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64*input_shape[1]*input_shape[2], 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

env = GameAi2048()
initial_state = env.reset()

print("Initial State:")
print(initial_state)
print(env._compress(initial_state[0]))
print(env._move(3))
print(env.board)
print(env._get_state2())
print(env._get_state())
