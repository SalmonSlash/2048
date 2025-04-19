import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import os
import datetime

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

        empty_cells = []
        for i in range(4):
            for j in range(4):
                if self.board[i, j] == 0:
                    empty_cells.append((i, j))

        if len(empty_cells) > 0:

            chosen_cell = random.choice(empty_cells)

            if random.random() < 0.1:
                self.board[chosen_cell[0], chosen_cell[1]] = 4
            else:
                self.board[chosen_cell[0], chosen_cell[1]] = 2

    def _compress(self, row):
        new_row = [tile for tile in row if tile != 0]
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1]:
                new_row[i] *= 2
                self.score += new_row[i]
                new_row.pop(i + 1)
                new_row.append(0)

        return new_row + [0] * (self.size - len(new_row))

    def _move(self, direction):
        original_grid = self.board.copy()

        if direction == 0:
            self.board = np.array([self._compress(row) for row in self.board])
        elif direction == 1:
            self.board = np.array([self._compress(row[::-1])[::-1] for row in self.board])
        elif direction == 2:
            self.board = np.array([self._compress(col) for col in self.board.T]).T
        elif direction == 3:
            self.board = np.array([self._compress(col[::-1])[::-1] for col in self.board.T]).T

        return not np.array_equal(original_grid, self.board)

    def _get_state(self):
        with np.errstate(divide='ignore'):
            state = np.where(self.board == 0, 0,np.log2(self.board)).astype(int)
            return state.flatten()

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
            nn.Linear(64 * input_shape[1] * input_shape[2], 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.input_shape = (1, 4, 4)
        self.num_actions = 4
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 128

        self.model = DQN(self.input_shape, self.num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.reward_threshold = 2000
        self.epoch = 0

        self.mean_score50 = 0

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.memory = deque(checkpoint.get('memory', []), maxlen=10000)
        self.epoch = checkpoint.get('epoch', self.epoch)

        self.epsilon =  0.50 # Reset epsilon to 1.0 for fresh training

        print(f"Loaded checkpoint from epoch {self.epoch}, epsilon: {self.epsilon:.2f}")
        # print(f"Memory size: {len(self.memory)}")

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(4)

        state = np.where(state == 0, 0, np.log2(np.maximum(state, 1e-8))).astype(int)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)

        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """เก็บประสบการณ์ใน replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    def replay(self):
        """ฝึกโมเดลจากประสบการณ์"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([np.where(t[0] == 0, 0, np.log2(np.maximum(t[0], 1e-8))) for t in minibatch])).unsqueeze(1)
        actions = torch.LongTensor([t[1] for t in minibatch])
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor(np.array([np.where(t[3] == 0, 0, np.log2(np.maximum(t[3], 1e-8))) for t in minibatch])).unsqueeze(1)
        dones = torch.FloatTensor([t[4] for t in minibatch])


        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        if self.epsilon > self.epsilon_min and self.mean_score50 > self.reward_threshold:
            self.epsilon *= self.epsilon_decay
            self.reward_threshold += 10

    def train(self, episodes):
        scores = []
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        for e in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                self.replay()
                total_reward += reward
                state = next_state

            scores.append(total_reward)
            self.mean_score50 = np.mean(scores[-20:])
            print(f"Episode: {e+1}, Score: {total_reward}, Epsilon: {self.epsilon:.2f}")

        filename = f"checkpoint_{timestamp}.pth"
        print(f"Saving checkpoint to {filename}...")
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory),  # แปลง deque ให้ save ได้
            'epoch': episodes
            }
        torch.save(checkpoint, filename)
        return scores


env = GameAi2048()
agent = DQNAgent(env)

checkpoint_path = "checkpoint_0408_010817.pth"
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    agent.load_checkpoint(checkpoint_path)
    scores = agent.train(episodes=3000)
    print("Training complete! : ", max(scores))
    print("Mean score (last 50 episodes): ", np.mean(scores[-50:]))
else:
    print("No checkpoint found. Starting fresh training...")
