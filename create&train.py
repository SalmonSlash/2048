import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
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
        with np.errstate(divide='ignore'):#ใช้ np.errstate เพื่อจัดการกับการหารด้วย 0 โดยไม่ให้เกิดข้อผิดพลาด
            state = np.where(self.board == 0, 0,np.log2(self.board)).astype(int) #เช็คว่าเป็น 0 หรือไม่ ถ้าใช่ให้เป็น 0 ถ้าไม่ใช่ให้ log2 ของตัวเลขในบอร์ด และทำเป้น int
            return state.flatten() #แปลงเป็น 1D Array

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
    def __init__(self, input_shape, num_actions): #ที่เขาสอนจะเป็น input_shape=(1,4,4) (ช่องสัญญาณ,กว้าง,สูง) แต่ที่เราทำจะเป็น input_shape=4
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), #1คือจำนวนช่องสัญญาณ (channels) ของภาพที่ป้อนเข้าไปในโมเดล ถ้าภาพสีมี 3 ช่องสัญญาณ (R, G, B) ถ้าเป็นภาพขาวดำจะมี 1 ช่องสัญญาณ
                                                        #32 Filter: หมายถึงสร้าง 32 ลวดลายที่ต่างกัน เพื่อตรวจจับสถานะเกม
                                                        # ตัวอย่างการทำงาน:
                                                        # Filter 1: ตรวจจับตัวเลขที่อยู่ติดกันในแนวตั้ง
                                                        # Filter 2: ตรวจจับตัวเลขที่อยู่ติดกันในแนวนอน
                                                        # Filter 3: ตรวจจับช่องว่างในบอร์ด
                                                        #kernel_size=3: ขนาดของ Filter คือ 3x3
                                                        #padding=1: เพิ่ม padding รอบๆ ขอบของภาพ เพื่อไม่ให้ขนาดของภาพลดลงหลังจากการ convolutions
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),#32->64 จำนวน Filter มากขึ้น → เรียนรู้ Features ได้ละเอียดขึ้น
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

# ตัวอย่างการใช้งานโมเดล DQN -----------------------------------------------------------------------------------------------------------------------
# input_grid = [
#     [0, 0, 0, 0],
#     [0, 0, 0, 0],
#     [0, 0, 2, 1],
#     [1, 0, 2, 0]
# ]
# # แปลงเป็น Tensor
# input_tensor = torch.FloatTensor(input_grid).unsqueeze(0).unsqueeze(0)  # Shape: (1,1,4,4)

# # ส่งเข้าโมเดล
# model = DQN(4,4)
# q_values = model(input_tensor)
# print("Q-values:", q_values)
# print("Output shape:", q_values.shape)
#-----------------------------------------------------------------------------------------------------------------------

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
        self.memory = deque(maxlen=10000) #ใช้ deque เพื่อเก็บประสบการณ์การเล่นเกม
        self.reward_threshold = 200 #ค่าคะแนนที่ต้องการให้ agent ได้

    def act(self, state):
        # """เลือกการกระทำโดยใช้ epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            # print("Random Action")
            return np.random.randint(4)
        # if self.epsilon <= 0.02:
        #     print(state)
        state = np.where(state == 0, 0, np.log2(state)).astype(int)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        # print("State Tensor: ", state_tensor.shape)
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
        states = torch.FloatTensor(np.array([np.where(t[0] == 0, 0, np.log2(t[0])) for t in minibatch])).unsqueeze(1)
        actions = torch.LongTensor([t[1] for t in minibatch])
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor(np.array([np.where(t[3] == 0, 0, np.log2(t[3])) for t in minibatch])).unsqueeze(1)
        dones = torch.FloatTensor([t[4] for t in minibatch])

        # print("States: ", states.shape)
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # *** การฝึกแบบขั้นบรรได (ดูจาก epsilon )สามารถบอกได้ว่าช่องไหนของบอร์ดที่ agent ใช้ในการสำรวจและเล่นจริง ทำให้รู้ว่าจำนวน episode พอหรือไม่ เจอเกมที่สำคัญรึยัง
        if self.epsilon > self.epsilon_min and self.mean_score50 > self.reward_threshold:
            self.epsilon *= self.epsilon_decay
            self.reward_threshold += 10 # เพิ่มค่าคะแนนที่ต้องการให้ agent ได้



    def train(self, episodes):
        """ฝึกโมเดลจำนวน episode ที่กำหนด"""
        scores = []
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

        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        filename = f"checkpoint_{timestamp}.pth"

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory),  # แปลง deque ให้ save ได้
            'epoch': episodes
            }
        torch.save(checkpoint, filename)

        return scores

# if __name__ == "__main__":
env = GameAi2048()
agent = DQNAgent(env)
scores = agent.train(episodes=100)
print("Training complete! : ", max(scores))
print("mean score50: ", np.mean(scores[-50:]))


# บันทึกโมเดล
# torch.save(agent.model.state_dict(), "2048_dqn.pth")

# print("Initial State:")
# # print(initial_state)
# # print(env._compress(initial_state[0]))
# print(env._move(3))
# print(env.board)
# print(env._get_state2())
# print(env._get_state())

