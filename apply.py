from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import datetime
import re

class Game2048Environment:
    def __init__(self):
        self.driver = None
        self.setup_driver()
        self.current_score = 0
        self.prev_score = 0

    def setup_driver(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        self.driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
        )
        self.driver.get("https://www.2048.org/")
        WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "game-container")))
        print("เจอแล้ว!")
# # 2. ฟังก์ชันดึงสถานะตารางจาก DOM (scrape grid)
# # ------------------------------------------------------------------------------
    def scrape_grid(self):
        grid = self.driver.execute_script("""
            let cells = document.querySelectorAll('.tile');
            let grid = Array(4).fill(0).map(() => Array(4).fill(0));
            cells.forEach(cell => {
                const classes = Array.from(cell.classList);
                const posClass = classes.find(c => c.startsWith('tile-position'));
                const [_, x, y] = posClass.split('-').map(Number);
                const value = parseInt(cell.querySelector('.tile-inner').textContent) || 0;
                grid[y-1][x-1] = Math.max(grid[y-1][x-1], value);
            });
            return grid;
        """)
        return np.array(grid, dtype=np.float32)

    def get_score(self):
        """ดึงคะแนนปัจจุบันจากเกม (ใช้ Regular Expression)"""
        try:
            elem = WebDriverWait(self.driver, 2).until(
                EC.presence_of_element_located((By.CLASS_NAME, "score-container"))
            )
            # ค้นหาตัวเลขทั้งหมดในข้อความ
            numbers = re.findall(r'\d+', elem.text)
            return int(numbers[0]) if numbers else 0
        except (TimeoutException, NoSuchElementException, ValueError):
            return 0

    def is_game_over(self):
        try:
            self.driver.find_element(By.CLASS_NAME, "game-over")
            return True
        except NoSuchElementException:
            return False

    def reset(self):
        """เริ่มเกมใหม่"""
        self.current_score = 0
        self.prev_score = 0
        time.sleep(0.5)  # รอให้เกมรีเซ็ต
        return self.scrape_grid()

    def step(self, action):
        actions = [Keys.ARROW_UP, Keys.ARROW_RIGHT, Keys.ARROW_DOWN, Keys.ARROW_LEFT]
        self.driver.find_element(By.TAG_NAME, "body").send_keys(actions[action])

        # รอการเคลื่อนไหวเสร็จสิ้น
        time.sleep(0.1)

        # ดึงสถานะใหม่และคำนวณรางวัล
        new_grid = self.scrape_grid()
        self.current_score = self.get_score()
        reward = self.current_score - self.prev_score
        self.prev_score = self.current_score
        done = self.is_game_over()

        return new_grid, reward, done

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

if __name__ == "__main__":
    env = Game2048Environment()
    agent = DQNAgent(env)
    checkpoint_path = "checkpoint_0408_ok.pth"
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        agent.load_checkpoint(checkpoint_path)
        agent.epsilon = 0.01  # ปิดการสุ่มเพื่อเล่นด้วย AI ล้วน

    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        state, reward, done = env.step(action)
        total_reward += reward
        print(f"Score: {total_reward}", end="\r")  # แสดงคะแนนแบบ real-time

    env.driver.quit()



    # grid = env.scrape_grid()
    # print("Current 4×4 grid state:")
    # for row in grid:
    #     print(row)

    # body = env.driver.find_element(By.TAG_NAME, "body")
    # body.send_keys(Keys.ARROW_UP)

    # time.sleep(1)

    # grid = env.scrape_grid()
    # print("Current 4×4 grid state:")
    # for row in grid:
    #     print(row)
    # time.sleep(5)  # รอ 5 วินาทีเพื่อดูผลลัพธ์
    # env.driver.quit()