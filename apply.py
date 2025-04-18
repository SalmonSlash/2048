import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import mss
import cv2
import torch
import pyautogui
from PIL import Image
import time
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def select_game_region():
    with mss.mss() as sct:
        # เลือกจอหลัก (monitor 1)
        primary_mon = sct.monitors[1]

        # จับภาพจอหลัก
        screen = sct.grab(primary_mon)
        img = cv2.cvtColor(np.array(screen), cv2.COLOR_BGRA2BGR)

        # ปรับขนาดภาพเพื่อแสดงผล
        scale_percent = 60  # ปรับค่านี้ตามต้องการ
        width = int(img.shape[1] * scale_percent/100)
        height = int(img.shape[0] * scale_percent/100)
        resized = cv2.resize(img, (width, height))

        # ให้ผู้ใช้เลือกพื้นที่
        roi = cv2.selectROI("เลือกพื้นที่เกมด้วยการลากเมาส์ (กด Space หรือ Enter เพื่อยืนยัน)", resized)
        cv2.destroyAllWindows()

        # คำนวณพิกัดจริงโดยคำนึงถึงการปรับขนาด
        scale_x = img.shape[1] / width
        scale_y = img.shape[0] / height

        x = int(roi[0] * scale_x) + primary_mon['left']
        y = int(roi[1] * scale_y) + primary_mon['top']
        w = int(roi[2] * scale_x)
        h = int(roi[3] * scale_y)
        return (x, y, w, h)

def capture_screen(bbox=None):
    with mss.mss() as sct:
        monitor = sct.monitors[2] if bbox is None else {
            "top": bbox[1],
            "left": bbox[0],
            "width": bbox[2],
            "height": bbox[3]
        }
        sct_img = sct.grab(monitor)
        return cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

possible_values = [2 ** i for i in range(1, 18)]  # 2 ถึง 131072

def correct_ocr_value(ocr_value, possible_values):
    """
    ปรับแก้ค่าที่ได้จาก OCR ให้ใกล้เคียงกับค่าที่เป็นไปได้ในเกม 2048
    """
    if ocr_value in possible_values:
        return ocr_value
    else:
        # คำนวณค่าที่ใกล้เคียงที่สุด
        closest_value = min(possible_values, key=lambda x: abs(x - ocr_value))
        # หากความแตกต่างน้อยกว่า 20% ให้ปรับเป็นค่าที่ใกล้เคียงที่สุด
        if abs(closest_value - ocr_value) / closest_value < 0.2:
            return closest_value
        else:
            return 0  # หากความแตกต่างมากเกินไป ให้ถือว่าเป็นช่องว่าง

def ocr_tile(tile_gray):
    # 1. ปรับขนาดภาพเพื่อเพิ่มความละเอียด
    tile = cv2.resize(tile_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 2. ปรับคอนทราสต์ด้วย CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    tile = clahe.apply(tile)

    # 3. ใช้ Threshold แบบ Otsu เพื่อแยกข้อความ
    _, tile = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. ตรวจสอบและปรับความเป็นขาว-ดำให้เหมาะสม
    white = cv2.countNonZero(tile)
    black = tile.size - white
    if white < black:
        tile = cv2.bitwise_not(tile)

    # 5. ใช้ Morphological Closing เพื่อเชื่อมต่อส่วนของตัวเลข
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    tile = cv2.morphologyEx(tile, cv2.MORPH_CLOSE, kernel)
    # 6. ใช้ Tesseract OCR เพื่ออ่านตัวเลข
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    txt = pytesseract.image_to_string(tile, config=config).strip()

    # 7. แปลงข้อความเป็นตัวเลขและปรับแก้
    try:
        ocr_value = int(txt)
        print(f"OCR raw result: '{txt}' → interpreted as: {ocr_value}")
    except ValueError:
        return 0  # หากไม่สามารถแปลงเป็นตัวเลขได้ ให้ถือว่าเป็นช่องว่าง

    return correct_ocr_value(ocr_value, possible_values)


def process_grid(img):
    # img: ภาพ BGR ของพื้นที่เกมที่จับมาแล้ว
    h, w = img.shape[:2]
    tile_h = h // 4
    tile_w = w // 4
    grid = []

    for r in range(4):
        row_vals = []
        for c in range(4):
            # ตัดขอบของแต่ละช่องเพื่อหลีกเลี่ยงกรอบ
            m_h, m_w = int(0.1 * tile_h), int(0.1 * tile_w)
            y1 = r * tile_h + m_h
            y2 = (r + 1) * tile_h - m_h
            x1 = c * tile_w + m_w
            x2 = (c + 1) * tile_w - m_w

            # แปลงภาพเป็น Grayscale
            tile_gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

            # อ่านค่าตัวเลขจากแต่ละช่อง
            val = ocr_tile(tile_gray)

            row_vals.append(val)
        grid.append(row_vals)

    return grid


def main_loop():
    # เลือกพื้นที่เกมครั้งเดียว
    bbox  = select_game_region()
    print(f"Game region: {bbox}")

    while True:
        frame = capture_screen(bbox)       # :contentReference[oaicite:0]{index=0}
        grid = process_grid(frame)

        # แสดงผลลัพธ์บนคอนโซลและหน้าต่าง
        for row in grid:
            print(row, end='\n')
        print("-" * 20)

        # รอ key press: กด 'q' เพื่อออกจากลูป
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    # main_loop()
    h = select_game_region()
    Image = capture_screen(h)
    resut = process_grid(Image)
    print(resut[0],end='\n')
    print(resut[1],end='\n')
    print(resut[2],end='\n')
    print(resut[3],end='\n')


# class Real2048Env:
#     def __init__(self):
#         self.current_dir = 1  # ทิศทางเริ่มต้น
#         self.model = torch.load('checkpoint_0408_ok.pth')
#         self.board_bbox = detect_2048_board()

#     def execute_move(self, action):
#         key_map = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
#         pyautogui.press(key_map[action])
#         time.sleep(0.2)  # รอให้เกมประมวลผล

#     def get_reward(self, prev_score, new_score):
#         return new_score - prev_score

#     def run_episode(self):
#         while True:
#             prev_state = image_to_state(self.board_bbox)
#             action = self.model.act(prev_state)
#             self.execute_move(action)

#             new_state = image_to_state(self.board_bbox)
#             done = (new_state == prev_state).all()

#             if done:
#                 break


# class DQN(nn.Module):
#     def __init__(self, input_shape, num_actions):
#         super(DQN, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(64 * input_shape[1] * input_shape[2], 512),
#             nn.ReLU(),
#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, num_actions)
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)


# class DQNAgent:
#     def __init__(self, env):
#         self.env = env
#         self.input_shape = (1, 4, 4)
#         self.num_actions = 4

#         self.model = DQN(self.input_shape, self.num_actions)

#         # ปิดการใช้งาน epsilon สำหรับสุ่ม action
#         self.epsilon = 0.0  # ปิดการสุ่ม action ทั้งหมด

#     def load_checkpoint(self, path):
#         """โหลดเฉพาะส่วนที่จำเป็นสำหรับ inference"""
#         checkpoint = torch.load(path)

#         # โหลดเฉพาะน้ำหนักโมเดล
#         self.model.load_state_dict(checkpoint['model_state_dict'])

#         # ตั้งค่าโมเดลเป็นโหมดประเมินผล
#         self.model.eval()

#         print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

#     def act(self, state):
#         """การทำนาย action จากโมเดล (ไม่มีส่วนสุ่ม)"""
#         # แปลง state ให้อยู่ในรูปแบบที่โมเดลรับได้
#         state_tensor = torch.FloatTensor(
#             np.where(state == 0, 0, np.log2(np.maximum(state, 1e-8)))
#         ).unsqueeze(0).unsqueeze(0)

#         # ทำนายด้วยโมเดล
#         with torch.no_grad():
#             q_values = self.model(state_tensor)

#         return torch.argmax(q_values).item()

# env = Real2048Env()
# agent = DQNAgent(env)

# checkpoint_path = "checkpoint_0408_181159.pth"
# if os.path.exists(checkpoint_path):
#     agent.load_checkpoint(checkpoint_path)
#     print("โหลดโมเดลสำเร็จ พร้อมเล่นเกม!")
# else:
#     print("ไม่พบไฟล์ checkpoint!")
#     exit()

# # ตัวอย่างการเล่นเกมจริง (ปรับใช้ตามระบบจับภาพหน้าจอ)
# state = env.reset()
# done = False

# while not done:
#     action = agent.act(state)
#     next_state, reward, done = env.step(action)
#     state = next_state