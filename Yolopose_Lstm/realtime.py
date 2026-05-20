import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from collections import deque
import time

# ─── 1. ĐỊNH NGHĨA MODEL LSTM ─────────────────────────────
class BenchPressLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, num_layers=2, num_classes=4, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_frame = lstm_out[:, -1, :]
        return self.classifier(last_frame)

# ─── 2. CẤU HÌNH & LOAD MODELS ───────────────────────────
SEQUENCE_LENGTH = 30
# NGƯỠNG CHIỀU DÀI: Tỉ lệ chiều dài lưng / chiều cao khung hình.
LENGTH_THRESHOLD = 0.15 
LABELS = ["Len_Dung", "Xuong_Dung", "Len_Sai", "Xuong_Sai"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("yolo26n-pose.pt") 
lstm_model = BenchPressLSTM(input_size=4).to(device)
lstm_model.load_state_dict(torch.load("models/lstm_model_4class.pth", map_location=device, weights_only=True))
lstm_model.eval()

# ─── 3. HÀM LOGIC CHIỀU DÀI LƯNG ───────
def check_is_lying_by_length(kps, frame_height):
    
    if kps[11][2] < 0.5 or kps[12][2] < 0.5:
        return False, 0.0
    # Lấy trung điểm 2 vai và 2 hông
    mid_shoulder = (kps[5][:2] + kps[6][:2]) / 2
    mid_hip = (kps[11][:2] + kps[12][:2]) / 2
    
    # Tính chiều dài lưng (khoảng cách Euclidean)
    torso_length = np.linalg.norm(mid_shoulder - mid_hip)
    
    # Tính tỉ lệ so với chiều cao khung hình
    ratio = torso_length / frame_height
    
    # Nếu chiều dài lưng chiếm hơn LENGTH_THRESHOLD (15%) khung hình thì là đang nằm
    if ratio > LENGTH_THRESHOLD:
        return True, ratio
    return False, ratio

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def get_4_features(kps, prev_angles):
    l_shoulder = calculate_angle(kps[7][:2], kps[5][:2], kps[11][:2])
    r_shoulder = calculate_angle(kps[8][:2], kps[6][:2], kps[12][:2])
    current_angles = [l_shoulder, r_shoulder]
    velocities = [0.0, 0.0] if prev_angles is None else [cur - prev for cur, prev in zip(current_angles, prev_angles)]
    return current_angles + velocities, current_angles

# ─── 4. GIAO DIỆN UI ─────────────────────────────────────
def draw_ui(frame, label_idx, confidence, n_frames, fps, is_ready, torso_ratio):
    h, w = frame.shape[:2]
    
    cv2.rectangle(frame, (0, 0), (w, 75), (0, 0, 0), -1) 
    
    if not is_ready:
        status_color = (0, 165, 255)
        # Khi ĐANG CHỜ (đứng): Vẫn hiện Ratio để dễ căn chỉnh camera
        cv2.putText(frame, f"WAITING: Please lie down (Torso Ratio: {torso_ratio:.2f})", (10, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    else:
        label_text = LABELS[label_idx]
        color = (0, 255, 0) if label_idx < 2 else (0, 0, 255)
        
        # Khi ĐANG TẬP (nằm): hiện Status và Conf
        cv2.putText(frame, f"STATUS: {label_text}", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        progress = int((n_frames / SEQUENCE_LENGTH) * w)
        cv2.rectangle(frame, (0, 70), (progress, 75), color, -1)

    cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    return frame

# ─── 5. VÒNG LẶP CHÍNH ───────────────────────────────────
cap = cv2.VideoCapture(r"D:\Option\CS\CS321_Nhap_mon_thi_giac_may_tinh\testnguoc.mov")

sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
prev_angles = None
current_label_idx, current_conf = 0, 0.0
is_ready = False
torso_ratio = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
        
    # Xoay frame nếu cần
    # frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Lấy chiều cao của video (để làm mẫu số tính tỉ lệ)
    h, w = frame.shape[:2]
    start_time = time.time()

    results = yolo_model.track(frame, persist=True, verbose=False, device=device)

    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        main_idx = np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
        kps = results[0].keypoints.data[main_idx].cpu().numpy()

        if kps[5:13, 2].mean() > 0.5:
            # 1. KIỂM TRA TRẠNG THÁI BẰNG TỈ LỆ CHIỀU DÀI LƯNG (Truyền h vào)
            is_lying, torso_ratio = check_is_lying_by_length(kps, h)

            if is_lying:
                if not is_ready:
                    is_ready = True
                    sequence_buffer.clear()
            else:
                if is_ready:
                    is_ready = False
                    sequence_buffer.clear()

            # 2. XỬ LÝ KHI ĐÃ SẴN SÀNG
            if is_ready:
                features, current_angles = get_4_features(kps, prev_angles)
                prev_angles = current_angles
                
                norm_feat = [features[0]/180.0, features[1]/180.0, features[2]/10.0, features[3]/10.0]
                sequence_buffer.append(norm_feat)

                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    input_seq = torch.FloatTensor(np.array(sequence_buffer)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = lstm_model(input_seq)
                        prob = torch.softmax(output, dim=1)
                        conf, predicted = torch.max(prob, 1)
                        current_label_idx, current_conf = predicted.item(), conf.item()

    # 3. UI VÀ HIỂN THỊ
    fps = 1.0 / (time.time() - start_time + 1e-6)
    frame = results[0].plot(labels=False)
    
    frame = draw_ui(frame, current_label_idx, current_conf, len(sequence_buffer), fps, is_ready, torso_ratio)

    display_scale = 0.5  
    disp_w = int(frame.shape[1] * display_scale)
    disp_h = int(frame.shape[0] * display_scale)
    display_frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    
    cv2.imshow("Bench Press AI Assistant", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()