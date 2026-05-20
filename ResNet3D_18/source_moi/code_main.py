import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.models.video as video_models
from collections import deque

MODEL_PATH = 'best_model_3d.pth'
VIDEO_PATH = '101.mp4'
CLASSES = ['Len_Dung', 'Xuong_Dung', 'Len_Sai', 'Xuong_Sai']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"chay bang {DEVICE}")

# Khoi tao model ResNet3D_18
def get_3d_model(num_classes=4):
    model = video_models.r3d_18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

print("Dang load model")
model = get_3d_model(num_classes=len(CLASSES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Loading complete")

#Doc video va du doan theo frame
print(f"Processing: {VIDEO_PATH}...")

cap = cv2.VideoCapture(VIDEO_PATH)

#Tao buffer chua 16 frame, khi frame thu 17 den se day frame so 1 ra ngoai 
frame_buffer = deque(maxlen=16)

#default label va confidence khi chua co du lieu de doan
current_label = "Waiting for data..."
current_confidence = 0.0
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #resize khung hinh ve 171x128
    frame_resized = cv2.resize(frame, (171, 128))
    
    #center crop khung hinh ve 112x112
    h, w, _ = frame_resized.shape
    start_y = h // 2 - 56
    start_x = w // 2 - 56
    frame_cropped = frame_resized[start_y:start_y+112, start_x:start_x+112]

    #chuyen sang rgb va them vao buffer
    frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
    frame_buffer.append(frame_rgb)
    frame_counter += 1

    # model du doan khi da co du 16 frame, va chi doan moi 4 frame 1 lan
    if len(frame_buffer) == 16 and frame_counter % 4 == 0:
        
        # List 16 frame thanh tensor (B, C, T, H, W) -> (1, 3, 16, 112, 112)
        frames_np = np.array(frame_buffer)
        frames_np = frames_np.transpose(3, 0, 1, 2) #(C, T, H, W)
        tensor_video = torch.FloatTensor(frames_np).unsqueeze(0) / 255.0 
        tensor_video = tensor_video.to(DEVICE)

        #normalize theo mean va std cua kinetics-400
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1).to(DEVICE)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1).to(DEVICE)
        tensor_video = (tensor_video - mean) / std

        with torch.no_grad():
            outputs = model(tensor_video)
            probabilities = torch.nn.functional.softmax(outputs, dim=1) 
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Update label va confidence
            current_label = CLASSES[predicted_idx.item()]
            current_confidence = confidence.item() * 100

    # Ghi text len video goc
    text = f"{current_label} ({current_confidence:.1f}%)"
    
    # Dung = xanh, Sai = do, Waiting = vang
    if "Dung" in current_label:
        color = (0, 255, 0)
    elif "Sai" in current_label:
        color = (0, 0, 255)
    else:
        color = (0, 255, 255) 

    # In khung den de hien thi text ro hon
    cv2.rectangle(frame, (10, 10), (500, 70), (0,0,0), -1)
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    cv2.imshow("Danh gia tu the benchpress", frame)
    
    # Dieu chinh toc do video va thoat khi nhan 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()