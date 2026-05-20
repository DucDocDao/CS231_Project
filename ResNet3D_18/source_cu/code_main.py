import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.models.video as video_models
from collections import deque # Thêm thư viện này để làm hộp chứa buffer

MODEL_PATH = 'best_model_3d.pth'
VIDEO_PATH = '7740721345176.mp4'
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
    
    #frame = cv2.flip(frame, 1)
    #Xu ly frame: resize ve 112x112 va chuyen sang RGB
    #Nhet vao buffer
    frame_resized = cv2.resize(frame, (112, 112))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_buffer.append(frame_rgb)
    frame_counter += 1

    #model du doan khi da co du 16 frame, va de tranh lag thi chi doan moi 4 frame 1 lan
    #de tranh lag thi chi doan moi 4 frame 1 lan
    if len(frame_buffer) == 16 and frame_counter % 4 == 0:
        
        #List 16 frame thanh 1 tensor (1, C, T, H, W) de dua vao model
        frames_np = np.array(frame_buffer)
        frames_np = frames_np.transpose(3, 0, 1, 2) #(C, T, H, W)
        tensor_video = torch.FloatTensor(frames_np).unsqueeze(0) / 255.0 
        tensor_video = tensor_video.to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor_video)
            probabilities = torch.nn.functional.softmax(outputs, dim=1) 
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            #Update label va confidence
            current_label = CLASSES[predicted_idx.item()]
            current_confidence = confidence.item() * 100

    #Ghi text len video
    text = f"{current_label} ({current_confidence:.1f}%)"
    
    #Dung = xanh, Sai = do, Waiting = vang
    if "Dung" in current_label:
        color = (0, 255, 0)
    elif "Sai" in current_label:
        color = (0, 0, 255)
    else:
        color = (0, 255, 255) 

    #In khung den de hien thi text ro hon
    cv2.rectangle(frame, (10, 10), (500, 70), (0,0,0), -1)
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    cv2.imshow("Danh gia tu the benchpress", frame)
    
    #Dieu chinh toc do video va thoat khi nhan 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()