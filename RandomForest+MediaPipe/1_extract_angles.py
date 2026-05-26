import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np


# 1. HÀM TOÁN HỌC: TÍNH GÓC GIỮA 3 ĐIỂM
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


# 2. CẤU HÌNH THƯ MỤC & NHÃN

# Đã trỏ đúng đường dẫn thực tế trên máy
DATASET_DIR = "Dataset/Dataset" 
SPLITS = ["train", "test"] # 2 tập dữ liệu
LABEL_MAP = {
    "Len_Dung": 0,
    "Xuong_Dung": 1,
    "Len_Sai": 2,
    "Xuong_Sai": 3
}

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose # type: ignore
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

data_list = []


# 3. QUÉT VIDEO VÀ TRÍCH XUẤT
print("🚀 BẮT ĐẦU TRÍCH XUẤT DỮ LIỆU...")

for split in SPLITS:
    for folder_name, label_id in LABEL_MAP.items():
        folder_path = os.path.join(DATASET_DIR, split, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"⚠️ Bỏ qua: Không tìm thấy '{folder_path}'")
            continue
            
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.mov', '.avi', '.webm'))]
        print(f"\n📂 Đang xử lý: Tập {split.upper()} - {folder_name} ({len(video_files)} videos)")
        
        for video_name in video_files:
            video_path = os.path.join(folder_path, video_name)
            cap = cv2.VideoCapture(video_path)
            
            # Khởi tạo biến đếm số frame của từng video
            frame_count = 0 
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: 
                    break
                    
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    # Tính góc nách
                    angle_l = calculate_angle([lm[23].x, lm[23].y], [lm[11].x, lm[11].y], [lm[13].x, lm[13].y])
                    angle_r = calculate_angle([lm[24].x, lm[24].y], [lm[12].x, lm[12].y], [lm[14].x, lm[14].y])
                    
                    data_list.append({
                        "left_armpit": angle_l,
                        "right_armpit": angle_r,
                        "label": label_id,
                        "split": split # Đánh dấu dòng này thuộc tập train hay test
                    })
                    frame_count += 1 # Tăng biến đếm lên 1 khi trích xuất thành công
            cap.release()
            
            # --- DÒNG THÔNG BÁO TIẾN ĐỘ ---
            print(f"   + Xong: {video_name} (Thu được {frame_count} frames)")

# 4. LƯU RA FILE CSV
df = pd.DataFrame(data_list)
output_file = "data_angles_4_classes.csv"
df.to_csv(output_file, index=False)

print("\n" + "="*50)
print(f"✅ HOÀN TẤT! Đã lưu tổng cộng {len(df)} dòng vào file '{output_file}'.")
print("="*50)