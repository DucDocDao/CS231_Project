import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# ====== CẤU HÌNH ======
window_size = 30
mp_pose = mp.solutions.pose
label_map = {"Len_Sai": 0, "Xuong_Sai": 1, "Len_Dung": 2, "Xuong_Dung": 3}

def calculate_angle2D(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_features_from_folder(folder_path):
    data_list = []    
    # Biến để theo dõi min/max
    global_min = float('inf')
    global_max = float('-inf')
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        for label_name, label_id in label_map.items():
            label_folder = os.path.join(folder_path, label_name)
            if not os.path.exists(label_folder): continue
            for video_name in os.listdir(label_folder):
                video_path = os.path.join(label_folder, video_name)
                cap = cv2.VideoCapture(video_path)
                temp_l_angles, temp_r_angles = [], []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image) #detect pose
                    if results.pose_landmarks:
                        lm = results.pose_landmarks.landmark
                        get_2d = lambda id: [lm[id].x, lm[id].y] #lấy tọa độ các điểm
                        l_angle = calculate_angle2D(get_2d(23), get_2d(11), get_2d(13))
                        r_angle = calculate_angle2D(get_2d(24), get_2d(12), get_2d(14))
                        temp_l_angles.append(l_angle)
                        temp_r_angles.append(r_angle)
                cap.release()

                if len(temp_l_angles) >= window_size:
                    indices = np.linspace(0, len(temp_l_angles) - 1, window_size).astype(int)
                    final_l = [temp_l_angles[i] for i in indices]
                    final_r = [temp_r_angles[i] for i in indices]
                    #hiệu số (vận tốc)
                    diff_l = [final_l[i] - final_l[i-1] for i in range(1, len(final_l))]
                    diff_r = [final_r[i] - final_r[i-1] for i in range(1, len(final_r))]

                    local_min = min(min(final_l), min(final_r))
                    local_max = max(max(final_l), max(final_r))

                    global_min = min(global_min, local_min)
                    global_max = max(global_max, local_max)

                    data_list.append({
                        'l': final_l,
                        'r': final_r,
                        'dl': diff_l,
                        'dr': diff_r,
                        'label': label_id
                    })
                    print(f"Xu ly: {video_name}")

    return data_list, global_min, global_max
    # ====== BÁO CÁO & GHI FILE ======
def save_to_csv(data, filename, min_val, max_val):
    header = ([f"L_{i}" for i in range(window_size)] +
              [f"R_{i}" for i in range(window_size)] +
              [f"dL_{i}" for i in range(window_size-1)] +
              [f"dR_{i}" for i in range(window_size-1)] +
              ["label"])
    denom = (max_val - min_val) if max_val != min_val else 1.0
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data:
            norm_l = [np.clip((a - min_val) / denom, 0, 1) for a in row['l']]
            norm_r = [np.clip((a - min_val) / denom, 0, 1) for a in row['r']]
            norm_dl = [d / 180.0 for d in row['dl']]
            norm_dr = [d / 180.0 for d in row['dr']]
            writer.writerow(norm_l + norm_r + norm_dl + norm_dr + [row['label']])
    print(f"--- Da luu file: {filename} ---")

def run_pipeline():
    print(">>> DANG XU LY TAP TRAIN...")
    train_data, train_min, train_max = extract_features_from_folder("Dataset/train")
    save_to_csv(train_data, "bench_press_train_4.csv", train_min, train_max)
    # Lưu lại tham số chuẩn hóa vào file txt/csv để kiểm tra nếu cần
    with open("scaling_params.txt", "w") as f:
        f.write(f"train_min,{train_min}\ntrain_max,{train_max}")
    # 2. Xử lý tập TEST sử dụng min/max của TRAIN
    print("\n>>> DANG XU LY TAP TEST...")
    test_data, _, _ = extract_features_from_folder("Dataset/test")
    # Quan trọng: Dùng train_min và train_max ở đây
    save_to_csv(test_data, "bench_press_test_4.csv", train_min, train_max)
    print(f"\nCOMPLETED!")
    print(f"Gia tri chuan hoa su dung: Min={train_min:.2f}, Max={train_max:.2f}")

if __name__ == "__main__":
    run_pipeline()
