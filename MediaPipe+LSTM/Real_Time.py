import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense, Dropout, Input
# ====== CẤU HÌNH (BẮT BUỘC KHỚP VỚI CODE TRAIN) ======
window_size = 30
# Dùng chính xác con số bạn đã fix trong code trích xuất
MIN_ANGLE = 0.12
MAX_ANGLE = 104.41
DENOM = MAX_ANGLE - MIN_ANGLE
# Mapping nhãn để hiển thị tiếng Việt cho dễ hiểu
# Cập nhật đủ 4 nhãn theo đúng thứ tự bạn đã train
LABEL_MAP = {
    0: "LEN SAI",
    1: "XUONG SAI",
    2: "LEN DUNG",
    3: "XUONG DUNG"
}
# 1. Load model đã train của bạn
# Đảm bảo file .h5 nằm cùng thư mục hoặc đúng đường dẫn
model = tf.keras.models.load_model("bench_press_final_model", compile=False)
print("DA LOAD MODEL THANH CONG!")
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle2D(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
# 3. Buffer để lưu trữ 30 frame gần nhất
temp_l_angles = []
temp_r_angles = []
cap = cv2.VideoCapture(0) # Mở webcam
print("Dang khoi dong camera")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    # Chuyển màu sang RGB cho Mediapipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.pose_landmarks:
        # Vẽ các điểm khớp xương để quan sát
        h, w, _ = image.shape
        lm = results.pose_landmarks.landmark
        def get_2d(id): return np.array([lm[id].x, lm[id].y])
        shoulder_2d_l = get_2d(11)
        elbow_2d_l = get_2d(13)
        hip_2d_l = get_2d(23)
        wrist_2d_l = get_2d(15)

        shoulder_2d_r = get_2d(12)
        elbow_2d_r = get_2d(14)
        hip_2d_r = get_2d(24)
        wrist_2d_r = get_2d(16)
        # Tính góc dựa trên chính xác các ID khớp bạn đã dùng: 23, 11, 13 và 24, 12, 14
        l_angle = calculate_angle2D(hip_2d_l, shoulder_2d_l, elbow_2d_l)
        r_angle = calculate_angle2D(hip_2d_r, shoulder_2d_r, elbow_2d_r)
        # ====== HIỂN THỊ GÓC ======
        def draw_angle(angle, point):
            x = int(point[0] * w)
            y = int(point[1] * h)
            cv2.putText(image, str(int(angle)),
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 2)
        draw_angle(l_angle, shoulder_2d_l)
        draw_angle(r_angle, shoulder_2d_r)
        # ====== VẼ ĐIỂM ======
        def draw_point(p, color=(0,255,255)):
            cv2.circle(image, (int(p[0]*w), int(p[1]*h)), 5, color, -1)
        for p in [shoulder_2d_l, elbow_2d_l, wrist_2d_l, hip_2d_l,
                shoulder_2d_r, elbow_2d_r, wrist_2d_r, hip_2d_r]:
            draw_point(p)
        # ====== VẼ LINE ======
        def draw_line(p1, p2):
            cv2.line(image,
                    (int(p1[0]*w), int(p1[1]*h)),
                    (int(p2[0]*w), int(p2[1]*h)),
                    (0,255,0), 2)
        # tay trái
        draw_line(shoulder_2d_l, elbow_2d_l)
        draw_line(elbow_2d_l, wrist_2d_l)
        # tay phải
        draw_line(shoulder_2d_r, elbow_2d_r)
        draw_line(elbow_2d_r, wrist_2d_r)
        # thân
        draw_line(shoulder_2d_l, shoulder_2d_r)
        draw_line(shoulder_2d_l, hip_2d_l)
        draw_line(shoulder_2d_r, hip_2d_r)
        draw_line(hip_2d_l, hip_2d_r)
        temp_l_angles.append(l_angle)
        temp_r_angles.append(r_angle)
        # Duy trì buffer luôn là 30 phần tử (Trượt cửa sổ)
        if len(temp_l_angles) > window_size:
            temp_l_angles.pop(0)
            temp_r_angles.pop(0)
        # Khi đã tích lũy đủ 30 frame
        if len(temp_l_angles) == window_size:
            # --- CHUẨN HÓA THEO ĐÚNG TEMPLATE CỦA BẠN ---
            norm_l = [(a - MIN_ANGLE) / DENOM for a in temp_l_angles]
            norm_r = [(a - MIN_ANGLE) / DENOM for a in temp_r_angles]

            norm_l = [x if x >= 0.0 else 0.0 for x in norm_l]
            norm_r = [x if x >= 0.0 else 0.0 for x in norm_r]

            norm_l = [x if x <= 1.0 else 1.0 for x in norm_l]
            norm_r = [x if x <= 1.0 else 1.0 for x in norm_r]
            # Tính hiệu số (vận tốc) - 29 giá trị
            diff_l = [norm_l[i] - norm_l[i-1] for i in range(1, len(norm_l))]
            diff_r = [norm_r[i] - norm_r[i-1] for i in range(1, len(norm_r))]
            # Bù số 0 vào đầu để đủ 30 đặc trưng như lúc train
            diff_l = [0.0] + diff_l
            diff_r = [0.0] + diff_r
            # --- CHUẨN BỊ INPUT CHO MODEL (Sửa lại để thoát 25%) ---
            # Chuyển đổi list thành mảng numpy trước khi stack
            norm_l_arr = np.array(norm_l)
            norm_r_arr = np.array(norm_r)

            diff_l_arr = np.array(diff_l)
            diff_r_arr = np.array(diff_r)
            # Stack theo cấu trúc (30, 4)
            input_data = np.column_stack([norm_l_arr, norm_r_arr, diff_l_arr, diff_r_arr])
            # Ép kiểu float32 và thêm chiều batch (1, 30, 4)
            input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
            # --- DỰ ĐOÁN ---
            input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
            infer = model.signatures["serving_default"]
            # Gọi đúng tên cổng mà lỗi đã báo: keras_tensor_5
            output = infer(keras_tensor_5=input_tensor)
            # Lấy kết quả đầu ra
            # Trong lỗi của bạn ghi: Dict[['output_0', ...]] -> Key là 'output_0'
            prediction = output['output_0'].numpy()
            class_idx = np.argmax(prediction)
            prob = np.max(prediction)
            # --- HIỂN THỊ KẾT QUẢ ---
            # Vẽ nền đen để nhìn rõ thông số
            # --- HIỂN THỊ KẾT QUẢ NÂNG CAO ---
            # 1. Vẽ khung nền trong suốt phía trên để chữ nổi bật
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

            # 2. Xác định màu sắc và nội dung dựa trên class_idx
            if prob >= 0.7:  # Ngưỡng tin cậy
                if class_idx in [0, 1]:  # Nhãn SAI (LEN SAI, XUONG SAI)
                    color = (0, 0, 255)    # Đỏ rực
                    status_icon = "!!!"
                else:                    # Nhãn ĐÚNG (LEN DUNG, XUONG DUNG)
                    color = (0, 255, 0)    # Xanh lá sáng
                    status_icon = "SAFE"

                label_text = LABEL_MAP[class_idx].upper()
                
                # 3. Vẽ viền cảnh báo quanh màn hình nếu SAI
                if class_idx in [0, 1]:
                    cv2.rectangle(image, (0, 0), (w, h), color, 20) # Viền đỏ dày 20px

                # 4. Hiển thị nhãn chính (To và đậm nhất ở giữa)
                # Tính toán kích thước chữ để căn giữa
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(image, label_text, (text_x, 80),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 4, cv2.LINE_AA)

                # 5. Hiển thị % tin cậy nhỏ hơn ở góc
                prob_text = f"{prob*100:.0f}%"
                cv2.putText(image, prob_text, (w - 100, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                # Trạng thái đang phân tích
                cv2.putText(image, "ANALYZING...", (w // 2 - 100, 80),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    # Hiển thị frame
    #image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image = cv2.resize(image, (360, 640))
    cv2.imshow('Bench Press Live Monitoring', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()
