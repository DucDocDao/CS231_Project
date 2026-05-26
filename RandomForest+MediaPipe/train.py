import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

print("Đang tải dữ liệu từ data_angles_4_classes.csv...")
df = pd.read_csv("data_angles_4_classes.csv")

# 1. TÁCH DỮ LIỆU THEO ĐÚNG TẬP TRAIN / TEST
# Lọc các dòng thuộc tập Train để học
train_df = df[df['split'] == 'train']
X_train = np.array(train_df[['left_armpit', 'right_armpit']])
y_train = np.array(train_df['label'])

# Lọc các dòng thuộc tập Test để thi (đánh giá)
test_df = df[df['split'] == 'test']
X_test = np.array(test_df[['left_armpit', 'right_armpit']])
y_test = np.array(test_df['label'])

print(f"📊 Số lượng frame Học (Train): {len(X_train)}")
print(f"📊 Số lượng frame Thi (Test): {len(X_test)}")

# 2. HUẤN LUYỆN MÔ HÌNH RANDOM FOREST
print("\n🤖 Đang huấn luyện Random Forest...")
# n_estimators=100 nghĩa là đang xây dựng 100 cây quyết định
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train) # Ép AI học trên tập Train

# 3. ĐÁNH GIÁ TRÊN TẬP TEST
print("\n🧪 Đang kiểm tra trên tập Test...")
y_pred = model.predict(X_test) # Cho AI làm bài thi trên tập Test

print("\n" + "="*55)
print("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH (4 TRẠNG THÁI)")
print("="*55)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy (Độ chính xác trên tập Test): {acc*100:.2f}%\n")

target_names = ['Len_Dung (L1)', 'Xuong_Dung (L2)', 'Len_Sai (L3)', 'Xuong_Sai (L4)']
report = classification_report(y_test, y_pred, target_names=target_names)
print(report)

# 4. LƯU BỘ NÃO AI
with open("model_angles_4_classes.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n Đã lưu bộ não dự đoán 4 trạng thái thành công: model_angles_4_classes.pkl")