import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import random
from datetime import datetime

# Khởi tạo session state để lưu lịch sử
if 'model_history' not in st.session_state:
    st.session_state.model_history = []

# --- Load Data ---
@st.cache_data
def load_data():
    train = pd.read_csv('mnist_train.csv')
    test = pd.read_csv('mnist_test.csv')
    return train, test


st.title("Phân loại chữ số viết tay MNIST")
st.write("Chọn mô hình, huấn luyện và xem kết quả trực quan!")

# --- Sidebar ---
model_type = st.sidebar.selectbox("Chọn mô hình", ["SVM (RBF)", "Random Forest"])
n_train = st.sidebar.slider("Số lượng mẫu train", 1000, 60000, 50000, step=1000)
n_test = st.sidebar.slider("Số lượng mẫu test", 500, 10000, 10000, step=500)
num_samples = st.sidebar.slider("Số lượng ảnh mẫu muốn xem", 1, 10, 5, step=1)

# --- tien xu ly ---
train, test = load_data()
X_train = train.iloc[:n_train, 1:].values
y_train = train.iloc[:n_train, 0].values
X_test = test.iloc[:n_test, 1:].values
y_test = test.iloc[:n_test, 0].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Model selection với tham số tối ưu ---
if model_type == "SVM (RBF)":
    st.write("**Mô hình: SVM với kernel RBF**")
    # Tham số tối ưu từ repository: C=5, gamma=0.05
    model = SVC(kernel='rbf', C=5, gamma=0.05, probability=True, cache_size=2000)
elif model_type == "Random Forest":
    st.write("**Mô hình: Random Forest**")
    # Tăng số lượng cây và độ sâu để cải thiện độ chính xác
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

# --- Train & Predict ---
if st.button("Huấn luyện & Dự đoán"):
    with st.spinner("Đang huấn luyện..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = (y_pred == y_test).mean()

        # Lưu lịch sử
        history_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_type': model_type,
            'n_train': n_train,
            'n_test': n_test,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report
        }
        st.session_state.model_history.append(history_entry)

        st.success("Huấn luyện xong!")

        # --- Hiển thị độ chính xác tổng thể ---
        st.metric("Độ chính xác tổng thể", f"{accuracy:.4f}")

        # --- Visualization ---
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.title('Confusion Matrix')
        st.pyplot(fig)

        st.subheader("Báo cáo phân loại")
        st.dataframe(pd.DataFrame(report).transpose())

        # --- Hiển thị ảnh mẫu và dự đoán ---
        st.subheader("Dự đoán trên một số ảnh mẫu")

        # Đảm bảo có đủ các chữ số từ 0-9
        samples_per_digit = num_samples // 10
        indices = []
        for digit in range(10):
            digit_indices = np.where(y_test == digit)[0]
            if len(digit_indices) > 0:
                selected_indices = np.random.choice(digit_indices, size=min(samples_per_digit, len(digit_indices)),
                                                    replace=False)
                indices.extend(selected_indices)

        # Thêm các mẫu ngẫu nhiên nếu cần
        remaining = num_samples - len(indices)
        if remaining > 0:
            other_indices = np.random.choice(np.setdiff1d(np.arange(len(y_test)), indices), size=remaining,
                                             replace=False)
            indices.extend(other_indices)

        # Hiển thị ảnh theo grid 5 cột
        cols = st.columns(5)
        for i, idx in enumerate(indices):
            img = scaler.inverse_transform(X_test[idx].reshape(1, -1)).reshape(28, 28)
            img = np.clip(img, 0, 255).astype(np.uint8)
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            confidence = y_pred_proba[idx][pred_label]

            with cols[i % 5]:
                st.image(img, width=100, caption=f"True: {true_label} | Pred: {pred_label}")
                st.write(f"Độ tin cậy: {confidence:.2%}")

        st.info("""
        **Lưu ý:**
        - SVM RBF với C=5, gamma=0.05 cho độ chính xác khoảng 98.5%
        - Random Forest với 200 cây cho độ chính xác khoảng 93.7%
        - Tăng số lượng mẫu train sẽ cải thiện độ chính xác
        """)

# Thêm tab Lịch sử mô hình
st.markdown("---")
st.subheader("Lịch sử mô hình")

if st.session_state.model_history:
    # Tạo DataFrame từ lịch sử
    history_df = pd.DataFrame([
        {
            'Thời gian': entry['timestamp'],
            'Mô hình': entry['model_type'],
            'Số mẫu train': entry['n_train'],
            'Số mẫu test': entry['n_test'],
            'Độ chính xác': f"{entry['accuracy']:.4f}"
        }
        for entry in st.session_state.model_history
    ])

    # Hiển thị bảng lịch sử
    st.dataframe(history_df)

    # Nút xóa lịch sử
    if st.button("Xóa lịch sử"):
        st.session_state.model_history = []
        st.experimental_rerun()
else:
    st.info("Chưa có lịch sử huấn luyện nào được lưu.")

st.markdown("""
---
**Hướng dẫn sử dụng:**
- Chọn mô hình phân loại ở thanh bên trái
- Điều chỉnh số lượng mẫu train/test để thử nghiệm
- SVM RBF thường cho kết quả tốt hơn nhưng huấn luyện lâu hơn
- Random Forest huấn luyện nhanh hơn nhưng độ chính xác thấp hơn
- Nhấn nút **Huấn luyện & Dự đoán** để xem kết quả
- Lịch sử huấn luyện sẽ được tự động lưu lại
""")