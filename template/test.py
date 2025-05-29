import pandas as pd

# Đọc file CSV (thay đổi tên file nếu cần)
file_path = 'mnist_train.csv'  # Hoặc 'mnist_test.csv'
data = pd.read_csv(file_path)

# Đếm số lượng dòng (tương ứng với số lượng ảnh)
num_images = len(data)

# In kết quả
print(f"Số lượng ảnh trong file {file_path}: {num_images}")