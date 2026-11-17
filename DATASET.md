# Nguồn dữ liệu

Notebook `Compare_3_Algorithms_HeartDisease.ipynb` sử dụng file `heart.csv`, thuộc **Heart Failure Prediction Dataset** do fedesoriano đăng trên Kaggle:

- Trang dữ liệu: <https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction>
- Kích thước dùng trong notebook: 918 mẫu, 11 biến đầu vào và nhãn `HeartDisease`.
- Tải file CSV từ Kaggle, đặt tên `heart.csv` và đặt ở thư mục gốc repo trước khi chạy notebook. Trong Google Colab, notebook mở hộp thoại tải file lên.
- Kiểm tra điều khoản sử dụng/giấy phép hiện hành trên trang Kaggle trước khi phân phối lại file dữ liệu.

Web không cần file CSV để chạy: nó nạp trực tiếp model đã huấn luyện `best_model_heart_disease.joblib`.
