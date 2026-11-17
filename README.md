# HeartCheck

Web demo tiếng Việt cho đồ án dự đoán bệnh tim. API nạp `best_model_heart_disease.joblib` (Logistic Regression được notebook chọn) và áp dụng tiền xử lý cùng quy tắc với notebook trước khi dự đoán.

Notebook huấn luyện và so sánh ba thuật toán: [Compare_3_Algorithms_HeartDisease.ipynb](Compare_3_Algorithms_HeartDisease.ipynb). Nguồn và cách lấy dữ liệu được ghi trong [DATASET.md](DATASET.md). Chỉ cần notebook + dữ liệu nếu muốn tái huấn luyện; để chạy web, dùng model đã lưu.

## Chạy trên Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8000
```

Mở http://127.0.0.1:8000. API mô tả tại http://127.0.0.1:8000/docs.

## Chạy trên Linux / macOS

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn webapp.server:app --host 127.0.0.1 --port 8000
```

Giữ `best_model_heart_disease.joblib` ở thư mục gốc dự án. Môi trường này ghim scikit-learn 1.6.1 để tương thích với artifact xuất từ Colab. Không cần cài torch, torchvision hoặc facenet-pytorch cho web này.

Web phân loại nhãn 1 khi `predict_proba` trả về xác suất từ 0,50 trở lên. Đây là ngưỡng mặc định của Logistic Regression, chưa được điều chỉnh theo recall. Nếu nghiên cứu ngưỡng khác, hãy chọn ngưỡng trên validation rồi chỉ đánh giá một lần trên test.

Sau khi có kết quả, người dùng có thể chủ động bấm **Tải kết quả (.txt)**. File được tạo ngay trên trình duyệt, chỉ theo yêu cầu; web không lưu lịch sử dự đoán.

## Gợi ý trình bày đồ án

- Mở đầu bằng mục tiêu, dữ liệu 918 mẫu và giải thích rằng đầu ra là dự đoán nhãn của mô hình.
- Trình bày ba thuật toán, cách chia train/validation/test, cross-validation và quy tắc chọn Logistic Regression.
- Demo nhập liệu, giải thích xác suất đầu ra, rồi nêu các chỉ số test và giới hạn dữ liệu.
- Nhấn mạnh đây là nguyên mẫu học thuật, chưa được kiểm chứng lâm sàng và không thay thế bác sĩ.
