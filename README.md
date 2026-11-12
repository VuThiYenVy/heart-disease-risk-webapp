# HeartCheck

Web demo tiếng Việt cho đồ án dự đoán bệnh tim. API nạp `best_model_heart_disease.joblib` (Logistic Regression được notebook chọn) và áp dụng tiền xử lý cùng quy tắc với notebook trước khi dự đoán.

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

## Gợi ý trình bày đồ án

- Mở đầu bằng mục tiêu, dữ liệu 918 mẫu và giải thích rằng đầu ra là dự đoán nhãn của mô hình.
- Trình bày ba thuật toán, cách chia train/validation/test, cross-validation và quy tắc chọn Logistic Regression.
- Demo nhập liệu, giải thích xác suất đầu ra, rồi nêu các chỉ số test và giới hạn dữ liệu.
- Nhấn mạnh đây là nguyên mẫu học thuật, chưa được kiểm chứng lâm sàng và không thay thế bác sĩ.
