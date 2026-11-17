from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "best_model_heart_disease.joblib"
PAGE_PATH = Path(__file__).resolve().parent / "index.html"
DECISION_THRESHOLD = 0.5

app = FastAPI(title="HeartCheck | Đánh giá nguy cơ bệnh tim", version="1.0.0")
model = joblib.load(MODEL_PATH)


class PatientInput(BaseModel):
    age: int = Field(ge=18, le=120, description="Tuổi")
    sex: Literal["M", "F"]
    chest_pain_type: Literal["TA", "ATA", "NAP", "ASY"]
    resting_bp: int = Field(ge=0, le=300, description="Huyết áp nghỉ (mmHg); 0 nếu không rõ")
    cholesterol: int = Field(ge=0, le=1000, description="Cholesterol (mg/dL); 0 nếu không rõ")
    fasting_bs: Literal[0, 1]
    resting_ecg: Literal["Normal", "ST", "LVH"]
    max_hr: int = Field(gt=0, le=250, description="Nhịp tim tối đa (bpm), phải lớn hơn 0")
    exercise_angina: Literal["Y", "N"]
    oldpeak: float = Field(ge=-10, le=20, description="ST depression")
    st_slope: Literal["Up", "Flat", "Down"]


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(PAGE_PATH)


@app.get("/api/health")
def health():
    return {"status": "ok", "model": "best_model_heart_disease.joblib"}


@app.post("/api/predict")
def predict(patient: PatientInput):
    row = pd.DataFrame([{
        "Age": patient.age,
        "Sex": patient.sex,
        "ChestPainType": patient.chest_pain_type,
        "RestingBP": patient.resting_bp,
        "Cholesterol": patient.cholesterol,
        "FastingBS": patient.fasting_bs,
        "RestingECG": patient.resting_ecg,
        "MaxHR": patient.max_hr,
        "ExerciseAngina": patient.exercise_angina,
        "Oldpeak": patient.oldpeak,
        "ST_Slope": patient.st_slope,
    }])
    # Match the notebook's clean() function before passing data to the pipeline.
    row["Cholesterol_missing"] = (row["Cholesterol"] == 0).astype(int)
    row["Cholesterol"] = row["Cholesterol"].replace(0, float("nan"))
    row["RestingBP"] = row["RestingBP"].replace(0, float("nan"))
    try:
        probability = float(model.predict_proba(row)[0, 1])
        label = int(probability >= DECISION_THRESHOLD)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Không thể chạy mô hình. Hãy kiểm tra phiên bản thư viện theo requirements.txt.") from exc
    return {
        "risk_probability": round(probability, 4),
        "prediction": label,
        "decision_threshold": DECISION_THRESHOLD,
        "threshold_description": "Gán nhãn 1 khi xác suất dự đoán >= 0.50; đây là ngưỡng mặc định, chưa được tối ưu theo recall.",
        "result": "Mô hình ghi nhận tín hiệu nguy cơ" if label else "Mô hình không ghi nhận tín hiệu nguy cơ",
        "model": "Logistic Regression",
        "metrics": {"test_roc_auc": 0.9374, "test_accuracy": 0.8986, "test_recall": 0.9211},
        "disclaimer": "Kết quả chỉ phục vụ học tập, không thay thế chẩn đoán hoặc tư vấn của bác sĩ.",
    }
