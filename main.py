import os
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from pycaret.regression import load_model, predict_model

# สร้างโฟลเดอร์สำหรับโปรเจคถ้ายังไม่มี
project_dirs = ["data", "assets", "models"]
for directory in project_dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)

# โหลดข้อมูลจากไฟล์ PM2.5.csv
data_file = "data/PM2.5.csv"
df = pd.read_csv(data_file)

# ตรวจสอบข้อมูลเบื้องต้น
df["timestamp"] = pd.to_datetime(df["timestamp"])

# โหลดโมเดลที่ฝึกไว้
model_path = "models/pm25_forecast_model.pkl"
best_model = load_model(model_path)

# สร้างข้อมูลพยากรณ์ล่วงหน้า 7 วัน
future_dates = pd.date_range(start=df["timestamp"].max(), periods=8, freq='D')[1:]
future_df = pd.DataFrame({"timestamp": future_dates})

# เติมค่าคุณลักษณะ (features) ที่ขาดหายไป
feature_cols = ["humidity", "pm_10", "temperature"]
for col in feature_cols:
    future_df[col] = df[col].mean()  # ใช้ค่าเฉลี่ยของข้อมูลเดิม

# ทำการพยากรณ์
future_predictions = predict_model(best_model, data=future_df)

# ตรวจสอบชื่อคอลัมน์ของผลลัพธ์
print("Columns in future_predictions:", future_predictions.columns)

# ใช้คอลัมน์ที่มีค่าพยากรณ์
prediction_col = "Label" if "Label" in future_predictions.columns else "prediction_label"
future_df["pm_2_5"] = future_predictions[prediction_col]

# รวมข้อมูลจริงและพยากรณ์
forecast_df = pd.concat([df, future_df])

# สร้างแอป Dash
app = dash.Dash(__name__)

# กราฟแสดงผลข้อมูล PM 2.5
fig = px.line(forecast_df, x="timestamp", y="pm_2_5", title="PM 2.5 Forecast", labels={"pm_2_5": "PM 2.5 Value"})

# Layout ของ Dashboard
app.layout = html.Div([
    html.H1("PM 2.5 Forecast Dashboard"),
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run_server(debug=True)
