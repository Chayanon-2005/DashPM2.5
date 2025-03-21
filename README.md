# PM2.5 Forecast Dashboard

## Overview  
**PM2.5 Forecast Dashboard** เป็นเว็บแอปที่ใช้แสดงการพยากรณ์ค่าฝุ่น PM2.5 สำหรับ 7 วันข้างหน้า โดยใช้โมเดล Machine Learning ที่พัฒนาและฝึกด้วย **PyCaret** และแสดงผลผ่าน **Dash** และ **Plotly**  

## Features  
- พยากรณ์ค่า PM2.5 ล่วงหน้า 7 วัน  
- แสดงค่าดัชนีคุณภาพอากาศ (AQI)  
- กราฟแสดงแนวโน้มค่า PM2.5 และ AQI  
- ตารางข้อมูลพยากรณ์  

## Installation  

### Prerequisites  
- **Python 3.8** ขึ้นไป  
- **Virtual Environment** (แนะนำให้ใช้ `venv` หรือ `conda`)  

### Install Dependencies  

1. **Clone โปรเจคนี้**  
   ```sh
   git clone https://github.com/yourusername/pm25-dashboard.git
   cd pm25-dashboard
   ```
   
2. **สร้าง Virtual Environment และติดตั้ง dependencies**  
   ```sh
   python -m venv venv
   source venv/bin/activate   # สำหรับ macOS/Linux  
   venv\Scripts\activate      # สำหรับ Windows  
   pip install -r requirements.txt  
   ```

## Usage  

1. รันเซิร์ฟเวอร์ Dash  
   ```sh
   python main.py
   ```
   
2. เปิดเบราว์เซอร์และไปที่  
   ```
   http://127.0.0.1:8050/
   ```

## Project Structure  
```
PM25-Dashboard/
│── models/           # ไฟล์โมเดลที่ฝึกไว้  
│── data/             # ไฟล์ข้อมูล PM2.5  
│── assets/           # ไฟล์ CSS หรือรูปภาพเพิ่มเติม  
│── main.py           # โค้ดหลักของ Dash App  
│── requirements.txt  # รายการ dependencies  
│── README.md         # คำแนะนำการใช้งาน  
```

## Dependencies  
- Dash  
- Plotly  
- Pandas  
- PyCaret  

## License  
โปรเจคนี้ใช้ **MIT License** [(LICENSE)]  
