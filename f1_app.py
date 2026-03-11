import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="F1 AI Predictor", layout="wide")
st.title("🏎️ F1 AI Predictor - Dự đoán kết quả & Tính điểm")
st.markdown("Hệ thống dự đoán dựa trên dữ liệu phân hạng (Qualifying) và lịch sử thi đấu.")

# --- BƯỚC 1: TẢI DỮ LIỆU ---
@st.cache_data # Lưu dữ liệu vào bộ nhớ tạm để app chạy nhanh hơn
def load_data():
    races = pd.read_csv('races.csv')
    results = pd.read_csv('results.csv')
    drivers = pd.read_csv('drivers.csv')
    constructors = pd.read_csv('constructors.csv')
    qualifying = pd.read_csv('qualifying.csv')
    
    data = results.merge(races, on='raceId') \
                  .merge(drivers, on='driverId') \
                  .merge(constructors, on='constructorId', suffixes=('_race', '_constructor'))
    data = data.merge(qualifying[['raceId', 'driverId', 'q3']], on=['raceId', 'driverId'], how='left')
    
    # Tiền xử lý thời gian Q3
    def t_to_s(t):
        if pd.isna(t) or t == r"\N" or t == "": return None
        try: m, s = t.split(':'); return int(m) * 60 + float(s)
        except: return None
    
    data['q3_s'] = data['q3'].apply(t_to_s)
    data['gap'] = data.groupby('raceId')['q3_s'].transform(lambda x: x - x.min())
    data['gap'] = data['gap'].fillna(data['gap'].max() + 5)
    return data, races

data, races_df = load_data()

# --- BƯỚC 2: THANH ĐIỀU KHIỂN (SIDEBAR) ---
st.sidebar.header("Cài đặt đầu vào")
selected_year = st.sidebar.selectbox("Chọn năm mùa giải", sorted(data['year'].unique(), reverse=True))
available_races = data[data['year'] == selected_year]['name_race'].unique()
selected_race = st.sidebar.selectbox("Chọn chặng đua (Ví dụ: French GP, US GP)", available_races)

# --- BƯỚC 3: XỬ LÝ DỰ ĐOÁN ---
if st.sidebar.button("Bắt đầu dự đoán"):
    st.subheader(f"Kết quả dự đoán cho: {selected_race} ({selected_year})")
    
    # Tách dữ liệu Train/Test
    current_race_id = data[(data['year'] == selected_year) & (data['name_race'] == selected_race)]['raceId'].iloc[0]
    
    train = data[data['raceId'] != current_race_id]
    test = data[data['raceId'] == current_race_id].copy()
    
    # Mã hóa ID
    train['c_id'] = train['constructorId'].astype('category').cat.codes
    train['d_id'] = train['driverId'].astype('category').cat.codes
    test['c_id'] = test['constructorId'].astype('category').cat.codes
    test['d_id'] = test['driverId'].astype('category').cat.codes
    
    # Huấn luyện AI nhanh
    features = ['grid', 'gap', 'c_id', 'd_id']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train[features], train['positionOrder'])
    
    # Dự đoán
    test['ai_score'] = model.predict(test[features])
    test['rank'] = test['ai_score'].rank(method='min').astype(int)
    
    # Tính điểm FIA
    pts = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
    test['points'] = test['rank'].apply(lambda x: pts.get(x, 0))
    test['Tên'] = test['forename'] + " " + test['surname']
    
    # Hiển thị bảng kết quả
    res_display = test[['rank', 'Tên', 'grid', 'points']].sort_values('rank')
    res_display.columns = ['Hạng dự đoán', 'Tay đua', 'Vị trí xuất phát', 'Điểm số dự kiến']
    
    st.table(res_display.head(10))
    
    # Vẽ biểu đồ điểm số
    st.bar_chart(res_display.head(10).set_index('Tay đua')['Điểm số dự kiến'])
    st.success("Dự đoán hoàn tất! Bạn có thể chọn chặng khác ở thanh bên trái.")
else:
    st.info("Hãy chọn một chặng đua ở thanh bên trái và nhấn nút 'Bắt đầu dự đoán' để xem kết quả.")